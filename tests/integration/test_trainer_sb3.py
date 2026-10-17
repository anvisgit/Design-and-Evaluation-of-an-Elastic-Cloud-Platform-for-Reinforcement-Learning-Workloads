import json
import threading
from pathlib import Path
import boto3
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")
from aws_clients import RunStore, Settings
from callbacks import RunState
from common import db
from config_schema import TrainingConfig
from main import Worker
import trainer

HP = {"n_steps": 64, "batch_size": 32, "policy": "MlpPolicy"}

# Builds a small training config
def cfg(steps=2000, **over):
    base = {"experiment_id": "e", "name": "t", "algo": "PPO", "env_id": "CartPole-v1", "seed": 7,
            "total_timesteps": steps, "n_envs": 1, "hyperparams": HP,
            "logging": {"metrics_every_steps": 200, "checkpoint_every_steps": 1000,
                        "eval_every_steps": 1000, "eval_episodes": 2}}
    return TrainingConfig.model_validate({**base, **over})

# Gives settings, a store and a running run
@pytest.fixture
def env(aws, tmp_path, monkeypatch):
    s = Settings(workdir=str(tmp_path), queue_url=aws["queue_url"], bucket="test-bucket", local_mode=True,
                 retry_delay_s=0)
    store = RunStore(s)
    now = db.now_iso()
    db.runs_table().put_item(Item={"run_id": "r1", "experiment_id": "e", "owner": "alice", "status": "RUNNING",
                                   "attempt": 1, "created_at": now, "last_heartbeat": now,
                                   "instance_id": s.instance_id, "cancel_requested": False})
    return s, store, tmp_path

# Checks that training saves metrics, checkpoints and the model
def test_train_writes_metrics_checkpoints_and_model(env):
    s, store, tmp = env
    res = trainer.train(cfg(), "r1", store, RunState(), threading.Event(), tmp)
    assert res.status == "finished" and res.steps >= 2000 and res.final_reward is not None
    rows = db.query_all(db.metrics_table(), KeyConditionExpression=boto3.dynamodb.conditions.Key("run_id").eq("r1"))
    assert len(rows) >= 5 and all("fps" in r and "expires_at" in r for r in rows)
    assert any("ep_rew_mean" in r for r in rows)
    keys = [o["Key"] for o in boto3.client("s3").list_objects_v2(Bucket="test-bucket")["Contents"]]
    assert "models/r1/final_model.zip" in keys and any(k.startswith("checkpoints/r1/step_") for k in keys)
    run = db.get_run("r1")
    assert run["latest_checkpoint_s3_uri"] and run["best_reward"] is not None

# Checks that cancel stops training and saves a checkpoint
def test_cancel_flag_stops_training_and_checkpoints(env):
    s, store, tmp = env
    db.update_run("r1", {"cancel_requested": True})
    state = RunState()
    res = trainer.train(cfg(20_000), "r1", store, state, threading.Event(), tmp)
    assert res.status == "cancelled" and res.steps < 20_000
    assert db.get_run("r1")["latest_checkpoint_s3_uri"]

# Checks that an interrupted run can resume
def test_spot_interrupt_then_resume_from_checkpoint(env):
    s, store, tmp = env
    ev = threading.Event()
    ev.set()
    res1 = trainer.train(cfg(), "r1", store, RunState(), ev, tmp)
    assert res1.status == "interrupted"
    uri = db.get_run("r1")["latest_checkpoint_s3_uri"]
    ckpt = store.download(uri, tmp / "resume.zip")
    res2 = trainer.train(cfg(), "r1", store, RunState(), threading.Event(), tmp, resume_from=ckpt)
    assert res2.status == "finished" and res2.steps >= 2000

# Checks that the time limit stops training
def test_runtime_limit(env):
    s, store, tmp = env
    c = cfg(200_000)
    c.resources.max_runtime_minutes = 1
    state = RunState()
    import callbacks
    orig = callbacks.ControlCallback.__init__
    callbacks.ControlCallback.__init__ = lambda self, st, rid, max_s, *a, **k: orig(self, st, rid, 0.0, *a, **k)
    try:
        res = trainer.train(c, "r1", store, state, threading.Event(), tmp)
    finally:
        callbacks.ControlCallback.__init__ = orig
    assert res.status == "timeout"

# Checks that the other algorithms train
@pytest.mark.parametrize("algo,env_id,hp", [
    ("A2C", "CartPole-v1", {"n_steps": 16}),
    ("DQN", "CartPole-v1", {"learning_starts": 100, "buffer_size": 5000, "train_freq": 4}),
    ("SAC", "Pendulum-v1", {"learning_starts": 100, "buffer_size": 5000}),
])
def test_other_algorithms_train(env, algo, env_id, hp):
    s, store, tmp = env
    res = trainer.train(cfg(1200, algo=algo, env_id=env_id, hyperparams=hp), "r1", store, RunState(),
                        threading.Event(), tmp)
    assert res.status == "finished"

# Checks the whole worker flow from a message
def test_full_worker_message_flow(aws, env):
    s, store, tmp = env
    boto3.client("s3").put_object(Bucket="test-bucket", Key="configs/r2/config.json",
                                  Body=cfg().model_dump_json().encode())
    now = db.now_iso()
    db.runs_table().put_item(Item={"run_id": "r2", "experiment_id": "e", "owner": "alice", "status": "QUEUED",
                                   "attempt": 1, "created_at": now, "last_heartbeat": now,
                                   "config_s3_uri": "s3://test-bucket/configs/r2/config.json", "cancel_requested": False})
    boto3.client("sqs").send_message(QueueUrl=aws["queue_url"], MessageBody=json.dumps({"run_id": "r2"}))
    w = Worker(s, store)
    assert w.poll_once(wait=0) is True
    run = db.get_run("r2")
    assert run["status"] == "SUCCEEDED" and run["model_s3_uri"] and run["library_versions"]["torch"]
