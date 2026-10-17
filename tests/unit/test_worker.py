import json
import threading
from pathlib import Path
from types import SimpleNamespace
import boto3
import pytest
from aws_clients import RunStore, Settings
from common import db
from main import Worker
from spot import SpotWatcher

CFG = {"experiment_id": "e", "name": "n", "algo": "PPO", "env_id": "CartPole-v1", "seed": 1,
       "total_timesteps": 2000}

# Writes a run, its config and its message
def make_run(aws, run_id="r1", status="QUEUED", **extra):
    boto3.client("s3").put_object(Bucket="test-bucket", Key=f"configs/{run_id}/config.json",
                                  Body=json.dumps(CFG).encode())
    now = db.now_iso()
    item = {"run_id": run_id, "experiment_id": "e", "owner": "alice", "status": status, "attempt": 1,
            "config_s3_uri": f"s3://test-bucket/configs/{run_id}/config.json",
            "created_at": now, "last_heartbeat": now, "cancel_requested": False, **extra}
    db.runs_table().put_item(Item=db.to_ddb(item))
    boto3.client("sqs").send_message(QueueUrl=aws["queue_url"], MessageBody=json.dumps(
        {"run_id": run_id, "experiment_id": "e", "config_s3_uri": item["config_s3_uri"], "attempt": 1}))

# Receives one queue message
def receive(aws, wait=0):
    m = boto3.client("sqs").receive_message(QueueUrl=aws["queue_url"], MaxNumberOfMessages=1,
                                            WaitTimeSeconds=wait, AttributeNames=["ApproximateReceiveCount"])
    return m["Messages"][0]

# Returns the number of messages in the queue
def queue_depth(aws):
    a = boto3.client("sqs").get_queue_attributes(QueueUrl=aws["queue_url"], AttributeNames=["All"])["Attributes"]
    return int(a["ApproximateNumberOfMessages"]) + int(a["ApproximateNumberOfMessagesNotVisible"])

class FakeTrainer:
    # Sets the fake result or error
    def __init__(self, status="finished", exc=None):
        self.status, self.exc, self.calls = status, exc, []
    # Pretends to train and returns a result
    def train(self, cfg, run_id, store, state, interrupt, workdir, resume_from=None):
        self.calls.append({"run_id": run_id, "resume": resume_from, "algo": cfg.algo})
        if self.exc:
            raise self.exc
        state.current_step = 2000
        return SimpleNamespace(status=self.status, steps=2000, final_reward=195.5, best_reward=200.0,
                               model_uri=f"s3://test-bucket/models/{run_id}/final_model.zip")

# Gives a worker with a fake trainer
@pytest.fixture
def worker(aws, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_MODE", "1")
    s = Settings(workdir=str(tmp_path), queue_url=aws["queue_url"], bucket="test-bucket", local_mode=True,
                 retry_delay_s=0)
    t = FakeTrainer()
    w = Worker(s, RunStore(s), t)
    w.heartbeat_interval = 0.05
    return w

# Checks a successful run
def test_success_path(aws, worker):
    make_run(aws)
    msg = receive(aws)
    assert worker.handle_message(msg) == "succeeded"
    run = db.get_run("r1")
    assert run["status"] == "SUCCEEDED" and run["final_reward"] == 195.5
    assert run["instance_id"] == worker.s.instance_id and "git_sha" in run and run["library_versions"]["python"]
    assert run["model_s3_uri"].endswith("final_model.zip")
    assert queue_depth(aws) == 0
    assert boto3.client("s3").head_object(Bucket="test-bucket", Key="logs/r1/train.log")
    assert db.workers_table().get_item(Key={"instance_id": worker.s.instance_id})["Item"]["status"] == "IDLE"

# Checks that dead messages are dropped
def test_terminal_or_missing_run_message_dropped(aws, worker):
    make_run(aws, status="SUCCEEDED")
    assert worker.handle_message(receive(aws)) == "dropped"
    assert queue_depth(aws) == 0
    boto3.client("sqs").send_message(QueueUrl=aws["queue_url"], MessageBody=json.dumps({"run_id": "ghost"}))
    assert worker.handle_message(receive(aws)) == "dropped"

# Checks that a cancelled run never trains
def test_cancelled_while_queued_never_trains(aws, worker):
    make_run(aws, status="CANCELLED")
    assert worker.handle_message(receive(aws)) == "dropped"
    assert worker.trainer.calls == []

# Checks cancel before the start
def test_cancel_requested_before_start(aws, worker):
    make_run(aws, status="CANCEL_REQUESTED", cancel_requested=True)
    assert worker.handle_message(receive(aws)) == "cancelled"
    assert db.get_run("r1")["status"] == "CANCELLED" and worker.trainer.calls == []

# Checks that a duplicate message is ignored
def test_duplicate_delivery_guard(aws, worker):
    make_run(aws)
    msg = receive(aws)
    db.update_run("r1", {"status": "RUNNING", "instance_id": "other", "last_heartbeat": db.now_iso()})
    assert worker.handle_message(msg) == "duplicate"
    assert worker.trainer.calls == [] and db.get_run("r1")["instance_id"] == "other"

# Checks that a dead worker run resumes
def test_takeover_of_dead_worker_resumes_from_checkpoint(aws, worker):
    boto3.client("s3").put_object(Bucket="test-bucket", Key="checkpoints/r1/step_1000.zip", Body=b"zip")
    make_run(aws, status="RUNNING", instance_id="dead", last_heartbeat="2020-01-01T00:00:00+00:00",
             latest_checkpoint_s3_uri="s3://test-bucket/checkpoints/r1/step_1000.zip")
    assert worker.handle_message(receive(aws)) == "succeeded"
    assert worker.trainer.calls[0]["resume"] is not None
    assert Path(worker.trainer.calls[0]["resume"]).read_bytes() == b"zip"

# Checks retries then failure
def test_exception_retries_then_fails_and_goes_to_dlq(aws, worker):
    worker.trainer.exc = RuntimeError("boom")
    make_run(aws)
    sqs = boto3.client("sqs")
    sqs.set_queue_attributes(QueueUrl=aws["queue_url"], Attributes={"VisibilityTimeout": "0"})
    assert worker.handle_message(receive(aws)) == "retry"
    run = db.get_run("r1")
    assert run["status"] == "REQUEUED" and "boom" in run["error"] and queue_depth(aws) == 1
    assert worker.handle_message(receive(aws)) == "retry"
    assert worker.handle_message(receive(aws)) == "failed"
    run = db.get_run("r1")
    assert run["status"] == "FAILED" and run["attempt"] == 3 and "RuntimeError" in run["error"]
    assert queue_depth(aws) == 1

# Checks that a bad config fails at once
def test_bad_config_fails_immediately(aws, worker):
    make_run(aws)
    boto3.client("s3").put_object(Bucket="test-bucket", Key="configs/r1/config.json",
                                  Body=json.dumps({**CFG, "env_id": "Evil-v0"}).encode())
    assert worker.handle_message(receive(aws)) == "failed"
    assert db.get_run("r1")["error"].startswith("invalid_config") and queue_depth(aws) == 0

# Checks the final status for each stop reason
@pytest.mark.parametrize("status,final,deleted", [
    ("cancelled", "CANCELLED", True), ("timeout", "FAILED", True)])
def test_trainer_stop_reasons(aws, worker, status, final, deleted):
    worker.trainer.status = status
    make_run(aws)
    worker.handle_message(receive(aws))
    assert db.get_run("r1")["status"] == final and (queue_depth(aws) == 0) == deleted

# Checks the spot interruption flow
def test_spot_interruption_requeues_and_exits(aws, worker):
    worker.trainer.status = "interrupted"
    make_run(aws)
    assert worker.handle_message(receive(aws)) == "requeued"
    run = db.get_run("r1")
    assert run["status"] == "REQUEUED" and run["attempt"] == 1
    assert worker.exit_requested and queue_depth(aws) == 1

# Checks that the spot watcher sets the event
def test_spot_watcher_sets_event():
    ev, calls = threading.Event(), iter([None, None, '{"action":"terminate"}'])
    w = SpotWatcher(ev, interval=0.01, getter=lambda p: next(calls))
    w.start()
    assert ev.wait(2) and w.is_alive() or ev.is_set()
    w.stop()

# Checks one heartbeat
def test_heartbeat_extends_visibility_and_updates_run(aws, worker):
    from main import Heartbeat
    from runstate import RunState
    make_run(aws, status="RUNNING", instance_id=worker.s.instance_id)
    msg = receive(aws)
    st = RunState(current_step=777)
    Heartbeat(worker.store, "r1", msg["ReceiptHandle"], st, Path("/nonexistent")).beat()
    run = db.get_run("r1")
    assert run["current_step"] == 777
    assert db.workers_table().get_item(Key={"instance_id": worker.s.instance_id})["Item"]["status"] == "BUSY"

# Checks that an idle worker shuts down
def test_idle_shutdown(aws, worker, monkeypatch):
    worker.s.idle_timeout_s = 0
    monkeypatch.setattr(worker, "poll_once", lambda wait=20: False)
    worker.run_forever()
    assert db.workers_table().get_item(Key={"instance_id": worker.s.instance_id})["Item"]["status"] == "TERMINATING"

# Checks that a failing job ends in the DLQ
def test_message_moves_to_dlq_after_three_failed_receives(aws, worker):
    sqs = boto3.client("sqs")
    dlq = sqs.create_queue(QueueName="training_jobs_dlq")["QueueUrl"]
    dlq_arn = sqs.get_queue_attributes(QueueUrl=dlq, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
    sqs.set_queue_attributes(QueueUrl=aws["queue_url"], Attributes={
        "VisibilityTimeout": "0",
        "RedrivePolicy": json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": 3})})
    worker.trainer.exc = RuntimeError("always broken")
    make_run(aws)
    outcomes = [worker.handle_message(receive(aws)) for _ in range(3)]
    assert outcomes == ["retry", "retry", "failed"]
    assert sqs.receive_message(QueueUrl=aws["queue_url"], WaitTimeSeconds=0).get("Messages") is None
    in_dlq = sqs.receive_message(QueueUrl=dlq, WaitTimeSeconds=0)["Messages"]
    assert json.loads(in_dlq[0]["Body"])["run_id"] == "r1"
    assert db.get_run("r1")["status"] == "FAILED"
