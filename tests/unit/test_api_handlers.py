import json
import boto3
import pytest
from common import db
from create_experiment.handler import handler as create_experiment
from list_experiments.handler import handler as list_experiments
from submit_run.handler import handler as submit_run
from get_run.handler import handler as get_run
from list_runs.handler import handler as list_runs
from cancel_run.handler import handler as cancel_run
from get_metrics.handler import handler as get_metrics
from compare_runs.handler import handler as compare_runs
from get_artifact_url.handler import handler as get_artifact_url

# Builds a run config
def run_cfg(exp_id, **over):
    return {"experiment_id": exp_id, "name": "r", "algo": "PPO", "env_id": "CartPole-v1",
            "seed": 1, "total_timesteps": 5000, **over}

# Creates an experiment
@pytest.fixture
def exp(aws, call):
    code, body = call(create_experiment, body={"name": "exp", "tags": ["a"]})
    assert code == 201
    return body["data"]["experiment_id"]

# Checks creating and listing experiments
def test_create_and_list_experiments(aws, call, exp):
    code, body = call(list_experiments)
    assert code == 200 and [e["experiment_id"] for e in body["data"]["items"]] == [exp]
    assert call(list_experiments, owner="bob")[1]["data"]["items"] == []
    code, body = call(create_experiment, body={"name": ""})
    assert code == 400 and body["error"]["code"] == "VALIDATION_ERROR"
    assert call(create_experiment, body="{bad")[0] == 400

# Checks a normal run submit
def test_submit_run_happy_path(aws, call, exp):
    code, body = call(submit_run, body=run_cfg(exp))
    assert code == 202 and body["data"]["status"] == "QUEUED"
    rid = body["data"]["run_id"]
    run = db.get_run(rid)
    assert run["status"] == "QUEUED" and run["attempt"] == 1 and run["owner"] == "alice"
    msgs = boto3.client("sqs").receive_message(QueueUrl=aws["queue_url"])["Messages"]
    assert json.loads(msgs[0]["Body"])["run_id"] == rid
    cfg = json.loads(boto3.client("s3").get_object(Bucket="test-bucket", Key=f"configs/{rid}/config.json")["Body"].read())
    assert cfg["algo"] == "PPO"
    assert db.get_experiment(exp)["run_count"] == 1

# Checks bad input and wrong owner on submit
def test_submit_validation_and_ownership(aws, call, exp):
    code, body = call(submit_run, body=run_cfg(exp, env_id="Nope-v0"))
    assert code == 400 and "env_id" in body["error"]["message"]
    assert call(submit_run, body=run_cfg("missing"))[0] == 404
    assert call(submit_run, body=run_cfg(exp), owner="mallory")[0] == 404

# Checks the active run limit
def test_quota(aws, call, exp):
    for _ in range(5):
        assert call(submit_run, body=run_cfg(exp))[0] == 202
    code, body = call(submit_run, body=run_cfg(exp))
    assert code == 429 and body["error"]["code"] == "QUOTA_EXCEEDED"

# Checks that the same key gives the same run
def test_idempotency_key(aws, call, exp):
    h = {"Idempotency-Key": "abc"}
    a = call(submit_run, body=run_cfg(exp), headers=h)[1]["data"]
    b = call(submit_run, body=run_cfg(exp), headers=h)[1]["data"]
    assert a["run_id"] == b["run_id"] and b["idempotent_replay"]
    assert boto3.client("sqs").get_queue_attributes(
        QueueUrl=aws["queue_url"], AttributeNames=["ApproximateNumberOfMessages"])["Attributes"]["ApproximateNumberOfMessages"] == "1"

# Checks getting and listing runs
def test_get_and_list_runs(aws, call, exp):
    rid = call(submit_run, body=run_cfg(exp))[1]["data"]["run_id"]
    code, body = call(get_run, path={"run_id": rid})
    assert code == 200 and body["data"]["run_id"] == rid
    assert call(get_run, path={"run_id": rid}, owner="bob")[0] == 404
    code, body = call(list_runs, query={"experiment_id": exp})
    assert [r["run_id"] for r in body["data"]["items"]] == [rid]
    assert "hyperparams" not in body["data"]["items"][0]
    assert call(list_runs, query={"status": "QUEUED"})[1]["data"]["items"][0]["run_id"] == rid
    assert call(list_runs, query={"status": "SUCCEEDED"})[1]["data"]["items"] == []
    assert call(list_runs, query={"status": "BOGUS"})[0] == 400

# Checks paging of runs
def test_list_runs_pagination(aws, call, exp):
    for i in range(3):
        call(submit_run, body=run_cfg(exp))
    code, body = call(list_runs, query={"limit": "2"})
    assert len(body["data"]["items"]) == 2 and body["data"]["next_cursor"]
    code, body2 = call(list_runs, query={"limit": "2", "cursor": body["data"]["next_cursor"]})
    assert len(body2["data"]["items"]) == 1

# Checks cancel for queued and running runs
def test_cancel_queued_and_running(aws, call, exp):
    q = call(submit_run, body=run_cfg(exp))[1]["data"]["run_id"]
    code, body = call(cancel_run, path={"run_id": q})
    assert code == 202 and body["data"]["status"] == "CANCELLED"
    assert call(cancel_run, path={"run_id": q})[0] == 409
    r = call(submit_run, body=run_cfg(exp))[1]["data"]["run_id"]
    db.update_run(r, {"status": "RUNNING"})
    code, body = call(cancel_run, path={"run_id": r})
    assert body["data"]["status"] == "CANCEL_REQUESTED"
    run = db.get_run(r)
    assert run["cancel_requested"] is True and run["status"] == "CANCEL_REQUESTED"

# Writes metric points for a run
def _metrics(rid, points):
    t = db.metrics_table()
    for step, v in points:
        t.put_item(Item=db.to_ddb({"run_id": rid, "step": step, "ep_rew_mean": v, "fps": 100.0}))

# Checks the step range and paging of metrics
def test_metrics_range_and_pagination(aws, call, exp):
    rid = call(submit_run, body=run_cfg(exp))[1]["data"]["run_id"]
    _metrics(rid, [(s, s / 100) for s in range(1000, 6000, 1000)])
    code, body = call(get_metrics, path={"run_id": rid}, query={"from_step": "2000", "to_step": "4000"})
    assert [m["step"] for m in body["data"]["items"]] == [2000, 3000, 4000]
    code, body = call(get_metrics, path={"run_id": rid}, query={"limit": "2"})
    assert len(body["data"]["items"]) == 2 and body["data"]["next_cursor"]
    code, body = call(get_metrics, path={"run_id": rid}, query={"limit": "2", "cursor": body["data"]["next_cursor"]})
    assert [m["step"] for m in body["data"]["items"]] == [3000, 4000]

# Checks the run comparison
def test_compare_runs(aws, call, exp):
    ids = [call(submit_run, body=run_cfg(exp, name=f"r{i}"))[1]["data"]["run_id"] for i in range(3)]
    for i, rid in enumerate(ids):
        _metrics(rid, [(s, (i + 1) * s / 1000) for s in range(1000, 11000, 1000)])
    code, body = call(compare_runs, path={"experiment_id": exp},
                      body={"run_ids": ids, "metric": "ep_rew_mean", "threshold": 5})
    assert code == 200, body
    d = body["data"]
    assert len(d["steps"]) == 200 and all(len(v) == 200 for v in d["series"].values())
    assert d["ranking"] == ids[::-1]
    best = {s["run_id"]: s for s in d["summary"]}
    assert best[ids[2]]["final"] == 30 and best[ids[2]]["steps_to_threshold"] == 2000
    assert best[ids[0]]["steps_to_threshold"] is None or best[ids[0]]["steps_to_threshold"] >= 5000
    assert call(compare_runs, path={"experiment_id": exp}, body={"run_ids": ["nope"]})[0] == 404
    assert call(compare_runs, path={"experiment_id": exp}, body={"run_ids": ids, "metric": "x"})[0] == 400

# Checks the download links
def test_artifact_urls(aws, call, exp):
    rid = call(submit_run, body=run_cfg(exp))[1]["data"]["run_id"]
    assert call(get_artifact_url, path={"run_id": rid, "kind": "model"})[0] == 404
    assert call(get_artifact_url, path={"run_id": rid, "kind": "weird"})[0] == 400
    code, body = call(get_artifact_url, path={"run_id": rid, "kind": "config"})
    assert code == 200 and body["data"]["url"].startswith("https://") and body["data"]["expires_in"] == 900
    boto3.client("s3").put_object(Bucket="test-bucket", Key=f"models/{rid}/final_model.zip", Body=b"x")
    db.update_run(rid, {"model_s3_uri": f"s3://test-bucket/models/{rid}/final_model.zip"})
    assert call(get_artifact_url, path={"run_id": rid, "kind": "model"})[0] == 200
