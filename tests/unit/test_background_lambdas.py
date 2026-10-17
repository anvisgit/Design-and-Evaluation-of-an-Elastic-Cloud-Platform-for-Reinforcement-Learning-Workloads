import json
from datetime import timedelta
import boto3
import pytest
from common import db
from notifier import handler as notifier
from scaler.handler import compute_desired
from watchdog import handler as watchdog
from conftest import Ctx

# Checks the worker count math
@pytest.mark.parametrize("visible,in_flight,current,mx,expected", [
    (0, 0, 0, 5, 0), (3, 0, 0, 5, 3), (10, 2, 0, 5, 5), (1, 1, 4, 5, 4), (0, 0, 2, 5, 2), (2, 1, 1, 5, 3),
])
def test_scaler_math(visible, in_flight, current, mx, expected):
    assert compute_desired(visible, in_flight, current, mx) == expected

# Writes a run into the table
def _run(rid, status="RUNNING", hb_age_min=0, started_min=1, attempt=1, **kw):
    t = db.now()
    item = {"run_id": rid, "experiment_id": "e", "owner": "alice", "status": status, "attempt": attempt,
            "config_s3_uri": "s3://test-bucket/c", "created_at": db.now_iso(),
            "last_heartbeat": (t - timedelta(minutes=hb_age_min)).isoformat(timespec="seconds"),
            "started_at": (t - timedelta(minutes=started_min)).isoformat(timespec="seconds"),
            "max_runtime_minutes": 120, **kw}
    db.runs_table().put_item(Item=db.to_ddb(item))
    return db.get_run(rid)

# Checks that a healthy run is left alone
def test_watchdog_healthy_untouched(aws):
    watchdog.handler({}, Ctx())
    r = _run("ok", hb_age_min=1)
    assert watchdog.check_run(r) == "ok"

# Checks requeue first and fail after
def test_watchdog_requeues_then_fails(aws):
    r = _run("a", hb_age_min=11, attempt=1)
    assert watchdog.check_run(r) == "requeued"
    run = db.get_run("a")
    assert run["status"] == "REQUEUED" and run["attempt"] == 2
    msgs = boto3.client("sqs").receive_message(QueueUrl=aws["queue_url"])["Messages"]
    assert json.loads(msgs[0]["Body"])["attempt"] == 2
    r2 = _run("b", hb_age_min=11, attempt=2)
    assert watchdog.check_run(r2) == "failed_heartbeat"
    assert db.get_run("b")["status"] == "FAILED" and db.get_run("b")["error"] == "heartbeat_lost"

# Checks the max runtime rule
def test_watchdog_runtime_limit(aws):
    r = _run("slow", hb_age_min=0, started_min=136)
    assert watchdog.check_run(r) == "failed_runtime"
    assert db.get_run("slow")["error"] == "max_runtime_exceeded"

# Checks the full watchdog and stale workers
def test_watchdog_handler_and_stale_workers(aws):
    _run("dead", hb_age_min=30)
    t = (db.now() - timedelta(minutes=30)).isoformat()
    db.workers_table().put_item(Item={"instance_id": "i-old", "status": "IDLE", "last_seen": t})
    db.workers_table().put_item(Item={"instance_id": "i-new", "status": "IDLE", "last_seen": db.now_iso()})
    out = watchdog.handler({}, Ctx())
    assert out["actions"] == {"requeued": 1} and out["stale_workers"] == 1
    assert db.workers_table().get_item(Key={"instance_id": "i-old"})["Item"]["status"] == "STALE"

# Builds a fake stream record
def _stream_rec(old, new, name="MODIFY"):
    from boto3.dynamodb.types import TypeSerializer
    ser = TypeSerializer()
    f = lambda d: {k: ser.serialize(v) for k, v in db.to_ddb(d).items()}
    return {"eventName": name, "dynamodb": {"OldImage": f(old), "NewImage": f(new)}}

# Checks that only ended runs send a message
def test_notifier_publishes_on_terminal_only(aws, monkeypatch):
    sns = boto3.client("sns")
    arn = sns.create_topic(Name="rl_notifications")["TopicArn"]
    monkeypatch.setenv("TOPIC_ARN", arn)
    monkeypatch.setenv("API_URL", "https://api.example/prod/")
    q = boto3.client("sqs").create_queue(QueueName="sub")["QueueUrl"]
    sns.subscribe(TopicArn=arn, Protocol="sqs", Endpoint="arn:aws:sqs:us-east-1:123456789012:sub",
                  Attributes={"RawMessageDelivery": "false"})
    base = {"run_id": "r1", "experiment_id": "e", "name": "n", "algo": "PPO", "env_id": "CartPole-v1",
            "started_at": "2026-01-01T00:00:00+00:00", "ended_at": "2026-01-01T00:10:00+00:00"}
    recs = [
        _stream_rec({**base, "status": "QUEUED"}, {**base, "status": "RUNNING"}),
        _stream_rec({**base, "status": "RUNNING"}, {**base, "status": "SUCCEEDED", "final_reward": 475.5}),
        _stream_rec({**base, "status": "RUNNING"}, {**base, "status": "FAILED", "error": "boom"}),
        _stream_rec({**base, "status": "FAILED"}, {**base, "status": "FAILED"}),
        _stream_rec({**base, "status": "A"}, {**base, "status": "SUCCEEDED"}, name="INSERT"),
    ]
    assert notifier.handler({"Records": recs}, None) == {"sent": 2}
    msgs = boto3.client("sqs").receive_message(QueueUrl=q, MaxNumberOfMessages=10).get("Messages", [])
    bodies = [json.loads(m["Body"]) for m in msgs]
    assert len(bodies) == 2
    assert "Duration: 600s" in bodies[0]["Message"] and "475.5" in bodies[0]["Message"]
    assert bodies[0]["MessageAttributes"]["event"]["Value"] == "run_succeeded"
    assert "Error: boom" in bodies[1]["Message"] and "https://api.example/prod/runs/r1" in bodies[1]["Message"]
