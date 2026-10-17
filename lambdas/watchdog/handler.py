import os
from botocore.exceptions import ClientError
from common import db
from common.log import logger

HEARTBEAT_TIMEOUT_S = 10 * 60
RUNTIME_GRACE_S = 15 * 60

# Updates a run only if it did not change since we read it
def _guarded_update(run, fields):
    try:
        db.update_run(run["run_id"], fields, condition="#s = :s AND #h = :h",
                      cond_names={"#s": "status", "#h": "last_heartbeat"},
                      cond_values={":s": run["status"], ":h": run["last_heartbeat"]})
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise

# Fails or requeues a run that is stuck
def check_run(run: dict) -> str:
    limit_s = int(run.get("max_runtime_minutes", 120)) * 60 + RUNTIME_GRACE_S
    if run.get("started_at") and db.age_seconds(run["started_at"]) > limit_s:
        ok = _guarded_update(run, {"status": "FAILED", "ended_at": db.now_iso(),
                                   "error": "max_runtime_exceeded"})
        return "failed_runtime" if ok else "skipped"
    if db.age_seconds(run.get("last_heartbeat")) <= HEARTBEAT_TIMEOUT_S:
        return "ok"
    attempt = int(run.get("attempt", 1)) + 1
    if attempt < db.MAX_ATTEMPTS:
        if not _guarded_update(run, {"status": "REQUEUED", "attempt": attempt,
                                     "last_heartbeat": db.now_iso()}):
            return "skipped"
        db.sqs().send_message(QueueUrl=os.environ["QUEUE_URL"],
                              MessageBody=db.queue_message(run, attempt))
        return "requeued"
    ok = _guarded_update(run, {"status": "FAILED", "attempt": attempt, "ended_at": db.now_iso(),
                               "error": "heartbeat_lost"})
    return "failed_heartbeat" if ok else "skipped"

# Marks workers that went quiet as stale
def mark_stale_workers() -> int:
    tbl, n = db.workers_table(), 0
    for w in db.from_ddb(tbl.scan().get("Items", [])):
        if w.get("status") != "STALE" and db.age_seconds(w.get("last_seen")) > HEARTBEAT_TIMEOUT_S:
            tbl.update_item(Key={"instance_id": w["instance_id"]},
                            UpdateExpression="SET #s = :s", ExpressionAttributeNames={"#s": "status"},
                            ExpressionAttributeValues={":s": "STALE"})
            n += 1
    return n

# Checks every running run and every worker
@logger.inject_lambda_context
def handler(event, context):
    from boto3.dynamodb.conditions import Key
    counts: dict[str, int] = {}
    for status in ("RUNNING", "CANCEL_REQUESTED"):
        for run in db.query_all(db.runs_table(), IndexName="status_index",
                                KeyConditionExpression=Key("status").eq(status)):
            run = db.from_ddb(run)
            logger.append_keys(run_id=run["run_id"])
            action = check_run(run)
            counts[action] = counts.get(action, 0) + 1
    stale = mark_stale_workers()
    logger.info("watchdog done", extra={"actions": counts, "stale_workers": stale})
    return {"actions": counts, "stale_workers": stale}
