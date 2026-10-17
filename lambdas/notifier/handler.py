import json
import os
import boto3
from boto3.dynamodb.types import TypeDeserializer
from common import db
from common.log import logger

_deser = TypeDeserializer()
EVENTS = {"SUCCEEDED": "run_succeeded", "FAILED": "run_failed", "CANCELLED": "run_cancelled"}

# Turns a stream image into a plain dict
def _plain(image: dict) -> dict:
    return db.from_ddb({k: _deser.deserialize(v) for k, v in image.items()})

# Returns how long a run took in seconds
def duration_s(run: dict):
    if run.get("started_at") and run.get("ended_at"):
        return int((db.parse_iso(run["ended_at"]) - db.parse_iso(run["started_at"])).total_seconds())
    return None

# Builds the text of the notification
def format_message(run: dict) -> str:
    dur = duration_s(run)
    lines = [f"Run {run['run_id']} ({run.get('name', '')}) is {run['status']}",
             f"Experiment: {run.get('experiment_id')}",
             f"Algo/Env: {run.get('algo')} / {run.get('env_id')}",
             f"Duration: {dur}s" if dur is not None else "Duration: n/a",
             f"Final reward: {run.get('final_reward', 'n/a')}"]
    if run.get("error"):
        lines.append(f"Error: {str(run['error'])[:500]}")
    base = os.environ.get("API_URL", "").rstrip("/")
    if base:
        lines.append(f"Link: {base}/runs/{run['run_id']}")
    return "\n".join(lines)

# Sends a notification when a run ends
def handler(event, context):
    sns, cw = boto3.client("sns"), boto3.client("cloudwatch")
    sent = 0
    for rec in event.get("Records", []):
        if rec.get("eventName") != "MODIFY":
            continue
        img = rec["dynamodb"]
        new, old = _plain(img["NewImage"]), _plain(img.get("OldImage", {}))
        if new["status"] not in EVENTS or old.get("status") == new["status"]:
            continue
        event_name = EVENTS[new["status"]]
        sns.publish(
            TopicArn=os.environ["TOPIC_ARN"],
            Subject=f"[rl-platform] {event_name}: {new.get('name', new['run_id'])}"[:100],
            Message=format_message(new),
            MessageAttributes={
                "event": {"DataType": "String", "StringValue": event_name},
                "run_id": {"DataType": "String", "StringValue": new["run_id"]},
                "experiment_id": {"DataType": "String", "StringValue": new["experiment_id"]},
            })
        cw.put_metric_data(Namespace="RLPlatform", MetricData=[{
            "MetricName": "RunsByStatus", "Value": 1, "Unit": "Count",
            "Dimensions": [{"Name": "Status", "Value": new["status"]}]}])
        logger.info(json.dumps({"msg": "notified", "run_id": new["run_id"], "event": event_name}))
        sent += 1
    return {"sent": sent}
