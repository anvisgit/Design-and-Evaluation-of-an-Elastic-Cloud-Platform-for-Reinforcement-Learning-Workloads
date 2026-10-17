from __future__ import annotations
import base64
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable, Optional
import boto3
from boto3.dynamodb.conditions import Key

TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
ACTIVE = ("SUBMITTED", "QUEUED", "RUNNING", "REQUEUED", "CANCEL_REQUESTED")
MAX_ATTEMPTS = 3
_cache: dict[str, Any] = {}

# Clears the cached AWS clients
def reset_clients() -> None:
    _cache.clear()

# Gets a cached boto3 resource
def _res(kind: str):
    if kind not in _cache:
        _cache[kind] = boto3.resource(kind)
    return _cache[kind]

# Gets a cached boto3 client
def _cli(kind: str, **kw):
    key = "c:" + kind
    if key not in _cache:
        _cache[key] = boto3.client(kind, **kw)
    return _cache[key]

# Returns the current UTC time
def now() -> datetime:
    return datetime.now(timezone.utc)

# Returns the current UTC time as text
def now_iso() -> str:
    return now().isoformat(timespec="seconds")

# Turns a time text into a datetime
def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)

# Returns how many seconds ago a time was
def age_seconds(iso: Optional[str]) -> float:
    if not iso:
        return float("inf")
    return (now() - parse_iso(iso)).total_seconds()

# Gets a table by the name in an env variable
def table(env_name: str):
    return _res("dynamodb").Table(os.environ[env_name])

# Gets the runs table
def runs_table():
    return table("RUNS_TABLE")

# Gets the experiments table
def experiments_table():
    return table("EXPERIMENTS_TABLE")

# Gets the metrics table
def metrics_table():
    return table("METRICS_TABLE")

# Gets the workers table
def workers_table():
    return table("WORKERS_TABLE")

# Gets the S3 client
def s3():
    return _cli("s3", config=boto3.session.Config(signature_version="s3v4"))

# Gets the SQS client
def sqs():
    return _cli("sqs")

# Returns the bucket name
def bucket() -> str:
    return os.environ["BUCKET"]

# Splits an s3 uri into bucket and key
def parse_s3_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k

# Turns floats into Decimal for DynamoDB
def to_ddb(obj: Any) -> Any:
    return json.loads(json.dumps(obj), parse_float=Decimal)

# Turns DynamoDB values into plain Python values
def from_ddb(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: from_ddb(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [from_ddb(v) for v in obj]
    return obj

# Turns a DynamoDB key into a page cursor
def encode_cursor(key: Optional[dict]) -> Optional[str]:
    if not key:
        return None
    return base64.urlsafe_b64encode(json.dumps(from_ddb(key)).encode()).decode()

# Turns a page cursor back into a DynamoDB key
def decode_cursor(cursor: Optional[str]) -> Optional[dict]:
    if not cursor:
        return None
    key = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    return {k: (Decimal(str(v)) if isinstance(v, (int, float)) else v) for k, v in key.items()}

# Builds a projection for the given fields
def projection(fields: Iterable[str]) -> dict:
    names = {f"#p{i}": f for i, f in enumerate(fields)}
    return {"ProjectionExpression": ", ".join(names), "ExpressionAttributeNames": names}

# Gets one run or None
def get_run(run_id: str) -> Optional[dict]:
    item = runs_table().get_item(Key={"run_id": run_id}).get("Item")
    return from_ddb(item) if item else None

# Gets one experiment or None
def get_experiment(experiment_id: str) -> Optional[dict]:
    item = experiments_table().get_item(Key={"experiment_id": experiment_id}).get("Item")
    return from_ddb(item) if item else None

# Sets and removes fields on a run, with an optional condition
def update_run(run_id: str, fields: dict, condition=None, cond_names=None, cond_values=None,
               remove: Iterable[str] = ()) -> None:
    names = {f"#f{i}": k for i, k in enumerate(fields)}
    values = {f":v{i}": to_ddb(v) for i, v in enumerate(fields.values())}
    expr = "SET " + ", ".join(f"#f{i} = :v{i}" for i in range(len(fields)))
    rem = list(remove)
    for j, r in enumerate(rem):
        names[f"#r{j}"] = r
    if rem:
        expr += " REMOVE " + ", ".join(f"#r{j}" for j in range(len(rem)))
    kwargs: dict[str, Any] = {"Key": {"run_id": run_id}, "UpdateExpression": expr}
    if condition:
        kwargs["ConditionExpression"] = condition
        names.update(cond_names or {})
        values.update({k: to_ddb(v) for k, v in (cond_values or {}).items()})
    kwargs["ExpressionAttributeNames"] = names
    if values:
        kwargs["ExpressionAttributeValues"] = values
    runs_table().update_item(**kwargs)

# Runs a query and follows every page
def query_all(tbl, **kwargs) -> list[dict]:
    items: list[dict] = []
    while True:
        resp = tbl.query(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            return items
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

# Lists active runs, for one owner if given
def active_runs(owner: Optional[str] = None) -> list[dict]:
    out: list[dict] = []
    for st in ACTIVE:
        out += query_all(runs_table(), IndexName="status_index",
                         KeyConditionExpression=Key("status").eq(st))
    out = [from_ddb(r) for r in out]
    return [r for r in out if owner is None or r.get("owner") == owner]

# Builds the queue message for a run
def queue_message(run: dict, attempt: Optional[int] = None) -> str:
    return json.dumps({
        "run_id": run["run_id"],
        "experiment_id": run["experiment_id"],
        "config_s3_uri": run["config_s3_uri"],
        "attempt": attempt if attempt is not None else run.get("attempt", 1),
    })

# Returns an expiry time in seconds, some days from now
def expires_in_days(days: int) -> int:
    return int((now() + timedelta(days=days)).timestamp())
