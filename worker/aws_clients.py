from __future__ import annotations
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional
import boto3
from botocore.exceptions import ClientError
import spot

log = logging.getLogger("worker.aws")
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
CLAIMABLE = ("SUBMITTED", "QUEUED", "REQUEUED")
HEARTBEAT_STALE_S = 600

# Turns floats into Decimal for DynamoDB
def _dec(v: Any) -> Any:
    if isinstance(v, float):
        return Decimal(str(v))
    if isinstance(v, dict):
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_dec(x) for x in v]
    return v

# Turns Decimal into int or float
def _plain(v: Any) -> Any:
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v

@dataclass
class Settings:
    runs_table: str = field(default_factory=lambda: os.environ.get("RUNS_TABLE", "runs"))
    metrics_table: str = field(default_factory=lambda: os.environ.get("METRICS_TABLE", "metrics"))
    workers_table: str = field(default_factory=lambda: os.environ.get("WORKERS_TABLE", "workers"))
    bucket: str = field(default_factory=lambda: os.environ.get("BUCKET", "rl-local"))
    queue_url: str = field(default_factory=lambda: os.environ.get("QUEUE_URL", ""))
    asg_name: str = field(default_factory=lambda: os.environ.get("ASG_NAME", ""))
    local_mode: bool = field(default_factory=lambda: os.environ.get("LOCAL_MODE", "0") == "1")
    idle_timeout_s: int = field(default_factory=lambda: int(os.environ.get("IDLE_TIMEOUT_S", "300")))
    workdir: str = field(default_factory=lambda: os.environ.get("WORKDIR", "/tmp/rl-worker"))
    retry_delay_s: int = field(default_factory=lambda: int(os.environ.get("RETRY_DELAY_S", "30")))
    instance_id: str = ""
    instance_type: str = "local"
    is_spot: bool = False
    # Finds the instance id and type
    def __post_init__(self):
        if self.local_mode:
            self.instance_id = os.environ.get("INSTANCE_ID", f"local-{socket.gethostname()}")
        else:
            self.instance_id = spot.imds_get("instance-id") or os.environ.get("INSTANCE_ID", socket.gethostname())
            self.instance_type = spot.imds_get("instance-type") or "unknown"
            self.is_spot = (spot.imds_get("instance-life-cycle") or "") == "spot"

class RunStore:
    # Creates the AWS clients
    def __init__(self, settings: Settings):
        self.s = settings
        ddb = boto3.resource("dynamodb")
        self.runs = ddb.Table(settings.runs_table)
        self.metrics = ddb.Table(settings.metrics_table)
        self.workers = ddb.Table(settings.workers_table)
        self.s3 = boto3.client("s3")
        self.sqs = boto3.client("sqs")
        self.cw = boto3.client("cloudwatch")
        self.asg = boto3.client("autoscaling")
    # Gets one run or None
    def get_run(self, run_id: str) -> Optional[dict]:
        item = self.runs.get_item(Key={"run_id": run_id}, ConsistentRead=True).get("Item")
        return _plain(item) if item else None
    # Sets fields on a run, with an optional condition
    def update(self, run_id: str, fields: dict, condition: Optional[str] = None,
               names: Optional[dict] = None, values: Optional[dict] = None) -> bool:
        n = {f"#f{i}": k for i, k in enumerate(fields)}
        v = {f":v{i}": _dec(x) for i, x in enumerate(fields.values())}
        kw: dict[str, Any] = {"Key": {"run_id": run_id},
                              "UpdateExpression": "SET " + ", ".join(f"#f{i} = :v{i}" for i in range(len(fields)))}
        if condition:
            kw["ConditionExpression"] = condition
            n.update(names or {})
            v.update({k: _dec(x) for k, x in (values or {}).items()})
        kw["ExpressionAttributeNames"], kw["ExpressionAttributeValues"] = n, v
        try:
            self.runs.update_item(**kw)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
    # Updates a run only if this worker owns it
    def update_mine(self, run_id: str, fields: dict) -> bool:
        return self.update(run_id, fields, "#i = :me", {"#i": "instance_id"}, {":me": self.s.instance_id})
    # Moves a run to running if it is free
    def claim(self, run: dict, receive_count: int, provenance: dict) -> bool:
        fields = {"status": "RUNNING", "instance_id": self.s.instance_id, "started_at": _now_iso(),
                  "last_heartbeat": _now_iso(), "attempt": max(int(run.get("attempt", 1)), receive_count),
                  "instance_type": self.s.instance_type, "is_spot": self.s.is_spot, **provenance}
        if run["status"] in CLAIMABLE:
            return self.update(run["run_id"], fields, "#s = :s", {"#s": "status"}, {":s": run["status"]})
        if run["status"] == "RUNNING" and heartbeat_age(run) > HEARTBEAT_STALE_S:
            return self.update(run["run_id"], fields, "#s = :s AND #h = :h",
                               {"#s": "status", "#h": "last_heartbeat"},
                               {":s": "RUNNING", ":h": run["last_heartbeat"]})
        return False
    # Saves the time and step of a running run
    def heartbeat(self, run_id: str, step: int) -> None:
        self.update_mine(run_id, {"last_heartbeat": _now_iso(), "current_step": int(step)})
    # Checks if a cancel was asked
    def is_cancel_requested(self, run_id: str) -> bool:
        item = self.runs.get_item(Key={"run_id": run_id}, ProjectionExpression="cancel_requested, #s",
                                  ExpressionAttributeNames={"#s": "status"}, ConsistentRead=True).get("Item", {})
        return bool(item.get("cancel_requested")) or item.get("status") == "CANCEL_REQUESTED"
    # Saves metric rows
    def put_metrics(self, run_id: str, rows: Iterable[dict]) -> None:
        with self.metrics.batch_writer() as bw:
            for r in rows:
                bw.put_item(Item=_dec({k: v for k, v in r.items() if v is not None} | {"run_id": run_id}))
    # Sends metrics to CloudWatch
    def put_cw(self, algo: str, env_id: str, values: dict[str, float]) -> None:
        dims = [{"Name": "Algo", "Value": algo}, {"Name": "EnvId", "Value": env_id}]
        try:
            self.cw.put_metric_data(Namespace="RLPlatform", MetricData=[
                {"MetricName": k, "Value": float(v), "Dimensions": dims,
                 "Unit": "Count/Second" if k == "StepsPerSecond" else "None"} for k, v in values.items()])
        except Exception:
            log.warning("put_metric_data failed", exc_info=True)
    # Uploads a file to S3
    def upload(self, path: str | Path, key: str) -> str:
        self.s3.upload_file(str(path), self.s.bucket, key)
        return f"s3://{self.s.bucket}/{key}"
    # Downloads a file from S3
    def download(self, uri: str, dest: str | Path) -> Path:
        bucket, _, key = uri[5:].partition("/")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(bucket, key, str(dest))
        return Path(dest)
    # Changes how long a message stays hidden
    def extend_visibility(self, receipt: str, seconds: int = 600) -> None:
        try:
            self.sqs.change_message_visibility(QueueUrl=self.s.queue_url, ReceiptHandle=receipt,
                                               VisibilityTimeout=seconds)
        except ClientError:
            log.warning("change_message_visibility failed", exc_info=True)
    # Deletes a queue message
    def delete_message(self, receipt: str) -> None:
        try:
            self.sqs.delete_message(QueueUrl=self.s.queue_url, ReceiptHandle=receipt)
        except ClientError:
            log.warning("delete_message failed", exc_info=True)
    # Saves the state of this worker
    def register_worker(self, status: str, run_id: Optional[str] = None) -> None:
        self.workers.put_item(Item={
            "instance_id": self.s.instance_id, "status": status, "current_run_id": run_id or "",
            "last_seen": _now_iso(), "instance_type": self.s.instance_type, "is_spot": self.s.is_spot})
    # Turns scale in protection on or off
    def set_protection(self, on: bool) -> None:
        if self.s.local_mode or not self.s.asg_name:
            return
        try:
            self.asg.set_instance_protection(InstanceIds=[self.s.instance_id],
                                             AutoScalingGroupName=self.s.asg_name, ProtectedFromScaleIn=on)
        except ClientError:
            log.warning("set_instance_protection failed", exc_info=True)
    # Removes this instance from the group
    def terminate_self(self) -> None:
        if self.s.local_mode or not self.s.asg_name:
            return
        self.set_protection(False)
        self.asg.terminate_instance_in_auto_scaling_group(
            InstanceId=self.s.instance_id, ShouldDecrementDesiredCapacity=True)

# Returns the current UTC time as text
def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Returns how old the last heartbeat is in seconds
def heartbeat_age(run: dict) -> float:
    from datetime import datetime, timezone
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(run["last_heartbeat"])).total_seconds()
    except (KeyError, ValueError):
        return float("inf")
