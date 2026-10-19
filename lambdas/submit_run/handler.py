import json
import os
import uuid
from botocore.exceptions import ClientError
from common import db
from common.config_schema import TrainingConfig
from common.log import logger
from common.responses import ApiError, api_handler, header, ok, owner_of, parse_body

# Checks, stores and queues a new run
@api_handler
def handler(event, context):
    owner = owner_of(event)
    cfg = TrainingConfig.model_validate(parse_body(event))
    exp = db.get_experiment(cfg.experiment_id)
    if not exp or exp["owner"] != owner:
        raise ApiError("EXPERIMENT_NOT_FOUND", "experiment does not exist", 404)
    idem = header(event, "Idempotency-Key")
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{owner}:{idem}")) if idem else str(uuid.uuid4())
    if idem:
        existing = db.get_run(run_id)
        if existing:
            return ok({"run_id": run_id, "status": existing["status"], "idempotent_replay": True}, 202)
    quota = int(os.environ.get("MAX_ACTIVE_RUNS", "5"))
    if len(db.active_runs(owner)) >= quota:
        raise ApiError("QUOTA_EXCEEDED", f"you already have {quota} active runs", 429)
    logger.append_keys(run_id=run_id)
    created = db.now_iso()
    config_key = f"configs/{run_id}/config.json"
    config_uri = f"s3://{db.bucket()}/{config_key}"
    run = {
        "run_id": run_id, "experiment_id": cfg.experiment_id, "owner": owner,
        "name": cfg.name, "status": "SUBMITTED", "algo": cfg.algo, "env_id": cfg.env_id,
        "seed": cfg.seed, "total_timesteps": cfg.total_timesteps,
        "hyperparams": cfg.hyperparams, "config_s3_uri": config_uri,
        "max_runtime_minutes": cfg.resources.max_runtime_minutes,
        "created_at": created, "last_heartbeat": created,
        "current_step": 0, "attempt": 1, "cancel_requested": False,
    }
    if idem:
        run["idempotency_key"] = idem
    try:
        db.runs_table().put_item(Item=db.to_ddb(run),
                                 ConditionExpression="attribute_not_exists(run_id)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return ok({"run_id": run_id, "status": "SUBMITTED", "idempotent_replay": True}, 202)
        raise
    try:
        db.s3().put_object(Bucket=db.bucket(), Key=config_key,
                           Body=json.dumps(cfg.model_dump()).encode(),
                           ContentType="application/json")
        db.update_run(run_id, {"status": "QUEUED"}, condition="#s = :sub",
                      cond_names={"#s": "status"}, cond_values={":sub": "SUBMITTED"})
        db.sqs().send_message(QueueUrl=os.environ["QUEUE_URL"], MessageBody=db.queue_message(run, 1))
    except Exception as e:
        db.update_run(run_id, {"status": "FAILED", "error": f"enqueue_failed: {e}"[:500],
                               "ended_at": db.now_iso()})
        raise
    db.experiments_table().update_item(
        Key={"experiment_id": cfg.experiment_id},
        UpdateExpression="ADD run_count :one", ExpressionAttributeValues={":one": 1})
    logger.info("run submitted")
    return ok({"run_id": run_id, "status": "QUEUED"}, 202)
