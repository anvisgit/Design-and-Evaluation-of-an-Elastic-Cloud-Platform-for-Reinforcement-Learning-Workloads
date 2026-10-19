from botocore.exceptions import ClientError
from common import db
from common.log import logger
from common.responses import ApiError, api_handler, ok, owner_of, path_param

# Updates a run only if its status is in the allowed list
def _cond(run_id, fields, allowed_status):
    names = {"#s": "status"}
    values = {f":a{i}": s for i, s in enumerate(allowed_status)}
    cond = "#s IN (" + ", ".join(values) + ")"
    db.update_run(run_id, fields, condition=cond, cond_names=names, cond_values=values)

# Cancels a run now or asks the worker to stop it
@api_handler
def handler(event, context):
    run_id = path_param(event, "run_id")
    run = db.get_run(run_id)
    if not run or run["owner"] != owner_of(event):
        raise ApiError("RUN_NOT_FOUND", "run does not exist", 404)
    status = run["status"]
    if status in db.TERMINAL:
        raise ApiError("RUN_FINISHED", f"run already {status}", 409)
    if status == "CANCEL_REQUESTED":
        return ok({"run_id": run_id, "status": status}, 202)
    try:
        if status in ("SUBMITTED", "QUEUED", "REQUEUED"):
            _cond(run_id, {"status": "CANCELLED", "cancel_requested": True,
                           "ended_at": db.now_iso()}, ["SUBMITTED", "QUEUED", "REQUEUED"])
            new = "CANCELLED"
        else:
            _cond(run_id, {"status": "CANCEL_REQUESTED", "cancel_requested": True}, ["RUNNING"])
            new = "CANCEL_REQUESTED"
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise ApiError("CONFLICT", "run state changed, retry", 409)
        raise
    logger.info("cancel requested", extra={"new_status": new})
    return ok({"run_id": run_id, "status": new}, 202)
