from boto3.dynamodb.conditions import Attr, Key
from common import db
from common.responses import ApiError, api_handler, int_param, ok, owner_of, query_param

FIELDS = ["run_id", "experiment_id", "name", "owner", "status", "algo", "env_id", "seed",
          "total_timesteps", "created_at", "started_at", "ended_at", "current_step",
          "best_reward", "final_reward", "attempt", "error"]
STATUSES = {"SUBMITTED", "QUEUED", "RUNNING", "REQUEUED", "CANCEL_REQUESTED",
            "SUCCEEDED", "FAILED", "CANCELLED"}

# Lists the runs of the user
@api_handler
def handler(event, context):
    owner = owner_of(event)
    exp_id, status = query_param(event, "experiment_id"), query_param(event, "status")
    if status and status not in STATUSES:
        raise ApiError("INVALID_PARAM", f"status must be one of {sorted(STATUSES)}", 400)
    kwargs = {"ScanIndexForward": False, "Limit": int_param(event, "limit", 50, 1, 100)}
    kwargs.update(db.projection(FIELDS))
    if exp_id:
        kwargs["IndexName"] = "experiment_index"
        kwargs["KeyConditionExpression"] = Key("experiment_id").eq(exp_id)
        flt = Attr("owner").eq(owner)
    else:
        kwargs["IndexName"] = "owner_index"
        kwargs["KeyConditionExpression"] = Key("owner").eq(owner)
        flt = None
    if status:
        flt = Attr("status").eq(status) if flt is None else flt & Attr("status").eq(status)
    if flt is not None:
        kwargs["FilterExpression"] = flt
    cursor = db.decode_cursor(query_param(event, "cursor"))
    if cursor:
        kwargs["ExclusiveStartKey"] = cursor
    resp = db.runs_table().query(**kwargs)
    return ok({"items": resp["Items"], "next_cursor": db.encode_cursor(resp.get("LastEvaluatedKey"))})
