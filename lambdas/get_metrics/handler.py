from decimal import Decimal
from boto3.dynamodb.conditions import Key
from common import db
from common.responses import ApiError, api_handler, int_param, ok, owner_of, path_param, query_param

# Returns a page of metrics for a run
@api_handler
def handler(event, context):
    run_id = path_param(event, "run_id")
    run = db.get_run(run_id)
    if not run or run["owner"] != owner_of(event):
        raise ApiError("RUN_NOT_FOUND", "run does not exist", 404)
    lo = int_param(event, "from_step", 0, 0, 2**40)
    hi = int_param(event, "to_step", 2**40, 0, 2**40)
    kwargs = {
        "KeyConditionExpression": Key("run_id").eq(run_id) & Key("step").between(Decimal(lo), Decimal(hi)),
        "Limit": int_param(event, "limit", 1000, 1, 5000),
        "ProjectionExpression": "step, ep_rew_mean, ep_len_mean, loss, value_loss, policy_loss, "
                                "entropy, learning_rate, fps, wall_time",
    }
    cursor = db.decode_cursor(query_param(event, "cursor"))
    if cursor:
        kwargs["ExclusiveStartKey"] = cursor
    resp = db.metrics_table().query(**kwargs)
    return ok({"run_id": run_id, "items": resp["Items"],
               "next_cursor": db.encode_cursor(resp.get("LastEvaluatedKey"))})
