from boto3.dynamodb.conditions import Key
from common import db
from common.responses import api_handler, int_param, ok, owner_of, query_param

# Lists the experiments of the user
@api_handler
def handler(event, context):
    owner = owner_of(event)
    kwargs = {
        "IndexName": "owner_index",
        "KeyConditionExpression": Key("owner").eq(owner),
        "ScanIndexForward": False,
        "Limit": int_param(event, "limit", 50, 1, 100),
    }
    cursor = db.decode_cursor(query_param(event, "cursor"))
    if cursor:
        kwargs["ExclusiveStartKey"] = cursor
    resp = db.experiments_table().query(**kwargs)
    return ok({"items": resp["Items"], "next_cursor": db.encode_cursor(resp.get("LastEvaluatedKey"))})
