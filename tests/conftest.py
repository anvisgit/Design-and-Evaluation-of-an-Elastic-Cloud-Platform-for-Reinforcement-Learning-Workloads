import os

os.environ.update({
    "AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_REGION": "us-east-1", "POWERTOOLS_TRACE_DISABLED": "true", "POWERTOOLS_SERVICE_NAME": "test",
    "LOG_LEVEL": "WARNING", "RUNS_TABLE": "runs", "EXPERIMENTS_TABLE": "experiments",
    "METRICS_TABLE": "metrics", "WORKERS_TABLE": "workers", "BUCKET": "test-bucket",
    "TOPIC_ARN": "arn:aws:sns:us-east-1:123456789012:rl_notifications", "MAX_ACTIVE_RUNS": "5",
})
os.environ.pop("AWS_ENDPOINT_URL", None)
import boto3
import pytest
from moto import mock_aws
from common import db

# Creates a DynamoDB table
def _table(ddb, name, hash_key, range_key=None, gsis=(), stream=False):
    attrs, schema = {}, [{"AttributeName": hash_key[0], "KeyType": "HASH"}]
    attrs[hash_key[0]] = hash_key[1]
    if range_key:
        schema.append({"AttributeName": range_key[0], "KeyType": "RANGE"})
        attrs[range_key[0]] = range_key[1]
    gsi_defs = []
    for gname, (hk, hkt), rk in gsis:
        keys = [{"AttributeName": hk, "KeyType": "HASH"}]
        attrs[hk] = hkt
        if rk:
            keys.append({"AttributeName": rk[0], "KeyType": "RANGE"})
            attrs[rk[0]] = rk[1]
        gsi_defs.append({"IndexName": gname, "KeySchema": keys, "Projection": {"ProjectionType": "ALL"}})
    kw = dict(TableName=name, KeySchema=schema, BillingMode="PAY_PER_REQUEST",
              AttributeDefinitions=[{"AttributeName": k, "AttributeType": v} for k, v in attrs.items()])
    if gsi_defs:
        kw["GlobalSecondaryIndexes"] = gsi_defs
    if stream:
        kw["StreamSpecification"] = {"StreamEnabled": True, "StreamViewType": "NEW_AND_OLD_IMAGES"}
    ddb.create_table(**kw)

# Gives fake tables, bucket and queue
@pytest.fixture
def aws():
    with mock_aws():
        db.reset_clients()
        ddb = boto3.client("dynamodb")
        _table(ddb, "experiments", ("experiment_id", "S"),
               gsis=[("owner_index", ("owner", "S"), ("created_at", "S"))])
        _table(ddb, "runs", ("run_id", "S"), stream=True, gsis=[
            ("experiment_index", ("experiment_id", "S"), ("created_at", "S")),
            ("status_index", ("status", "S"), ("last_heartbeat", "S")),
            ("owner_index", ("owner", "S"), ("created_at", "S"))])
        _table(ddb, "metrics", ("run_id", "S"), ("step", "N"))
        _table(ddb, "workers", ("instance_id", "S"))
        boto3.client("s3").create_bucket(Bucket="test-bucket")
        q = boto3.client("sqs").create_queue(QueueName="training_jobs")["QueueUrl"]
        os.environ["QUEUE_URL"] = q
        yield {"queue_url": q}
        db.reset_clients()

# Builds a fake API event
def event(body=None, owner="alice", path=None, query=None, headers=None):
    return {
        "body": None if body is None else (body if isinstance(body, str) else __import__("json").dumps(body)),
        "pathParameters": path, "queryStringParameters": query, "headers": headers or {},
        "requestContext": {"authorizer": {"claims": {"cognito:username": owner}}},
    }

class Ctx:
    function_name, memory_limit_in_mb, invoked_function_arn, aws_request_id = "t", 256, "arn:aws:lambda:x", "r"
    # Returns a fixed time left
    def get_remaining_time_in_millis(self):
        return 1000

# Gives a helper that calls a handler and returns code and body
@pytest.fixture
def call():
    import json
    # Calls a handler and parses the response
    def _call(handler, **kw):
        resp = handler(event(**kw), Ctx())
        return resp["statusCode"], json.loads(resp["body"])
    return _call
