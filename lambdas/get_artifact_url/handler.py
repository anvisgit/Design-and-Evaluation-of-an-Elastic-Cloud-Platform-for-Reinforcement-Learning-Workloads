from botocore.exceptions import ClientError
from common import db
from common.responses import ApiError, api_handler, ok, owner_of, path_param

KINDS = {"model": "model_s3_uri", "checkpoint": "latest_checkpoint_s3_uri",
         "config": "config_s3_uri", "log": None}
EXPIRES = 900

# Returns a short download link for a run file
@api_handler
def handler(event, context):
    run_id, kind = path_param(event, "run_id"), path_param(event, "kind")
    if kind not in KINDS:
        raise ApiError("INVALID_KIND", f"kind must be one of {sorted(KINDS)}", 400)
    run = db.get_run(run_id)
    if not run or run["owner"] != owner_of(event):
        raise ApiError("RUN_NOT_FOUND", "run does not exist", 404)
    uri = run.get(KINDS[kind]) if KINDS[kind] else f"s3://{db.bucket()}/logs/{run_id}/train.log"
    if not uri:
        raise ApiError("ARTIFACT_NOT_AVAILABLE", f"no {kind} artifact yet", 404)
    bucket, key = db.parse_s3_uri(uri)
    try:
        db.s3().head_object(Bucket=bucket, Key=key)
    except ClientError:
        raise ApiError("ARTIFACT_NOT_AVAILABLE", f"no {kind} artifact yet", 404)
    url = db.s3().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                         ExpiresIn=EXPIRES)
    return ok({"url": url, "expires_in": EXPIRES, "kind": kind})
