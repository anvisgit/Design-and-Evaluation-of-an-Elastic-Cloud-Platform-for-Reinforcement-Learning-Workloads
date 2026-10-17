from __future__ import annotations
import functools
import json
from typing import Any, Callable, Optional
from botocore.exceptions import ClientError
from pydantic import ValidationError
from common.db import from_ddb
from common.log import logger, tracer

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,Idempotency-Key",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

class ApiError(Exception):
    # Keeps the code, message and status
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status

# Builds an API Gateway response
def _resp(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json", **CORS},
            "body": json.dumps(body, default=str)}

# Builds a success response
def ok(data: Any, status: int = 200) -> dict:
    return _resp(status, {"data": from_ddb(data), "error": None})

# Builds an error response
def fail(code: str, message: str, status: int) -> dict:
    return _resp(status, {"data": None, "error": {"code": code, "message": message}})

# Reads the JSON body of a request
def parse_body(event: dict) -> dict:
    raw = event.get("body")
    if not raw:
        raise ApiError("INVALID_BODY", "request body is required", 400)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise ApiError("INVALID_JSON", "body is not valid JSON", 400)
    if not isinstance(body, dict):
        raise ApiError("INVALID_BODY", "body must be a JSON object", 400)
    return body

# Returns the user who made the request
def owner_of(event: dict) -> str:
    claims = (event.get("requestContext") or {}).get("authorizer", {}).get("claims") or {}
    owner = claims.get("cognito:username") or claims.get("sub")
    if not owner:
        raise ApiError("UNAUTHORIZED", "missing identity", 401)
    return owner

# Returns a path parameter or raises
def path_param(event: dict, name: str) -> str:
    v = (event.get("pathParameters") or {}).get(name)
    if not v:
        raise ApiError("MISSING_PARAM", f"path parameter {name} is required", 400)
    return v

# Returns a query parameter
def query_param(event: dict, name: str, default: Optional[str] = None) -> Optional[str]:
    return (event.get("queryStringParameters") or {}).get(name, default)

# Returns an integer query parameter inside limits
def int_param(event: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = query_param(event, name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ApiError("INVALID_PARAM", f"{name} must be an integer", 400)
    return max(lo, min(hi, v))

# Returns a header, ignoring the letter case
def header(event: dict, name: str) -> Optional[str]:
    for k, v in (event.get("headers") or {}).items():
        if k.lower() == name.lower():
            return v
    return None

# Turns a validation error into one line
def format_validation_error(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'body'}: {err['msg']}"
                     for err in e.errors())

# Wraps a handler with logging, tracing and error responses
def api_handler(fn: Callable) -> Callable:
    # Runs the handler and maps errors to responses
    @functools.wraps(fn)
    @logger.inject_lambda_context
    @tracer.capture_lambda_handler
    def wrapper(event, context):
        try:
            rid = (event.get("pathParameters") or {}).get("run_id")
            if rid:
                logger.append_keys(run_id=rid)
            return fn(event, context)
        except ApiError as e:
            return fail(e.code, e.message, e.status)
        except ValidationError as e:
            return fail("VALIDATION_ERROR", format_validation_error(e), 400)
        except ClientError:
            logger.exception("aws error")
            return fail("INTERNAL_ERROR", "upstream service error", 500)
        except Exception:
            logger.exception("unhandled error")
            return fail("INTERNAL_ERROR", "internal error", 500)
    return wrapper
