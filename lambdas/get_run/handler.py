from common import db
from common.responses import ApiError, api_handler, ok, owner_of, path_param

# Loads a run and checks that the user owns it
def load_owned_run(event) -> dict:
    run = db.get_run(path_param(event, "run_id"))
    if not run or run["owner"] != owner_of(event):
        raise ApiError("RUN_NOT_FOUND", "run does not exist", 404)
    return run

# Returns one run
@api_handler
def handler(event, context):
    return ok(load_owned_run(event))
