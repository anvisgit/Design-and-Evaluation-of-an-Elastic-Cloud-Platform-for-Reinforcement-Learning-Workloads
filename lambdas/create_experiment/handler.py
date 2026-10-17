import uuid
from common import db
from common.config_schema import ExperimentIn
from common.responses import api_handler, ok, owner_of, parse_body

# Creates an experiment
@api_handler
def handler(event, context):
    owner = owner_of(event)
    body = ExperimentIn.model_validate(parse_body(event))
    item = {
        "experiment_id": str(uuid.uuid4()),
        "name": body.name,
        "description": body.description,
        "owner": owner,
        "tags": body.tags,
        "created_at": db.now_iso(),
        "run_count": 0,
    }
    db.experiments_table().put_item(Item=db.to_ddb(item))
    return ok(item, 201)
