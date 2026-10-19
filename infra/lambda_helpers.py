from __future__ import annotations
from pathlib import Path
from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_logs as logs
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS_DIR = ROOT / "lambdas"
LAYER_DIR = ROOT / "build" / "layer"

# Gets the shared dependency layer
def deps_layer(scope: Construct) -> _lambda.LayerVersion:
    cached = getattr(scope, "_deps_layer", None)
    if cached is None:
        LAYER_DIR.mkdir(parents=True, exist_ok=True)
        cached = _lambda.LayerVersion(
            scope, "DepsLayer", code=_lambda.Code.from_asset(str(LAYER_DIR)),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_11],
            compatible_architectures=[_lambda.Architecture.ARM_64],
            description="pydantic + aws-lambda-powertools")
        scope._deps_layer = cached
    return cached

# Creates a Lambda function with the common settings
def make_function(scope: Construct, id: str, package: str, *, env: dict | None = None,
                  timeout_s: int = 15, memory: int = 256, handler: str = "handler.handler") -> _lambda.Function:
    return _lambda.Function(
        scope, id,
        runtime=_lambda.Runtime.PYTHON_3_11, architecture=_lambda.Architecture.ARM_64,
        handler=f"{package}.{handler}",
        code=_lambda.Code.from_asset(str(LAMBDAS_DIR), exclude=["__pycache__", "*.pyc", "requirements.txt"]),
        layers=[deps_layer(scope)], memory_size=memory, timeout=Duration.seconds(timeout_s),
        tracing=_lambda.Tracing.ACTIVE,
        log_group=logs.LogGroup(scope, f"{id}Logs", retention=logs.RetentionDays.ONE_MONTH,
                                removal_policy=RemovalPolicy.DESTROY),
        environment={"POWERTOOLS_SERVICE_NAME": package, "LOG_LEVEL": "INFO", **(env or {})},
    )
