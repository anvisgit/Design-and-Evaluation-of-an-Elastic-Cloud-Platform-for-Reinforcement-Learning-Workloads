from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_cognito as cognito
from constructs import Construct
from lambda_helpers import make_function

class ApiStack(Stack):
    # Builds the login, the Lambdas and the REST API
    def __init__(self, scope: Construct, id: str, *, storage, messaging, stage: str = "dev",
                 max_active_runs: int = 5, **kw):
        super().__init__(scope, id, **kw)
        prod = stage == "prod"
        self.user_pool = cognito.UserPool(
            self, "Users", user_pool_name="rl-platform-users", self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(username=True, email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            password_policy=cognito.PasswordPolicy(min_length=12, require_digits=True, require_symbols=False),
            removal_policy=RemovalPolicy.RETAIN if prod else RemovalPolicy.DESTROY)
        self.client = self.user_pool.add_client(
            "Cli", user_pool_client_name="rl-platform-cli", generate_secret=False,
            auth_flows=cognito.AuthFlow(user_password=True, user_srp=True))
        authorizer = apigw.CognitoUserPoolsAuthorizer(self, "Authorizer", cognito_user_pools=[self.user_pool])
        base_env = {"RUNS_TABLE": storage.runs.table_name, "EXPERIMENTS_TABLE": storage.experiments.table_name,
                    "METRICS_TABLE": storage.metrics.table_name, "BUCKET": storage.bucket.bucket_name}
        f = lambda id_, pkg, **kw_: make_function(self, id_, pkg, env={**base_env, **kw_.pop("env", {})}, **kw_)
        create_exp = f("CreateExperiment", "create_experiment")
        list_exp = f("ListExperiments", "list_experiments")
        submit = f("SubmitRun", "submit_run", timeout_s=20,
                   env={"QUEUE_URL": messaging.queue.queue_url, "MAX_ACTIVE_RUNS": str(max_active_runs)})
        get_run = f("GetRun", "get_run")
        list_runs = f("ListRuns", "list_runs")
        cancel = f("CancelRun", "cancel_run")
        metrics = f("GetMetrics", "get_metrics")
        compare = f("CompareRuns", "compare_runs", timeout_s=29, memory=512)
        artifact = f("GetArtifactUrl", "get_artifact_url")
        storage.experiments.grant_write_data(create_exp)
        storage.experiments.grant_read_data(list_exp)
        storage.experiments.grant_read_write_data(submit)
        storage.runs.grant_read_write_data(submit)
        storage.bucket.grant_put(submit, "configs/*")
        messaging.queue.grant_send_messages(submit)
        for fn in (get_run, list_runs, artifact):
            storage.runs.grant_read_data(fn)
        for fn in (metrics, compare):
            storage.runs.grant_read_data(fn)
            storage.metrics.grant_read_data(fn)
        storage.runs.grant_read_write_data(cancel)
        for prefix in ("models/*", "checkpoints/*", "logs/*", "configs/*"):
            storage.bucket.grant_read(artifact, prefix)
        self.api = apigw.RestApi(
            self, "Api", rest_api_name="rl-platform-api", description="RL training platform",
            deploy_options=apigw.StageOptions(stage_name="v1", tracing_enabled=True, metrics_enabled=True,
                                              throttling_rate_limit=20, throttling_burst_limit=40),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS, allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization", "Idempotency-Key"]),
            default_method_options=apigw.MethodOptions(authorizer=authorizer,
                                                       authorization_type=apigw.AuthorizationType.COGNITO))
        self.api.add_usage_plan("Plan", name="rl-platform-default",
                                throttle=apigw.ThrottleSettings(rate_limit=20, burst_limit=40),
                                quota=apigw.QuotaSettings(limit=10000, period=apigw.Period.DAY)
                                ).add_api_stage(stage=self.api.deployment_stage)
        validator = apigw.RequestValidator(self, "Validator", rest_api=self.api,
                                           validate_request_body=True, validate_request_parameters=True)
        T = apigw.JsonSchemaType
        model = lambda id_, schema: self.api.add_model(id_, content_type="application/json", model_name=id_,
                                                       schema=apigw.JsonSchema(schema=apigw.JsonSchemaVersion.DRAFT4,
                                                                               title=id_, **schema))
        exp_model = model("CreateExperimentModel", dict(
            type=T.OBJECT, required=["name"], properties={"name": apigw.JsonSchema(type=T.STRING, min_length=1)}))
        run_model = model("SubmitRunModel", dict(
            type=T.OBJECT, required=["experiment_id", "name", "algo", "env_id", "total_timesteps"],
            properties={"experiment_id": apigw.JsonSchema(type=T.STRING),
                        "algo": apigw.JsonSchema(type=T.STRING, enum=["PPO", "A2C", "DQN", "SAC"]),
                        "total_timesteps": apigw.JsonSchema(type=T.INTEGER, minimum=1, maximum=5000000)}))
        cmp_model = model("CompareModel", dict(
            type=T.OBJECT, required=["run_ids"], properties={
                "run_ids": apigw.JsonSchema(type=T.ARRAY, min_items=1, max_items=10,
                                            items=apigw.JsonSchema(type=T.STRING))}))
        # Adds one method to a resource
        def route(resource, method, fn, request_model=None, params=None):
            opts = {}
            if request_model:
                opts.update(request_validator=validator, request_models={"application/json": request_model})
            elif params:
                opts.update(request_validator=validator, request_parameters=params)
            resource.add_method(method, apigw.LambdaIntegration(fn), **opts)
        health = self.api.root.add_resource("health")
        health.add_method(
            "GET", apigw.MockIntegration(
                request_templates={"application/json": '{"statusCode": 200}'},
                integration_responses=[apigw.IntegrationResponse(
                    status_code="200", response_templates={"application/json": '{"data":{"status":"ok"},"error":null}'})]),
            authorization_type=apigw.AuthorizationType.NONE,
            method_responses=[apigw.MethodResponse(status_code="200")])
        experiments = self.api.root.add_resource("experiments")
        route(experiments, "POST", create_exp, exp_model)
        route(experiments, "GET", list_exp)
        route(experiments.add_resource("{experiment_id}").add_resource("compare"), "POST", compare, cmp_model)
        runs = self.api.root.add_resource("runs")
        route(runs, "POST", submit, run_model)
        route(runs, "GET", list_runs)
        run = runs.add_resource("{run_id}")
        route(run, "GET", get_run)
        route(run.add_resource("cancel"), "POST", cancel)
        route(run.add_resource("metrics"), "GET", metrics)
        route(run.add_resource("artifacts").add_resource("{kind}"), "GET", artifact)
        CfnOutput(self, "ApiUrl", value=self.api.url)
        CfnOutput(self, "UserPoolId", value=self.user_pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=self.client.user_pool_client_id)
