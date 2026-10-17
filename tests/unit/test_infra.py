import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template
from app import build_app

ENV = cdk.Environment(account="123456789012", region="us-east-1")

# Builds the templates of all stacks
@pytest.fixture(scope="module")
def tpl():
    app = build_app({"stage": "dev", "owner_email": "me@example.com", "max_workers": 5}, ENV)
    return {n: Template.from_stack(app.node.find_child(n)) for n in
            ("StorageStack", "MessagingStack", "ApiStack", "ComputeStack", "MonitoringStack")}, app

# Checks the storage stack
def test_storage(tpl):
    t = tpl[0]["StorageStack"]
    t.resource_count_is("AWS::DynamoDB::Table", 4)
    t.has_resource_properties("AWS::DynamoDB::Table", {
        "KeySchema": [{"AttributeName": "run_id", "KeyType": "HASH"}],
        "StreamSpecification": {"StreamViewType": "NEW_AND_OLD_IMAGES"},
        "BillingMode": "PAY_PER_REQUEST",
        "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
        "GlobalSecondaryIndexes": Match.array_with([
            Match.object_like({"IndexName": "experiment_index"}), Match.object_like({"IndexName": "status_index"})])})
    t.has_resource_properties("AWS::DynamoDB::Table", {
        "KeySchema": [{"AttributeName": "run_id", "KeyType": "HASH"}, {"AttributeName": "step", "KeyType": "RANGE"}],
        "TimeToLiveSpecification": {"AttributeName": "expires_at", "Enabled": True}})
    t.has_resource_properties("AWS::S3::Bucket", {
        "PublicAccessBlockConfiguration": {"BlockPublicAcls": True, "BlockPublicPolicy": True,
                                           "IgnorePublicAcls": True, "RestrictPublicBuckets": True},
        "LifecycleConfiguration": {"Rules": [Match.object_like({"Prefix": "checkpoints/", "ExpirationInDays": 90})]}})
    t.has_resource_properties("AWS::S3::BucketPolicy", {"PolicyDocument": {"Statement": Match.array_with([
        Match.object_like({"Effect": "Deny", "Condition": {"Bool": {"aws:SecureTransport": "false"}}})])}})

# Checks the messaging stack
def test_messaging(tpl):
    t = tpl[0]["MessagingStack"]
    t.has_resource_properties("AWS::SQS::Queue", {
        "QueueName": "training_jobs", "VisibilityTimeout": 900, "ReceiveMessageWaitTimeSeconds": 20,
        "RedrivePolicy": Match.object_like({"maxReceiveCount": 3})})
    t.has_resource_properties("AWS::SQS::Queue", {"QueueName": "training_jobs_dlq", "MessageRetentionPeriod": 14 * 86400})
    t.has_resource_properties("AWS::SNS::Topic", {"TopicName": "rl_notifications"})
    t.has_resource_properties("AWS::SNS::Subscription", {"Protocol": "email", "Endpoint": "me@example.com"})

# Checks the API stack
def test_api(tpl):
    t = tpl[0]["ApiStack"]
    t.has_resource_properties("AWS::Cognito::UserPool", {"AdminCreateUserConfig": {"AllowAdminCreateUserOnly": True}})
    t.has_resource_properties("AWS::ApiGateway::Authorizer", {"Type": "COGNITO_USER_POOLS"})
    t.resource_count_is("AWS::Lambda::Function", 9)
    t.has_resource_properties("AWS::Lambda::Function", {"Runtime": "python3.11", "Architectures": ["arm64"],
                                                         "TracingConfig": {"Mode": "Active"}, "MemorySize": 256})
    t.has_resource_properties("AWS::ApiGateway::UsagePlan", {"Quota": {"Limit": 10000, "Period": "DAY"},
                                                              "Throttle": {"RateLimit": 20, "BurstLimit": 40}})
    t.has_resource_properties("AWS::ApiGateway::Resource", {"PathPart": "{kind}"})
    methods = t.find_resources("AWS::ApiGateway::Method")
    auths = {(m["Properties"]["HttpMethod"], m["Properties"]["AuthorizationType"]) for m in methods.values()}
    assert ("GET", "NONE") in auths and ("POST", "COGNITO_USER_POOLS") in auths
    t.has_resource_properties("AWS::ApiGateway::Method", {"HttpMethod": "POST", "RequestValidatorId": Match.any_value()})

# Checks that submit cannot read models
def test_least_privilege_submit_cannot_touch_models(tpl):
    t = tpl[0]["ApiStack"]
    pols = t.find_resources("AWS::IAM::Policy")
    submit = [p for k, p in pols.items() if k.startswith("SubmitRun")]
    assert submit, "SubmitRun policy missing"
    assert "s3:GetObject*" not in str(submit[0]["Properties"]["PolicyDocument"])

# Checks the compute stack
def test_compute(tpl):
    t = tpl[0]["ComputeStack"]
    t.has_resource_properties("AWS::AutoScaling::AutoScalingGroup", {
        "MinSize": "0", "MaxSize": "5", "CapacityRebalance": True,
        "MixedInstancesPolicy": Match.object_like({"InstancesDistribution": Match.object_like({
            "OnDemandPercentageAboveBaseCapacity": 0, "SpotAllocationStrategy": "price-capacity-optimized"})})})
    t.has_resource_properties("AWS::EC2::LaunchTemplate", {"LaunchTemplateData": Match.object_like({
        "MetadataOptions": Match.object_like({"HttpTokens": "required", "HttpPutResponseHopLimit": 2}),
        "BlockDeviceMappings": Match.array_with([Match.object_like({"Ebs": Match.object_like({"VolumeSize": 30, "VolumeType": "gp3"})})])})})
    sgs = t.find_resources("AWS::EC2::SecurityGroup")
    assert all("SecurityGroupIngress" not in v["Properties"] for v in sgs.values())
    t.has_resource_properties("AWS::Events::Rule", {"ScheduleExpression": "rate(1 minute)"})
    t.has_resource_properties("AWS::Events::Rule", {"ScheduleExpression": "rate(5 minutes)"})
    t.has_resource_properties("AWS::Logs::LogGroup", {"LogGroupName": "/rl-platform/worker", "RetentionInDays": 30})

# Checks the monitoring stack
def test_monitoring(tpl):
    t = tpl[0]["MonitoringStack"]
    t.resource_count_is("AWS::CloudWatch::Alarm", 7)
    t.has_resource_properties("AWS::CloudWatch::Dashboard", {"DashboardName": "RLPlatform"})
    t.has_resource_properties("AWS::Logs::MetricFilter", {"FilterPattern": "\"ERROR\""})
    t.has_resource_properties("AWS::Lambda::EventSourceMapping", {"StartingPosition": "LATEST"})
    budgets = t.find_resources("AWS::Budgets::Budget")
    b = next(iter(budgets.values()))["Properties"]
    assert [n["Notification"]["Threshold"] for n in b["NotificationsWithSubscribers"]] == [50, 80, 100]
    assert b["Budget"]["BudgetLimit"]["Amount"] == 50

# Checks the stack tags
def test_every_stack_tagged(tpl):
    _, app = tpl
    for name in ("StorageStack", "MessagingStack", "ApiStack", "ComputeStack", "MonitoringStack"):
        stack = app.node.find_child(name)
        assert cdk.Tags is not None
        assert any(t == "Project" for t in [k for k in cdk.Stack.of(stack).tags.tag_values().keys()]) or \
            cdk.Stack.of(stack).tags.tag_values().get("Project") == "rl-platform"
