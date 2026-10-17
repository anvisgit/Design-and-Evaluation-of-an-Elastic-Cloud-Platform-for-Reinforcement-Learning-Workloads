from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_s3 as s3
from constructs import Construct

class StorageStack(Stack):
    # Builds the bucket, the tables and the image repo
    def __init__(self, scope: Construct, id: str, *, stage: str = "dev", **kw):
        super().__init__(scope, id, **kw)
        prod = stage == "prod"
        removal = RemovalPolicy.RETAIN if prod else RemovalPolicy.DESTROY
        self.bucket = s3.Bucket(
            self, "Artifacts", bucket_name=f"rl-platform-artifacts-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL, encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True, versioned=False, removal_policy=removal, auto_delete_objects=not prod,
            lifecycle_rules=[s3.LifecycleRule(
                id="checkpoints-tiering", prefix="checkpoints/",
                transitions=[s3.Transition(storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                                           transition_after=Duration.days(30))],
                expiration=Duration.days(90))])
        # Creates one DynamoDB table
        def table(id_, pk, sk=None, stream=False, ttl=None):
            return ddb.Table(
                self, id_, partition_key=ddb.Attribute(name=pk[0], type=pk[1]),
                sort_key=ddb.Attribute(name=sk[0], type=sk[1]) if sk else None,
                billing_mode=ddb.BillingMode.PAY_PER_REQUEST, removal_policy=removal,
                point_in_time_recovery_specification=ddb.PointInTimeRecoverySpecification(
                    point_in_time_recovery_enabled=True),
                stream=ddb.StreamViewType.NEW_AND_OLD_IMAGES if stream else None,
                time_to_live_attribute=ttl, encryption=ddb.TableEncryption.AWS_MANAGED)
        S, N = ddb.AttributeType.STRING, ddb.AttributeType.NUMBER
        self.experiments = table("Experiments", ("experiment_id", S))
        self.experiments.add_global_secondary_index(
            index_name="owner_index", partition_key=ddb.Attribute(name="owner", type=S),
            sort_key=ddb.Attribute(name="created_at", type=S))
        self.runs = table("Runs", ("run_id", S), stream=True, ttl="ttl")
        for name, pk, sk in (("experiment_index", "experiment_id", "created_at"),
                             ("status_index", "status", "last_heartbeat"),
                             ("owner_index", "owner", "created_at")):
            self.runs.add_global_secondary_index(
                index_name=name, partition_key=ddb.Attribute(name=pk, type=S),
                sort_key=ddb.Attribute(name=sk, type=S))
        self.metrics = table("Metrics", ("run_id", S), ("step", N), ttl="expires_at")
        self.workers = table("Workers", ("instance_id", S))
        self.repo = ecr.Repository(
            self, "WorkerRepo", repository_name="rl-platform-worker", image_scan_on_push=True,
            removal_policy=removal, empty_on_delete=not prod,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=20)])
        CfnOutput(self, "BucketName", value=self.bucket.bucket_name)
        CfnOutput(self, "RepositoryUri", value=self.repo.repository_uri)
