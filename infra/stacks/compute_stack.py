from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, Tags
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from constructs import Construct
from lambda_helpers import make_function

WORKER_LOG_GROUP = "/rl-platform/worker"
CW_AGENT_CONFIG = (
    '{"metrics":{"append_dimensions":{"AutoScalingGroupName":"${aws:AutoScalingGroupName}"},'
    '"metrics_collected":{"cpu":{"measurement":["cpu_usage_active"],"metrics_collection_interval":60,'
    '"totalcpu":true},"mem":{"measurement":["mem_used_percent"],"metrics_collection_interval":60}}}}')

class ComputeStack(Stack):
    # Builds the worker fleet, the scaler and the watchdog
    def __init__(self, scope: Construct, id: str, *, storage, messaging, stage: str = "dev",
                 max_workers: int = 5, instance_type: str = "c6i.xlarge", use_spot: bool = True,
                 image_tag: str = "latest", **kw):
        super().__init__(scope, id, **kw)
        self.asg_name = f"rl-platform-{stage}-workers"
        asg_arn = f"arn:aws:autoscaling:{self.region}:{self.account}:autoScalingGroup:*:autoScalingGroupName/{self.asg_name}"
        self.worker_logs = logs.LogGroup(self, "WorkerLogs", log_group_name=WORKER_LOG_GROUP,
                                         retention=logs.RetentionDays.ONE_MONTH,
                                         removal_policy=RemovalPolicy.DESTROY)
        vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=0,
                      subnet_configuration=[ec2.SubnetConfiguration(
                          name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)])
        sg = ec2.SecurityGroup(self, "WorkerSg", vpc=vpc, allow_all_outbound=True,
                               description="rl-platform workers: outbound only")
        role = iam.Role(self, "WorkerRole", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
                        managed_policies=[
                            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                            iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy")])
        messaging.queue.grant_consume_messages(role)
        for t in (storage.runs, storage.metrics, storage.workers):
            t.grant_read_write_data(role)
        storage.bucket.grant_put(role)
        storage.bucket.grant_read(role)
        storage.repo.grant_pull(role)
        role.add_to_policy(iam.PolicyStatement(actions=["cloudwatch:PutMetricData"], resources=["*"],
                                               conditions={"StringEquals": {"cloudwatch:namespace": "RLPlatform"}}))
        role.add_to_policy(iam.PolicyStatement(
            actions=["autoscaling:SetInstanceProtection", "autoscaling:TerminateInstanceInAutoScalingGroup"],
            resources=[asg_arn]))
        self.worker_logs.grant_write(role)
        image = f"{storage.repo.repository_uri}:{image_tag}"
        ud = ec2.UserData.for_linux()
        ud.add_commands(
            "set -euxo pipefail",
            "dnf install -y docker amazon-cloudwatch-agent",
            "systemctl enable --now docker",
            f"cat > /opt/cw-agent.json <<'JSON'\n{CW_AGENT_CONFIG}\nJSON",
            "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 "
            "-c file:/opt/cw-agent.json -s",
            f"aws ecr get-login-password --region {self.region} | docker login --username AWS "
            f"--password-stdin {self.account}.dkr.ecr.{self.region}.amazonaws.com",
            f"docker pull {image}",
            "docker run -d --name rl-worker --restart unless-stopped "
            f"--log-driver awslogs --log-opt awslogs-region={self.region} "
            f"--log-opt awslogs-group={WORKER_LOG_GROUP} --log-opt awslogs-create-group=false "
            f"-e AWS_REGION={self.region} -e AWS_DEFAULT_REGION={self.region} "
            f"-e RUNS_TABLE={storage.runs.table_name} -e METRICS_TABLE={storage.metrics.table_name} "
            f"-e WORKERS_TABLE={storage.workers.table_name} -e BUCKET={storage.bucket.bucket_name} "
            f"-e QUEUE_URL={messaging.queue.queue_url} -e ASG_NAME={self.asg_name} {image}")
        lt = ec2.LaunchTemplate(
            self, "WorkerLt", machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            instance_type=ec2.InstanceType(instance_type), role=role, security_group=sg, user_data=ud,
            require_imdsv2=True, http_put_response_hop_limit=2,
            block_devices=[ec2.BlockDevice(device_name="/dev/xvda", volume=ec2.BlockDeviceVolume.ebs(
                30, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True, delete_on_termination=True))])
        overrides = [instance_type] + [t for t in ("c6i.xlarge", "c5.xlarge", "m5.xlarge") if t != instance_type]
        self.asg = autoscaling.AutoScalingGroup(
            self, "Workers", auto_scaling_group_name=self.asg_name, vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            min_capacity=0, max_capacity=max_workers, desired_capacity=0,
            capacity_rebalance=use_spot, group_metrics=[autoscaling.GroupMetrics.all()],
            mixed_instances_policy=autoscaling.MixedInstancesPolicy(
                launch_template=lt,
                instances_distribution=autoscaling.InstancesDistribution(
                    on_demand_base_capacity=0,
                    on_demand_percentage_above_base_capacity=0 if use_spot else 100,
                    spot_allocation_strategy=autoscaling.SpotAllocationStrategy.PRICE_CAPACITY_OPTIMIZED),
                launch_template_overrides=[autoscaling.LaunchTemplateOverrides(instance_type=ec2.InstanceType(t))
                                           for t in overrides]))
        Tags.of(self.asg).add("Name", f"rl-platform-{stage}-worker", apply_to_launched_instances=True)
        scaler = make_function(self, "Scaler", "scaler", env={
            "QUEUE_URL": messaging.queue.queue_url, "ASG_NAME": self.asg_name, "MAX_WORKERS": str(max_workers)})
        messaging.queue.grant(scaler, "sqs:GetQueueAttributes")
        scaler.add_to_role_policy(iam.PolicyStatement(actions=["autoscaling:DescribeAutoScalingGroups"], resources=["*"]))
        scaler.add_to_role_policy(iam.PolicyStatement(actions=["autoscaling:SetDesiredCapacity"], resources=[asg_arn]))
        events.Rule(self, "ScalerSchedule", schedule=events.Schedule.rate(Duration.minutes(1)),
                    targets=[targets.LambdaFunction(scaler)])
        watchdog = make_function(self, "Watchdog", "watchdog", timeout_s=60, env={
            "RUNS_TABLE": storage.runs.table_name, "WORKERS_TABLE": storage.workers.table_name,
            "QUEUE_URL": messaging.queue.queue_url})
        storage.runs.grant_read_write_data(watchdog)
        storage.workers.grant_read_write_data(watchdog)
        messaging.queue.grant_send_messages(watchdog)
        events.Rule(self, "WatchdogSchedule", schedule=events.Schedule.rate(Duration.minutes(5)),
                    targets=[targets.LambdaFunction(watchdog)])
        CfnOutput(self, "AsgName", value=self.asg_name)
        CfnOutput(self, "WorkerImage", value=image)
