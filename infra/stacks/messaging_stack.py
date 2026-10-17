from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from constructs import Construct

class MessagingStack(Stack):
    # Builds the job queue and the notification topic
    def __init__(self, scope: Construct, id: str, *, owner_email: str = "", slack_webhook_url: str = "", **kw):
        super().__init__(scope, id, **kw)
        self.dlq = sqs.Queue(self, "TrainingJobsDlq", queue_name="training_jobs_dlq",
                             retention_period=Duration.days(14), encryption=sqs.QueueEncryption.SQS_MANAGED,
                             enforce_ssl=True)
        self.queue = sqs.Queue(
            self, "TrainingJobs", queue_name="training_jobs", visibility_timeout=Duration.minutes(15),
            receive_message_wait_time=Duration.seconds(20), encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True, dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=3, queue=self.dlq))
        self.topic = sns.Topic(self, "Notifications", topic_name="rl_notifications")
        if owner_email:
            self.topic.add_subscription(subs.EmailSubscription(owner_email))
        if slack_webhook_url:
            self.topic.add_subscription(subs.UrlSubscription(slack_webhook_url))
        self.topic.add_to_resource_policy(iam.PolicyStatement(
            actions=["sns:Publish"], resources=[self.topic.topic_arn],
            principals=[iam.ServicePrincipal("budgets.amazonaws.com")],
            conditions={"StringEquals": {"aws:SourceAccount": self.account}}))
        CfnOutput(self, "QueueUrl", value=self.queue.queue_url)
        CfnOutput(self, "TopicArn", value=self.topic.topic_arn)
