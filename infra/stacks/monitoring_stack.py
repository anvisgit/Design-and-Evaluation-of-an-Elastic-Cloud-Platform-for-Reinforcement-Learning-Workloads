from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_lambda_event_sources as sources
from aws_cdk import aws_logs as logs
from constructs import Construct
from lambda_helpers import make_function

NS = "RLPlatform"

class MonitoringStack(Stack):
    # Builds the notifier, alarms, dashboard and budget
    def __init__(self, scope: Construct, id: str, *, storage, messaging, api, compute,
                 monthly_budget_usd: float = 50, stage: str = "dev", **kw):
        super().__init__(scope, id, **kw)
        topic, queue, dlq = messaging.topic, messaging.queue, messaging.dlq
        action = cw_actions.SnsAction(topic)
        notifier = make_function(self, "Notifier", "notifier", env={
            "TOPIC_ARN": topic.topic_arn, "API_URL": api.api.url, "RUNS_TABLE": storage.runs.table_name})
        topic.grant_publish(notifier)
        notifier.add_to_role_policy(__import__("aws_cdk").aws_iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"], resources=["*"],
            conditions={"StringEquals": {"cloudwatch:namespace": NS}}))
        notifier.add_event_source(sources.DynamoEventSource(
            storage.runs, starting_position=_lambda.StartingPosition.LATEST, batch_size=10,
            retry_attempts=2, bisect_batch_on_error=True,
            filters=[_lambda.FilterCriteria.filter({"eventName": _lambda.FilterRule.is_equal("MODIFY")})]))
        worker_logs = logs.LogGroup.from_log_group_name(self, "WorkerLogs", "/rl-platform/worker")
        logs.MetricFilter(self, "WorkerErrorFilter", log_group=worker_logs, metric_namespace=NS,
                          metric_name="WorkerErrors", filter_pattern=logs.FilterPattern.any_term("ERROR"),
                          metric_value="1")
        m1, m5 = Duration.minutes(1), Duration.minutes(5)
        visible = queue.metric_approximate_number_of_messages_visible(period=m1, statistic="Maximum")
        age = queue.metric_approximate_age_of_oldest_message(period=m1, statistic="Maximum")
        dlq_depth = dlq.metric_approximate_number_of_messages_visible(period=m1, statistic="Maximum")
        in_service = cw.Metric(namespace="AWS/AutoScaling", metric_name="GroupInServiceInstances",
                               dimensions_map={"AutoScalingGroupName": compute.asg_name}, period=m1, statistic="Maximum")
        lambda_errors = cw.Metric(namespace="AWS/Lambda", metric_name="Errors", statistic="Sum", period=m5)
        lambda_duration = cw.Metric(namespace="AWS/Lambda", metric_name="Duration", statistic="Average", period=m5)
        api_5xx = api.api.metric_server_error(period=m5, statistic="Sum")
        api_4xx = api.api.metric_client_error(period=m5, statistic="Sum")
        api_latency = api.api.metric_latency(period=m5)
        failed_runs = cw.Metric(namespace=NS, metric_name="RunsByStatus", statistic="Sum",
                                dimensions_map={"Status": "FAILED"}, period=Duration.hours(1))
        cpu = cw.Metric(namespace="CWAgent", metric_name="cpu_usage_active", statistic="Average", period=m5,
                        dimensions_map={"AutoScalingGroupName": compute.asg_name})
        mem = cw.Metric(namespace="CWAgent", metric_name="mem_used_percent", statistic="Average", period=m5,
                        dimensions_map={"AutoScalingGroupName": compute.asg_name})
        ddb_throttles = cw.MathExpression(
            expression="SUM(SEARCH('{AWS/DynamoDB,TableName} MetricName=\"ThrottledRequests\"', 'Sum', 300))",
            label="DynamoDB throttles", period=m5)
        # Adds an alarm that sends to the topic
        def alarm(id_, metric, threshold, periods=1, op=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
                  desc="", datapoints=None):
            a = cw.Alarm(self, id_, alarm_name=f"rl-platform-{id_}", metric=metric, threshold=threshold,
                         evaluation_periods=periods, datapoints_to_alarm=datapoints or periods,
                         comparison_operator=op, treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
                         alarm_description=desc)
            a.add_alarm_action(action)
            return a
        alarm("DlqNotEmpty", dlq_depth, 0, desc="Messages in the training DLQ (a run failed 3 times)")
        alarm("OldestMessageAge", age, 30 * 60, desc="Oldest queued job is older than 30 minutes")
        no_workers = cw.MathExpression(expression="IF(jobs > 0 AND workers == 0, 1, 0)",
                                       using_metrics={"jobs": visible, "workers": in_service}, period=m1)
        alarm("QueuedButNoWorkers", no_workers, 0, periods=10,
              desc="Jobs are waiting but no worker has been in service for 10 minutes")
        alarm("LambdaErrors", lambda_errors, 3, desc="More than 3 Lambda errors in 5 minutes")
        alarm("Api5xx", api_5xx, 5, desc="More than 5 API 5XX in 5 minutes")
        alarm("RunFailures", failed_runs, 3, desc="More than 3 failed runs in 1 hour")
        alarm("WorkerCpuStuck", cpu, 95, periods=6, desc="Worker CPU above 95% for 30 minutes (stuck?)")
        reward_by_env = cw.MathExpression(
            expression="SEARCH('{RLPlatform,Algo,EnvId} MetricName=\"EpisodeRewardMean\"', 'Average', 300)",
            label="", period=m5)
        sps = cw.MathExpression(
            expression="SEARCH('{RLPlatform,Algo,EnvId} MetricName=\"StepsPerSecond\"', 'Average', 300)",
            label="", period=m5)
        runs_status = cw.MathExpression(
            expression="SEARCH('{RLPlatform,Status} MetricName=\"RunsByStatus\"', 'Sum', 3600)", label="", period=Duration.hours(1))
        dash = cw.Dashboard(self, "Dashboard", dashboard_name="RLPlatform")
        dash.add_widgets(
            cw.GraphWidget(title="Queue depth & age", left=[visible, queue.metric_approximate_number_of_messages_not_visible(period=m1)],
                           right=[age], width=8),
            cw.GraphWidget(title="DLQ depth", left=[dlq_depth], width=4),
            cw.GraphWidget(title="Running workers (ASG in service)", left=[in_service], width=6),
            cw.GraphWidget(title="Runs finished by status (1h)", left=[runs_status], width=6))
        dash.add_widgets(
            cw.GraphWidget(title="EpisodeRewardMean by env", left=[reward_by_env], width=12),
            cw.GraphWidget(title="StepsPerSecond", left=[sps], width=12))
        dash.add_widgets(
            cw.GraphWidget(title="Lambda errors", left=[lambda_errors], width=8),
            cw.GraphWidget(title="Lambda duration (avg ms)", left=[lambda_duration], width=8),
            cw.GraphWidget(title="DynamoDB throttles", left=[ddb_throttles], width=8))
        dash.add_widgets(
            cw.GraphWidget(title="API 4XX / 5XX", left=[api_4xx, api_5xx], width=12),
            cw.GraphWidget(title="API latency", left=[api_latency], width=12))
        dash.add_widgets(
            cw.GraphWidget(title="Worker CPU % / memory %", left=[cpu, mem], width=12),
            cw.GraphWidget(title="Worker ERROR log lines", left=[cw.Metric(
                namespace=NS, metric_name="WorkerErrors", statistic="Sum", period=m5)], width=12))
        B = budgets.CfnBudget
        self.budget = B(
            self, "MonthlyBudget",
            budget=B.BudgetDataProperty(
                budget_name=f"rl-platform-{stage}-monthly", budget_type="COST", time_unit="MONTHLY",
                budget_limit=B.SpendProperty(amount=monthly_budget_usd, unit="USD")),
            notifications_with_subscribers=[
                B.NotificationWithSubscribersProperty(
                    notification=B.NotificationProperty(notification_type="ACTUAL",
                                                        comparison_operator="GREATER_THAN",
                                                        threshold=pct, threshold_type="PERCENTAGE"),
                    subscribers=[B.SubscriberProperty(subscription_type="SNS", address=topic.topic_arn)])
                for pct in (50, 80, 100)])
        CfnOutput(self, "DashboardUrl", value=(
            f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}"
            "#dashboards:name=RLPlatform"))
