import os
import boto3
from common.log import logger

# Returns the worker count wanted, never lower than now
def compute_desired(visible: int, in_flight: int, current: int, max_workers: int) -> int:
    wanted = min(max_workers, visible + in_flight)
    return max(current, wanted)

# Raises the worker count from the queue size
@logger.inject_lambda_context
def handler(event, context):
    sqs, asg = boto3.client("sqs"), boto3.client("autoscaling")
    attrs = sqs.get_queue_attributes(
        QueueUrl=os.environ["QUEUE_URL"],
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
    )["Attributes"]
    visible, in_flight = int(attrs["ApproximateNumberOfMessages"]), int(attrs["ApproximateNumberOfMessagesNotVisible"])
    name = os.environ["ASG_NAME"]
    groups = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[name])["AutoScalingGroups"]
    if not groups:
        raise RuntimeError(f"ASG {name} not found")
    current = groups[0]["DesiredCapacity"]
    desired = compute_desired(visible, in_flight, current, int(os.environ["MAX_WORKERS"]))
    logger.info("scaler", extra={"visible": visible, "in_flight": in_flight,
                                 "current": current, "desired": desired})
    if desired != current:
        asg.set_desired_capacity(AutoScalingGroupName=name, DesiredCapacity=desired, HonorCooldown=False)
    return {"visible": visible, "in_flight": in_flight, "current": current, "desired": desired}
