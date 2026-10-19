from aws_lambda_powertools import Logger, Tracer

logger = Logger(service="rl-platform")
tracer = Tracer(service="rl-platform")
