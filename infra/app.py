#!/usr/bin/env python3
import os
import aws_cdk as cdk
from stacks.api_stack import ApiStack
from stacks.compute_stack import ComputeStack
from stacks.messaging_stack import MessagingStack
from stacks.monitoring_stack import MonitoringStack
from stacks.storage_stack import StorageStack

# Builds all the stacks of the app
def build_app(context: dict | None = None, env: cdk.Environment | None = None) -> cdk.App:
    app = cdk.App(context=context)
    c = lambda k, d=None: app.node.try_get_context(k) if app.node.try_get_context(k) is not None else d
    env = env or cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                                 region=os.environ.get("CDK_DEFAULT_REGION"))
    stage = c("stage", "dev")
    storage = StorageStack(app, "StorageStack", stage=stage, env=env)
    messaging = MessagingStack(app, "MessagingStack", owner_email=c("owner_email", ""),
                               slack_webhook_url=c("slack_webhook_url", ""), env=env)
    api = ApiStack(app, "ApiStack", storage=storage, messaging=messaging, stage=stage,
                   max_active_runs=int(c("max_active_runs_per_user", 5)), env=env)
    compute = ComputeStack(app, "ComputeStack", storage=storage, messaging=messaging, stage=stage,
                           max_workers=int(c("max_workers", 5)), instance_type=c("instance_type", "c6i.xlarge"),
                           use_spot=str(c("use_spot", True)).lower() != "false",
                           image_tag=c("image_tag", "latest"), env=env)
    MonitoringStack(app, "MonitoringStack", storage=storage, messaging=messaging, api=api, compute=compute,
                    monthly_budget_usd=float(c("monthly_budget_usd", 50)), stage=stage, env=env)
    cdk.Tags.of(app).add("Project", "rl-platform")
    cdk.Tags.of(app).add("Stage", stage)
    return app

if __name__ == "__main__":
    build_app().synth()
