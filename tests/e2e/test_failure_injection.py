import os
import time
import pytest
from api_client import ApiClient, ApiClientError

pytestmark = pytest.mark.e2e
API, TOKEN = os.environ.get("RL_API_URL"), os.environ.get("RL_ID_TOKEN")
if not (API and TOKEN):
    pytest.skip("set RL_API_URL and RL_ID_TOKEN to run e2e tests", allow_module_level=True)
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}

# Gives an API client
@pytest.fixture(scope="module")
def c():
    return ApiClient(API, TOKEN)

# Creates an experiment
@pytest.fixture(scope="module")
def exp(c):
    return c.create_experiment(f"e2e-{int(time.time())}")["experiment_id"]

# Builds a run config
def cfg(exp, steps=20000, **hp):
    return {"experiment_id": exp, "name": "e2e", "algo": "PPO", "env_id": "CartPole-v1", "seed": 1,
            "total_timesteps": steps, "hyperparams": {"n_steps": 128, "batch_size": 64, **hp}}

# Waits until a run matches a check
def wait_for(c, rid, pred, timeout=1500):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get_run(rid)
        if pred(r):
            return r
        time.sleep(10)
    raise TimeoutError(f"run {rid} stuck in {r['status']}")

# Checks that a bad config is rejected
def test_bad_config_returns_400(c, exp):
    with pytest.raises(ApiClientError) as e:
        c.submit_run({**cfg(exp), "env_id": "NotAnEnv-v0"})
    assert e.value.status == 400

# Checks that a crashing run ends as failed
def test_force_exception_ends_failed_after_three_attempts(c, exp):
    rid = c.submit_run(cfg(exp, batch_size=-1))["run_id"]
    r = wait_for(c, rid, lambda r: r["status"] in TERMINAL, timeout=2400)
    assert r["status"] == "FAILED" and r["attempt"] >= 3
    print("check: the DLQ alarm (rl-platform-DlqNotEmpty) and the run_failed email")

# Checks that a running job can be cancelled
def test_cancel_running_job(c, exp):
    rid = c.submit_run(cfg(exp, steps=2_000_000))["run_id"]
    wait_for(c, rid, lambda r: r["status"] == "RUNNING")
    c.cancel_run(rid)
    r = wait_for(c, rid, lambda r: r["status"] == "CANCELLED", timeout=300)
    assert r.get("latest_checkpoint_s3_uri")

# Checks that a killed worker run resumes from a checkpoint
@pytest.mark.skipif(not os.environ.get("RL_ASG_NAME"), reason="RL_ASG_NAME not set")
def test_kill_worker_mid_run_resumes_from_checkpoint(c, exp):
    import boto3
    rid = c.submit_run({**cfg(exp, steps=300_000), "logging": {"checkpoint_every_steps": 10000}})["run_id"]
    r = wait_for(c, rid, lambda r: r["status"] == "RUNNING" and r.get("latest_checkpoint_s3_uri"))
    boto3.client("ec2").terminate_instances(InstanceIds=[r["instance_id"]])
    r2 = wait_for(c, rid, lambda r: r["status"] in TERMINAL, timeout=3000)
    assert r2["status"] == "SUCCEEDED" and r2["attempt"] >= 2
    assert r2["instance_id"] != r["instance_id"]
