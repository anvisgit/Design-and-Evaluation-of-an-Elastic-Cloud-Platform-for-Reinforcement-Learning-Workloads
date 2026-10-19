#!/usr/bin/env python3
import argparse, os, sys, time
from pathlib import Path
import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
from api_client import ApiClient

# Returns the workers in service and the max size
def in_service(asg, name):
    g = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[name])["AutoScalingGroups"][0]
    return len([i for i in g["Instances"] if i["LifecycleState"] == "InService"]), g["MaxSize"]

# Submits many runs and watches the fleet scale
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True); ap.add_argument("--token", default=os.environ.get("RL_ID_TOKEN"))
    ap.add_argument("--asg", required=True); ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--region", default=None)
    a = ap.parse_args()
    c, asg = ApiClient(a.api, a.token), boto3.client("autoscaling", region_name=a.region)
    exp = c.create_experiment(f"load-{int(time.time())}")["experiment_id"]
    ids = []
    for i in range(a.runs):
        cfg = {"experiment_id": exp, "name": f"load_{i}", "algo": "PPO", "env_id": "CartPole-v1", "seed": i,
               "total_timesteps": 10000, "hyperparams": {"n_steps": 128, "batch_size": 64}}
        while True:
            try:
                ids.append(c.submit_run(cfg)["run_id"]); break
            except Exception as e:
                if "QUOTA_EXCEEDED" not in str(e): raise
                time.sleep(20)
    peak, t0 = 0, time.time()
    while time.time() - t0 < 3600:
        n, mx = in_service(asg, a.asg)
        peak = max(peak, n)
        st = [c.get_run(r)["status"] for r in ids]
        print(f"workers={n}/{mx} peak={peak} " + " ".join(f"{s}={st.count(s)}" for s in sorted(set(st))))
        if all(s in ("SUCCEEDED", "FAILED", "CANCELLED") for s in st) and n == 0:
            break
        time.sleep(30)
    print(f"peak workers {peak}; final workers {in_service(asg, a.asg)[0]} (expect 0 after idle timeout)")

if __name__ == "__main__":
    main()
