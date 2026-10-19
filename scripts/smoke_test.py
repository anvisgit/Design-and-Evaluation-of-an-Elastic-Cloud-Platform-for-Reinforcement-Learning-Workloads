#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
from api_client import ApiClient, cognito_login

# Prints a step title
def step(msg): print(f"\n== {msg}", flush=True)

# Runs the full check against a live stack
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--token", default=os.environ.get("RL_ID_TOKEN"))
    ap.add_argument("--client-id"); ap.add_argument("--region"); ap.add_argument("--username"); ap.add_argument("--password")
    ap.add_argument("--runs", type=int, default=1, help="use 3 to check the compare acceptance criterion")
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--timeout-min", type=int, default=25)
    a = ap.parse_args()
    token = a.token or cognito_login(a.client_id, a.username, a.password, a.region)
    c = ApiClient(a.api, token)
    step("health (no auth)")
    print(requests.get(a.api.rstrip("/") + "/health", timeout=10).json())
    step("create experiment")
    exp = c.create_experiment(f"smoke-{int(time.time())}", "smoke test", ["smoke"])
    print(exp["experiment_id"])
    step(f"submit {a.runs} run(s)")
    ids = []
    for i in range(a.runs):
        cfg = {"experiment_id": exp["experiment_id"], "name": f"smoke_ppo_{i}", "algo": "PPO",
               "env_id": "CartPole-v1", "seed": i, "total_timesteps": a.timesteps, "n_envs": 2,
               "hyperparams": {"n_steps": 256, "batch_size": 64, "policy": "MlpPolicy"},
               "logging": {"metrics_every_steps": 1000, "checkpoint_every_steps": 5000,
                           "eval_every_steps": 10000, "eval_episodes": 5},
               "resources": {"max_runtime_minutes": 20}}
        t0 = time.time()
        res = c.submit_run(cfg)
        dt = time.time() - t0
        assert res["status"] == "QUEUED", res
        assert dt < 1.0 or i == 0, f"POST /runs took {dt:.2f}s"
        print(f"run {res['run_id']} queued in {dt:.2f}s")
        ids.append(res["run_id"])
    step("wait for completion (scale-up from zero takes a few minutes)")
    deadline, seen = time.time() + a.timeout_min * 60, {}
    while time.time() < deadline:
        runs = {r: c.get_run(r) for r in ids}
        for r, v in runs.items():
            if seen.get(r) != v["status"]:
                print(f"  {r[:8]} -> {v['status']}")
                seen[r] = v["status"]
        if all(v["status"] in ("SUCCEEDED", "FAILED", "CANCELLED") for v in runs.values()):
            break
        time.sleep(15)
    bad = {r: v for r, v in runs.items() if v["status"] != "SUCCEEDED"}
    assert not bad, f"runs did not succeed: { {r: (v['status'], v.get('error')) for r, v in bad.items()} }"
    step("metrics and artifacts")
    for r in ids:
        rows = c.all_metrics(r)
        assert rows and "ep_rew_mean" in rows[-1], f"no metrics for {r}"
        print(f"  {r[:8]}: {len(rows)} metric rows, final reward {runs[r].get('final_reward')}")
        url = c.artifact_url(r, "model")["url"]
        assert requests.get(url, timeout=60).status_code == 200
    step("compare")
    cmp_ = c.compare(exp["experiment_id"], ids)
    print("ranking:", cmp_["ranking"])
    assert len(cmp_["steps"]) > 0 and set(cmp_["series"]) == set(ids)
    print("\nSMOKE TEST PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
