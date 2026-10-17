#!/usr/bin/env python3
from __future__ import annotations
import functools
import json
import sys
import time
from pathlib import Path
from typing import Optional
import typer

sys.path.insert(0, str(Path(__file__).parent))
from api_client import ApiClient, ApiClientError, cognito_login, load_config, save_config

app = typer.Typer(help="RL training platform CLI", no_args_is_help=True)
experiments = typer.Typer(help="Manage experiments", no_args_is_help=True)
app.add_typer(experiments, name="experiments")
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}

# Builds an API client from the saved login
def client() -> ApiClient:
    cfg = load_config()
    if not cfg.get("token"):
        typer.echo("Not logged in. Run: rlctl login", err=True)
        raise typer.Exit(1)
    return ApiClient(cfg["api"], cfg["token"])

# Prints data as JSON
def show(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))

# Turns API errors into a clean exit
def guarded(fn):
    # Runs the command and catches API errors
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except ApiClientError as e:
            typer.echo(f"error: {e}", err=True)
            if e.status == 401:
                typer.echo("token expired? run `rlctl login` again", err=True)
            raise typer.Exit(1)
    return wrapper

# Logs in and saves the token
@app.command()
def login(api: str = typer.Option(..., help="API URL (ApiUrl stack output)"),
          client_id: str = typer.Option(..., help="UserPoolClientId stack output"),
          region: str = typer.Option(..., help="AWS region of the user pool"),
          username: str = typer.Option(...),
          password: str = typer.Option(..., prompt=True, hide_input=True)):
    token = cognito_login(client_id, username, password, region)
    save_config({"api": api, "client_id": client_id, "region": region, "username": username, "token": token})
    typer.echo("logged in")

# Creates an experiment
@experiments.command("create")
@guarded
def exp_create(name: str, description: str = "", tag: list[str] = typer.Option([], "--tag")):
    show(client().create_experiment(name, description, tag))

# Prints all experiments
@experiments.command("list")
@guarded
def exp_list(limit: int = 50):
    for e in client().list_experiments(limit)["items"]:
        typer.echo(f"{e['experiment_id']}  {e['name']:<30} runs={e.get('run_count', 0)}  {e['created_at']}")

# Submits a run from a JSON file
@app.command()
@guarded
def submit(config_file: Path, wait: bool = typer.Option(False, "--wait", help="poll until finished"),
           idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key")):
    c = client()
    res = c.submit_run(json.loads(config_file.read_text()), idempotency_key)
    show(res)
    if wait:
        watch_run(c, res["run_id"])

# Prints run progress until it ends
def watch_run(c: ApiClient, run_id: str, interval: float = 10) -> None:
    while True:
        r = c.get_run(run_id)
        typer.echo(f"{r['status']:<17} step={r.get('current_step', 0)}/{r['total_timesteps']} "
                   f"best={r.get('best_reward')}")
        if r["status"] in TERMINAL:
            raise typer.Exit(0 if r["status"] == "SUCCEEDED" else 2)
        time.sleep(interval)

# Prints a list of runs
@app.command()
@guarded
def runs(experiment_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50):
    for r in client().list_runs(experiment_id, status, limit)["items"]:
        typer.echo(f"{r['run_id']}  {r['status']:<17} {r['algo']:<4} {r['env_id']:<16} "
                   f"step={r.get('current_step', 0)} final={r.get('final_reward')}")

# Prints one run
@app.command()
@guarded
def get(run_id: str):
    show(client().get_run(run_id))

# Follows a run until it ends
@app.command()
@guarded
def watch(run_id: str, interval: float = 10):
    watch_run(client(), run_id, interval)

# Cancels a run
@app.command()
@guarded
def cancel(run_id: str):
    show(client().cancel_run(run_id))

# Prints the metrics of a run
@app.command()
@guarded
def metrics(run_id: str, from_step: Optional[int] = None, to_step: Optional[int] = None, csv: bool = False):
    rows = client().all_metrics(run_id)
    rows = [r for r in rows if (from_step is None or r["step"] >= from_step) and (to_step is None or r["step"] <= to_step)]
    if csv:
        typer.echo("step,ep_rew_mean,ep_len_mean,loss,fps")
        for r in rows:
            typer.echo(",".join(str(r.get(k, "")) for k in ("step", "ep_rew_mean", "ep_len_mean", "loss", "fps")))
    else:
        show(rows)

# Prints a comparison of runs
@app.command()
@guarded
def compare(experiment_id: str, run_ids: list[str], metric: str = "ep_rew_mean",
            threshold: Optional[float] = None):
    res = client().compare(experiment_id, run_ids, metric, threshold)
    for s in res["summary"]:
        typer.echo(f"{s['run_id']}  best={s['best']}  final={s['final']}  auc={s['auc']}  to_threshold={s['steps_to_threshold']}")
    typer.echo("ranking: " + " > ".join(res["ranking"]))

# Downloads a run file
@app.command()
@guarded
def download(run_id: str, kind: str = typer.Argument("model", help="model|checkpoint|log|config"),
             out: Optional[Path] = None):
    import requests
    url = client().artifact_url(run_id, kind)["url"]
    out = out or Path(f"{run_id}_{kind}{'.zip' if kind in ('model', 'checkpoint') else '.json' if kind == 'config' else '.log'}")
    out.write_bytes(requests.get(url, timeout=120).content)
    typer.echo(f"saved {out}")

if __name__ == "__main__":
    app()
