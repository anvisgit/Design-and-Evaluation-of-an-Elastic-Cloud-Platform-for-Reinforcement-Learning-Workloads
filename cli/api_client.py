from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional
import requests

CONFIG_PATH = Path(os.environ.get("RLCTL_CONFIG", Path.home() / ".rlctl.json"))

class ApiClientError(Exception):
    # Keeps the status, code and message of an API error
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message

# Logs in to Cognito and returns the ID token
def cognito_login(client_id: str, username: str, password: str, region: str) -> str:
    import boto3
    idp = boto3.client("cognito-idp", region_name=region)
    resp = idp.initiate_auth(ClientId=client_id, AuthFlow="USER_PASSWORD_AUTH",
                             AuthParameters={"USERNAME": username, "PASSWORD": password})
    return resp["AuthenticationResult"]["IdToken"]

# Reads the saved login settings
def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

# Saves the login settings and locks the file
def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    CONFIG_PATH.chmod(0o600)

class ApiClient:
    # Sets the base url and the auth headers
    def __init__(self, base_url: str, token: str, timeout: float = 30):
        self.base, self.timeout = base_url.rstrip("/"), timeout
        self.h = {"Authorization": token, "Content-Type": "application/json"}
    # Sends a request and returns the data or raises an error
    def _req(self, method: str, path: str, **kw) -> Any:
        r = requests.request(method, f"{self.base}{path}", headers={**self.h, **kw.pop("headers", {})},
                             timeout=self.timeout, **kw)
        body = r.json()
        if r.status_code >= 400 or body.get("error"):
            err = body.get("error") or {"code": "HTTP", "message": r.text}
            raise ApiClientError(r.status_code, err.get("code", "ERROR"), err.get("message", ""))
        return body["data"]
    # Checks that the API is up
    def health(self): return self._req("GET", "/health")
    # Creates a new experiment
    def create_experiment(self, name, description="", tags=None):
        return self._req("POST", "/experiments", json={"name": name, "description": description, "tags": tags or []})
    # Lists experiments one page at a time
    def list_experiments(self, limit=50, cursor=None):
        return self._req("GET", "/experiments", params={"limit": limit, **({"cursor": cursor} if cursor else {})})
    # Sends a new training run
    def submit_run(self, config: dict, idempotency_key: Optional[str] = None):
        return self._req("POST", "/runs", json=config,
                         headers={"Idempotency-Key": idempotency_key} if idempotency_key else {})
    # Gets one run
    def get_run(self, run_id): return self._req("GET", f"/runs/{run_id}")
    # Lists runs with optional filters
    def list_runs(self, experiment_id=None, status=None, limit=50, cursor=None):
        p = {k: v for k, v in dict(experiment_id=experiment_id, status=status, limit=limit, cursor=cursor).items() if v}
        return self._req("GET", "/runs", params=p)
    # Asks to cancel a run
    def cancel_run(self, run_id): return self._req("POST", f"/runs/{run_id}/cancel")
    # Gets one page of metrics
    def metrics(self, run_id, from_step=None, to_step=None, limit=1000, cursor=None):
        p = {k: v for k, v in dict(from_step=from_step, to_step=to_step, limit=limit, cursor=cursor).items() if v is not None}
        return self._req("GET", f"/runs/{run_id}/metrics", params=p)
    # Gets every page of metrics
    def all_metrics(self, run_id) -> list[dict]:
        items, cursor = [], None
        while True:
            page = self.metrics(run_id, limit=5000, cursor=cursor)
            items += page["items"]
            cursor = page["next_cursor"]
            if not cursor:
                return items
    # Compares several runs
    def compare(self, experiment_id, run_ids, metric="ep_rew_mean", threshold=None):
        body = {"run_ids": run_ids, "metric": metric, **({"threshold": threshold} if threshold is not None else {})}
        return self._req("POST", f"/experiments/{experiment_id}/compare", json=body)
    # Gets a download link for a run file
    def artifact_url(self, run_id, kind): return self._req("GET", f"/runs/{run_id}/artifacts/{kind}")
