import json
import pytest
from typer.testing import CliRunner
import api_client
import rlctl

class FakeResp:
    # Keeps a fake status and body
    def __init__(self, status, body): self.status_code, self._b, self.text = status, body, json.dumps(body)
    # Returns the fake body
    def json(self): return self._b

# Saves a fake login
@pytest.fixture
def logged_in(tmp_path, monkeypatch):
    monkeypatch.setattr(api_client, "CONFIG_PATH", tmp_path / "cfg.json")
    api_client.save_config({"api": "https://api.test/v1", "token": "tok"})

# Checks the client reads data and errors
def test_client_envelope_and_errors(monkeypatch):
    calls = []
    # Fakes one HTTP call
    def fake(method, url, **kw):
        calls.append((method, url, kw["headers"]["Authorization"]))
        if url.endswith("/runs/bad"):
            return FakeResp(404, {"data": None, "error": {"code": "RUN_NOT_FOUND", "message": "nope"}})
        return FakeResp(200, {"data": {"run_id": "r1", "status": "QUEUED"}, "error": None})
    monkeypatch.setattr(api_client.requests, "request", fake)
    c = api_client.ApiClient("https://api.test/v1/", "tok")
    assert c.get_run("r1")["status"] == "QUEUED" and calls[0] == ("GET", "https://api.test/v1/runs/r1", "tok")
    with pytest.raises(api_client.ApiClientError) as e:
        c.get_run("bad")
    assert e.value.status == 404 and e.value.code == "RUN_NOT_FOUND"

# Checks the runs command output
def test_cli_runs_listing(logged_in, monkeypatch):
    monkeypatch.setattr(api_client.ApiClient, "list_runs", lambda self, *a, **k: {"items": [
        {"run_id": "r1", "status": "SUCCEEDED", "algo": "PPO", "env_id": "CartPole-v1", "current_step": 10, "final_reward": 99}]})
    out = CliRunner().invoke(rlctl.app, ["runs"])
    assert out.exit_code == 0 and "r1" in out.output and "SUCCEEDED" in out.output

# Checks that commands need a login
def test_cli_requires_login(tmp_path, monkeypatch):
    monkeypatch.setattr(api_client, "CONFIG_PATH", tmp_path / "none.json")
    monkeypatch.setattr(rlctl, "load_config", api_client.load_config)
    assert CliRunner().invoke(rlctl.app, ["runs"]).exit_code == 1
