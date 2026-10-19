from __future__ import annotations
import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import ValidationError
from aws_clients import RunStore, Settings, TERMINAL, heartbeat_age
from runstate import RunState
from spot import SpotWatcher

log = logging.getLogger("worker")
MAX_ATTEMPTS = 3
HEARTBEAT_S = 30
LOG_UPLOAD_S = 300

# Returns the current UTC time as text
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Returns the git sha and library versions
def provenance() -> dict:
    versions = {"python": platform.python_version()}
    for pkg in ("torch", "stable_baselines3", "gymnasium", "numpy"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    return {"git_sha": os.environ.get("GIT_SHA", "unknown"), "library_versions": versions}

class Tee:
    # Opens the log file
    def __init__(self, stream, path: Path):
        self.stream, self.f = stream, open(path, "a", buffering=1)
    # Writes to the screen and the log file
    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
    # Flushes both outputs
    def flush(self):
        self.stream.flush()
        self.f.flush()
    # Closes the log file
    def close(self):
        self.f.close()

class Heartbeat(threading.Thread):
    # Keeps what the heartbeat needs
    def __init__(self, store: RunStore, run_id: str, receipt: str, state: RunState, log_path: Path,
                 interval: float = HEARTBEAT_S):
        super().__init__(daemon=True, name="heartbeat")
        self.store, self.run_id, self.receipt, self.state = store, run_id, receipt, state
        self.log_path, self.interval, self._done = log_path, interval, threading.Event()
        self._last_upload = time.monotonic()
    # Does one heartbeat
    def beat(self):
        self.store.heartbeat(self.run_id, self.state.current_step)
        self.store.extend_visibility(self.receipt, 600)
        self.store.register_worker("BUSY", self.run_id)
        if time.monotonic() - self._last_upload >= LOG_UPLOAD_S:
            self.upload_log()
    # Uploads the log file
    def upload_log(self):
        try:
            if self.log_path.exists():
                self.store.upload(self.log_path, f"logs/{self.run_id}/train.log")
            self._last_upload = time.monotonic()
        except Exception:
            log.warning("log upload failed", exc_info=True)
    # Beats again and again until stopped
    def run(self):
        while not self._done.wait(self.interval):
            try:
                self.beat()
            except Exception:
                log.warning("heartbeat failed", exc_info=True)
    # Stops the heartbeat
    def stop(self):
        self._done.set()

class Worker:
    # Sets up the settings and the store
    def __init__(self, settings: Optional[Settings] = None, store: Optional[RunStore] = None, trainer=None):
        self.s = settings or Settings()
        self.store = store or RunStore(self.s)
        self.interrupt = threading.Event()
        self._trainer = trainer
        self.heartbeat_interval = HEARTBEAT_S
        self.exit_requested = False
    # Loads the trainer when first needed
    @property
    def trainer(self):
        if self._trainer is None:
            import trainer
            self._trainer = trainer
        return self._trainer
    # Polls for jobs and shuts down when idle
    def run_forever(self):
        self.store.register_worker("IDLE")
        if not self.s.local_mode:
            SpotWatcher(self.interrupt).start()
        idle_since = time.monotonic()
        while not self.exit_requested and not self.interrupt.is_set():
            if self.poll_once():
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= self.s.idle_timeout_s:
                log.info("idle for %ss with empty queue: shutting down", self.s.idle_timeout_s)
                self.store.register_worker("TERMINATING")
                self.store.terminate_self()
                return
            self.store.register_worker("IDLE")
    # Gets one message and handles it
    def poll_once(self, wait: int = 20) -> bool:
        resp = self.store.sqs.receive_message(
            QueueUrl=self.s.queue_url, WaitTimeSeconds=wait, MaxNumberOfMessages=1,
            AttributeNames=["ApproximateReceiveCount"])
        msgs = resp.get("Messages", [])
        if not msgs:
            return False
        self.handle_message(msgs[0])
        return True
    # Claims a run, trains it and reports the result
    def handle_message(self, msg: dict) -> str:
        body, receipt = json.loads(msg["Body"]), msg["ReceiptHandle"]
        receive_count = int(msg.get("Attributes", {}).get("ApproximateReceiveCount", 1))
        run_id = body["run_id"]
        run = self.store.get_run(run_id)
        if run is None or run["status"] in TERMINAL:
            self.store.delete_message(receipt)
            return "dropped"
        if run.get("cancel_requested") or run["status"] == "CANCEL_REQUESTED":
            self.store.update(run_id, {"status": "CANCELLED", "ended_at": now_iso()})
            self.store.delete_message(receipt)
            return "cancelled"
        if not self.store.claim(run, receive_count, provenance()):
            if run["status"] == "RUNNING" and heartbeat_age(run) < 600:
                self.store.delete_message(receipt)
                return "duplicate"
            return "skipped"
        run = self.store.get_run(run_id) or run
        attempt = int(run.get("attempt", 1))
        self.store.register_worker("BUSY", run_id)
        self.store.set_protection(True)
        workdir = Path(self.s.workdir) / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = workdir / "train.log"
        state = RunState()
        hb = Heartbeat(RunStore(self.s), run_id, receipt, state, log_path, self.heartbeat_interval)
        tee, real_stdout = Tee(sys.stdout, log_path), sys.stdout
        sys.stdout = tee
        fh = logging.FileHandler(log_path)
        logging.getLogger().addHandler(fh)
        hb.start()
        try:
            return self._execute(run, receipt, attempt, workdir, state, hb)
        finally:
            hb.stop()
            sys.stdout = real_stdout
            logging.getLogger().removeHandler(fh)
            fh.close()
            tee.close()
            hb.upload_log()
            self.store.set_protection(False)
            self.store.register_worker("IDLE")
    # Runs the training and saves the final status
    def _execute(self, run: dict, receipt: str, attempt: int, workdir: Path, state: RunState, hb: Heartbeat) -> str:
        from config_schema import TrainingConfig
        run_id = run["run_id"]
        try:
            cfg = TrainingConfig.model_validate_json(
                self.store.download(run["config_s3_uri"], workdir / "config.json").read_text())
            resume = None
            if run.get("latest_checkpoint_s3_uri"):
                resume = self.store.download(run["latest_checkpoint_s3_uri"], workdir / "resume.zip")
            result = self.trainer.train(cfg, run_id, self.store, state, self.interrupt, workdir, resume)
        except ValidationError as e:
            self.store.update_mine(run_id, {"status": "FAILED", "ended_at": now_iso(),
                                            "error": f"invalid_config: {e}"[:3000]})
            self.store.delete_message(receipt)
            return "failed"
        except Exception:
            err = traceback.format_exc()[-3000:]
            log.error("training failed:\n%s", err)
            if attempt >= MAX_ATTEMPTS:
                self.store.update_mine(run_id, {"status": "FAILED", "ended_at": now_iso(), "error": err})
                return "failed"
            self.store.update_mine(run_id, {"status": "REQUEUED", "error": err, "attempt": attempt})
            hb.stop(); hb.join(10)
            self.store.extend_visibility(receipt, self.s.retry_delay_s)
            return "retry"
        if result.status == "finished":
            self.store.update_mine(run_id, {
                "status": "SUCCEEDED", "ended_at": now_iso(), "final_reward": result.final_reward,
                "best_reward": result.best_reward, "model_s3_uri": result.model_uri,
                "current_step": result.steps, "last_heartbeat": now_iso()})
            self.store.delete_message(receipt)
            return "succeeded"
        if result.status == "cancelled":
            self.store.update_mine(run_id, {"status": "CANCELLED", "ended_at": now_iso(),
                                            "current_step": result.steps})
            self.store.delete_message(receipt)
            return "cancelled"
        if result.status == "timeout":
            self.store.update_mine(run_id, {"status": "FAILED", "ended_at": now_iso(),
                                            "error": "max_runtime_exceeded", "current_step": result.steps})
            self.store.delete_message(receipt)
            return "timeout"
        self.store.update_mine(run_id, {"status": "REQUEUED", "current_step": result.steps,
                                        "last_heartbeat": now_iso()})
        hb.stop(); hb.join(10)
        self.store.extend_visibility(receipt, 0)
        self.store.register_worker("TERMINATING")
        self.exit_requested = True
        return "requeued"

# Starts the worker
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    Worker().run_forever()

if __name__ == "__main__":
    main()
