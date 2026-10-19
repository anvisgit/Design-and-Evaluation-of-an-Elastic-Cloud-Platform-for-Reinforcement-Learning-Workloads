from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from aws_clients import RunStore
from runstate import RunState

log = logging.getLogger("worker.callbacks")

# Turns a value into a float or None
def _f(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None

class MetricsCallback(BaseCallback):
    # Keeps the store, the run and the state
    def __init__(self, store: RunStore, run_id: str, algo: str, env_id: str, every: int, state: RunState):
        super().__init__()
        self.store, self.run_id, self.algo, self.env_id, self.every, self.state = store, run_id, algo, env_id, every, state
        self._last_step = 0
        self._t0 = time.time()
    # Resets the step and the timers
    def _on_training_start(self):
        self._last_step = self.model.num_timesteps
        self._t0 = time.time()
        self._t_last = self._t0
    # Saves metrics every few steps
    def _on_step(self) -> bool:
        self.state.current_step = self.num_timesteps
        if self.num_timesteps - self._last_step < self.every:
            return True
        now = time.time()
        fps = (self.num_timesteps - self._last_step) / max(now - self._t_last, 1e-6)
        self._last_step, self._t_last = self.num_timesteps, now
        nv = getattr(self.model.logger, "name_to_value", {})
        buf = list(getattr(self.model, "ep_info_buffer", []) or [])
        row = {
            "step": int(self.num_timesteps),
            "ep_rew_mean": _f(np.mean([e["r"] for e in buf])) if buf else None,
            "ep_len_mean": _f(np.mean([e["l"] for e in buf])) if buf else None,
            "loss": _f(nv.get("train/loss")),
            "value_loss": _f(nv.get("train/value_loss")),
            "policy_loss": _f(nv.get("train/policy_gradient_loss", nv.get("train/actor_loss"))),
            "entropy": _f(nv.get("train/entropy_loss", nv.get("train/ent_coef"))),
            "learning_rate": _f(nv.get("train/learning_rate")),
            "fps": _f(fps),
            "wall_time": _f(now - self._t0),
            "expires_at": int(now) + 180 * 86400,
        }
        self.store.put_metrics(self.run_id, [row])
        cw = {"StepsPerSecond": fps}
        if row["ep_rew_mean"] is not None:
            cw["EpisodeRewardMean"] = row["ep_rew_mean"]
        self.store.put_cw(self.algo, self.env_id, cw)
        return True

class CheckpointCallback(BaseCallback):
    # Keeps the store and makes the folder
    def __init__(self, store: RunStore, run_id: str, every: int, directory: Path, keep: int = 3):
        super().__init__()
        self.store, self.run_id, self.every, self.dir, self.keep = store, run_id, every, Path(directory), keep
        self.dir.mkdir(parents=True, exist_ok=True)
        self._saved: list[Path] = []
        self._last = 0
    # Resets the step counter
    def _on_training_start(self):
        self._last = self.model.num_timesteps
    # Saves and uploads a checkpoint
    def save_now(self) -> str:
        step = int(self.model.num_timesteps)
        path = self.dir / f"step_{step}.zip"
        self.model.save(str(path))
        uri = self.store.upload(path, f"checkpoints/{self.run_id}/step_{step}.zip")
        self.store.update(self.run_id, {"latest_checkpoint_s3_uri": uri, "current_step": step})
        self._saved.append(path)
        while len(self._saved) > self.keep:
            self._saved.pop(0).unlink(missing_ok=True)
        log.info("checkpoint saved step=%s", step)
        return uri
    # Saves a checkpoint every few steps
    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self.save_now()
        return True

class EvalCallback(BaseCallback):
    # Keeps the store and the eval env
    def __init__(self, store: RunStore, run_id: str, eval_env, every: int, episodes: int, directory: Path):
        super().__init__()
        self.store, self.run_id, self.env, self.every, self.episodes = store, run_id, eval_env, every, episodes
        self.dir = Path(directory)
        self.best = -np.inf
        self._last = 0
    # Resets the step counter
    def _on_training_start(self):
        self._last = self.model.num_timesteps
    # Tests the model and keeps the best one
    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        mean, std = evaluate_policy(self.model, self.env, n_eval_episodes=self.episodes)
        log.info("eval step=%s mean=%.2f std=%.2f", self.num_timesteps, mean, std)
        if mean > self.best:
            self.best = float(mean)
            path = self.dir / "best_model.zip"
            self.model.save(str(path))
            self.store.upload(path, f"models/{self.run_id}/best_model.zip")
            self.store.update(self.run_id, {"best_reward": float(mean)})
        return True

class ControlCallback(BaseCallback):
    # Keeps the limits and the stop signals
    def __init__(self, store: RunStore, run_id: str, max_runtime_s: float, interrupt: threading.Event,
                 state: RunState, check_every: int = 1000):
        super().__init__()
        self.store, self.run_id, self.max_s, self.interrupt = store, run_id, max_runtime_s, interrupt
        self.state, self.check_every = state, check_every
        self._t0 = time.monotonic()
        self._last = 0
    # Resets the timer and the step counter
    def _on_training_start(self):
        self._t0 = time.monotonic()
        self._last = self.model.num_timesteps
    # Stops training on interrupt, timeout or cancel
    def _on_step(self) -> bool:
        if self.interrupt.is_set():
            self.state.stop_reason = "interrupted"
            return False
        if self.num_timesteps - self._last < self.check_every:
            return True
        self._last = self.num_timesteps
        if time.monotonic() - self._t0 > self.max_s:
            self.state.stop_reason = "timeout"
            return False
        if self.store.is_cancel_requested(self.run_id):
            self.state.stop_reason = "cancelled"
            return False
        return True
