from __future__ import annotations
import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from aws_clients import RunStore
from callbacks import CheckpointCallback, ControlCallback, EvalCallback, MetricsCallback, RunState
from config_schema import TrainingConfig

log = logging.getLogger("worker.trainer")
ALGOS = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "SAC": SAC}

@dataclass
class TrainResult:
    status: str
    steps: int
    final_reward: Optional[float] = None
    best_reward: Optional[float] = None
    model_uri: Optional[str] = None

# Sets every random seed
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

# Trains a model and returns the result
def train(cfg: TrainingConfig, run_id: str, store: RunStore, state: RunState,
          interrupt: threading.Event, workdir: Path, resume_from: Optional[Path] = None) -> TrainResult:
    seed_everything(cfg.seed)
    env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)
    eval_env = make_vec_env(cfg.env_id, n_envs=1, seed=cfg.seed + 10_000)
    cls = ALGOS[cfg.algo]
    policy = cfg.hyperparams.get("policy", "MlpPolicy")
    if resume_from is not None:
        model = cls.load(str(resume_from), env=env, device="cpu")
        log.info("resumed from %s at step %s", resume_from, model.num_timesteps)
    else:
        model = cls(policy, env, seed=cfg.seed, verbose=1, device="cpu", **cfg.sb3_kwargs())
    lg = cfg.logging
    ckpt = CheckpointCallback(store, run_id, lg.checkpoint_every_steps, workdir / "checkpoints")
    evalcb = EvalCallback(store, run_id, eval_env, lg.eval_every_steps, lg.eval_episodes, workdir)
    callbacks = CallbackList([
        MetricsCallback(store, run_id, cfg.algo, cfg.env_id, lg.metrics_every_steps, state),
        ckpt, evalcb,
        ControlCallback(store, run_id, cfg.resources.max_runtime_minutes * 60, interrupt, state),
    ])
    remaining = cfg.total_timesteps - model.num_timesteps
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=resume_from is None,
                    log_interval=10)
    steps = int(model.num_timesteps)
    state.current_step = steps
    if state.stop_reason:
        ckpt.save_now()
        return TrainResult(state.stop_reason, steps, best_reward=evalcb.best if evalcb.best > -np.inf else None)
    final_path = workdir / "final_model.zip"
    model.save(str(final_path))
    uri = store.upload(final_path, f"models/{run_id}/final_model.zip")
    mean, _ = evaluate_policy(model, eval_env, n_eval_episodes=lg.eval_episodes)
    best = evalcb.best if evalcb.best > -np.inf else float(mean)
    return TrainResult("finished", steps, float(mean), max(best, float(mean)), uri)
