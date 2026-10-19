from __future__ import annotations
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALGOS = ("PPO", "A2C", "DQN", "SAC")
ENV_ALLOWLIST: dict[str, str] = {
    "CartPole-v1": "discrete",
    "LunarLander-v3": "discrete",
    "Acrobot-v1": "discrete",
    "MountainCar-v0": "discrete",
    "Pendulum-v1": "continuous",
    "BipedalWalker-v3": "continuous",
}
MAX_TOTAL_TIMESTEPS = 5_000_000
MAX_RUNTIME_MINUTES = 240
_COMMON_OFFPOLICY = {
    "learning_rate", "gamma", "buffer_size", "learning_starts", "batch_size",
    "tau", "train_freq", "gradient_steps", "policy",
}
ALLOWED_HYPERPARAMS: dict[str, set[str]] = {
    "PPO": {"learning_rate", "gamma", "n_steps", "batch_size", "n_epochs", "gae_lambda",
            "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "policy"},
    "A2C": {"learning_rate", "gamma", "n_steps", "gae_lambda", "ent_coef", "vf_coef",
            "max_grad_norm", "rms_prop_eps", "use_rms_prop", "normalize_advantage", "policy"},
    "DQN": _COMMON_OFFPOLICY | {"target_update_interval", "exploration_fraction",
                                "exploration_initial_eps", "exploration_final_eps", "max_grad_norm"},
    "SAC": _COMMON_OFFPOLICY | {"ent_coef"},
}
HPValue = Union[float, int, bool, str]

class LoggingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics_every_steps: int = Field(2000, ge=100, le=1_000_000)
    checkpoint_every_steps: int = Field(25_000, ge=1000, le=5_000_000)
    eval_every_steps: int = Field(25_000, ge=1000, le=5_000_000)
    eval_episodes: int = Field(10, ge=1, le=100)

class ResourcesCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_class: Literal["cpu"] = "cpu"
    max_runtime_minutes: int = Field(120, ge=1, le=MAX_RUNTIME_MINUTES)

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=100)
    algo: Literal["PPO", "A2C", "DQN", "SAC"]
    env_id: str
    seed: int = Field(0, ge=0, le=2**31 - 1)
    total_timesteps: int = Field(gt=0, le=MAX_TOTAL_TIMESTEPS)
    n_envs: int = Field(1, ge=1, le=16)
    hyperparams: dict[str, HPValue] = Field(default_factory=dict)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    resources: ResourcesCfg = Field(default_factory=ResourcesCfg)
    # Checks that the env is on the allow list
    @field_validator("env_id")
    @classmethod
    def _env_allowed(cls, v: str) -> str:
        if v not in ENV_ALLOWLIST:
            raise ValueError(f"env_id '{v}' not allowed; choose one of {sorted(ENV_ALLOWLIST)}")
        return v
    # Checks the hyperparameters and the algo and env match
    @model_validator(mode="after")
    def _check_algo_rules(self) -> "TrainingConfig":
        allowed = ALLOWED_HYPERPARAMS[self.algo]
        unknown = set(self.hyperparams) - allowed
        if unknown:
            raise ValueError(f"unknown hyperparams for {self.algo}: {sorted(unknown)}")
        space = ENV_ALLOWLIST[self.env_id]
        if self.algo == "DQN" and space != "discrete":
            raise ValueError("DQN requires a discrete action space")
        if self.algo == "SAC" and space != "continuous":
            raise ValueError("SAC requires a continuous action space")
        for k, v in self.hyperparams.items():
            if k == "policy":
                if v != "MlpPolicy":
                    raise ValueError("only MlpPolicy is supported")
            elif k == "ent_coef" and isinstance(v, str):
                if not v.startswith("auto") or self.algo != "SAC":
                    raise ValueError("string ent_coef is only valid as 'auto' for SAC")
            elif isinstance(v, str):
                raise ValueError(f"hyperparam {k} must be numeric")
        return self
    # Returns the hyperparameters for the training library
    def sb3_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in self.hyperparams.items() if k != "policy"}

class ExperimentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    description: str = Field("", max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    # Checks the length of each tag
    @field_validator("tags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        if any(not t or len(t) > 50 for t in v):
            raise ValueError("tags must be 1-50 characters")
        return v

class CompareIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: list[str] = Field(min_length=1, max_length=10)
    metric: str = "ep_rew_mean"
    threshold: Optional[float] = None
    # Checks that the metric is allowed
    @field_validator("metric")
    @classmethod
    def _metric(cls, v: str) -> str:
        allowed = {"ep_rew_mean", "ep_len_mean", "loss", "value_loss", "policy_loss",
                   "entropy", "learning_rate", "fps"}
        if v not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}")
        return v
