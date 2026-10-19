import pytest
from pydantic import ValidationError
from common.config_schema import TrainingConfig

BASE = {"experiment_id": "e1", "name": "n", "algo": "PPO", "env_id": "CartPole-v1",
        "seed": 1, "total_timesteps": 10_000, "n_envs": 2,
        "hyperparams": {"learning_rate": 3e-4, "n_steps": 128, "policy": "MlpPolicy"}}

# Builds a config with some changes
def cfg(**over):
    return TrainingConfig.model_validate({**BASE, **over})

# Checks the default values
def test_valid_defaults():
    c = cfg()
    assert c.logging.metrics_every_steps == 2000 and c.resources.max_runtime_minutes == 120
    assert "policy" not in c.sb3_kwargs()

# Checks that bad configs are rejected
@pytest.mark.parametrize("over", [
    {"algo": "TD3"},
    {"env_id": "Evil-v0"},
    {"total_timesteps": 5_000_001},
    {"total_timesteps": 0},
    {"hyperparams": {"hack": 1}},
    {"hyperparams": {"policy": "CnnPolicy"}},
    {"algo": "DQN", "env_id": "Pendulum-v1", "hyperparams": {}},
    {"algo": "SAC", "env_id": "CartPole-v1", "hyperparams": {}},
    {"resources": {"max_runtime_minutes": 241}},
    {"unexpected": 1},
    {"hyperparams": {"learning_rate": "fast"}},
])
def test_rejected(over):
    with pytest.raises(ValidationError):
        cfg(**over)

# Checks that auto entropy works for SAC
def test_sac_auto_entropy_ok():
    cfg(algo="SAC", env_id="Pendulum-v1", hyperparams={"ent_coef": "auto"})
