import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from panda_rl_env import PandaTrackingEnv


# Training configuration

parser = argparse.ArgumentParser()

parser.add_argument(
    "--steps",
    type=int,
    default=20_000,
    help="Number of training timesteps"
)

args = parser.parse_args()

TOTAL_TIMESTPES = args.steps

SEED = 42

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Create the MuJoCo environment 

env = PandaTrackingEnv()

env = Monitor(
    env,
    filename=str(LOG_DIR / "training_monitor.csv")
)

env.reset(seed=SEED)

print("Panda reinforcement learning environmnt ready!")

# Create the PPO agent

model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=3e-4,

    n_steps=1024,
    batch_size=64,
    n_epochs=10,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,
    ent_coef=0.0,

    policy_kwargs=dict(
        net_arch=[128, 128]
    ),

    tensorboard_log=str(LOG_DIR / "tensorboard"),

    verbose=1, 
    seed=SEED,
    device="cpu"
)

# Save intermediate checkpoints

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,

    save_path=str(CHECKPOINT_DIR),

    name_prefix="panda_ppo"

)

# Train PPO

print("\nSTARTING PPO TRAINING")
print(f"Requested timesteps: {TOTAL_TIMESTPES}")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTPES,
        callback=checkpoint_callback,
        progress_bar=True
    )

finally:
    # Close the training environment
    env.close()

# Save the training model

model_path = MODEL_DIR / "panda_ppo"

model.save(str(model_path))

print("\nTRAINING COMPLETE")
print(f"Model saved to: {model_path}.zip")