import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from stable_baselines3 import PPO

from panda_rl_env import PandaTrackingEnv


# ==========================================
# 1. Load the trained PPO model
# ==========================================

model_path = Path("models/panda_ppo.zip")

if not model_path.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_ppo.py first."
    )

model = PPO.load(
    str(model_path),
    device="cpu"
)

print("Trained PPO model loaded!")


# ==========================================
# 2. Create evaluation environment
# ==========================================

env = PandaTrackingEnv()

obs, info = env.reset(seed=123)


# ==========================================
# 3. Evaluate trained policy
# ==========================================

time_history = []
desired_history = []
actual_history = []
error_history = []

total_reward = 0.0

terminated = False
truncated = False

while not (terminated or truncated):

    # The trained policy chooses the action
    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward

    # Observation layout from panda_rl_env.py
    actual_position = obs[14:17].copy()
    desired_position = obs[20:23].copy()

    time_history.append(info["simulation_time"])

    desired_history.append(desired_position)
    actual_history.append(actual_position)

    error_history.append(
        np.linalg.norm(
            desired_position - actual_position
        )
    )


# ==========================================
# 4. Calculate tracking metrics
# ==========================================

time_history = np.array(time_history)
desired_history = np.array(desired_history)
actual_history = np.array(actual_history)
error_history = np.array(error_history)

mean_error = np.mean(error_history)

rmse = np.sqrt(
    np.mean(error_history ** 2)
)

max_error = np.max(error_history)

p95_error = np.percentile(
    error_history,
    95
)

print("\nPPO TRACKING EVALUATION")
print("=" * 40)

print(f"Steps: {len(time_history)}")

print(f"Mean error: {mean_error * 1000:.3f} mm")
print(f"RMSE: {rmse * 1000:.3f} mm")
print(f"Maximum error: {max_error * 1000:.3f} mm")

print(
    f"95th percentile error: "
    f"{p95_error * 1000:.3f} mm"
)

print(f"Total reward: {total_reward:.3f}")


# ==========================================
# 5. Save evaluation results
# ==========================================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

df = pd.DataFrame({
    "time": time_history,

    "desired_x": desired_history[:, 0],
    "desired_y": desired_history[:, 1],
    "desired_z": desired_history[:, 2],

    "actual_x": actual_history[:, 0],
    "actual_y": actual_history[:, 1],
    "actual_z": actual_history[:, 2],

    "tracking_error": error_history
})

df.to_csv(
    results_dir / "ppo_tracking.csv",
    index=False
)


# ==========================================
# 6. Plot desired vs actual trajectory
# ==========================================

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    desired_history[:, 0],
    desired_history[:, 1],
    desired_history[:, 2],
    label="Desired trajectory",
    linewidth=2
)

ax.plot(
    actual_history[:, 0],
    actual_history[:, 1],
    actual_history[:, 2],
    label="PPO trajectory",
    linewidth=2
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title(
    "Franka Panda: PPO Circular Trajectory Tracking"
)

ax.legend()

ax.set_box_aspect((1, 1, 1))

plt.tight_layout()

plt.savefig(
    results_dir / "ppo_tracking_3d.png",
    dpi=300
)


# ==========================================
# 7. Plot tracking error over time
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    time_history,
    error_history * 1000
)

plt.xlabel("Time (s)")
plt.ylabel("Tracking error (mm)")

plt.title("PPO Tracking Error Over Time")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir / "ppo_tracking_error.png",
    dpi=300
)

plt.show()

env.close()

print("\nPPO evaluation complete!")