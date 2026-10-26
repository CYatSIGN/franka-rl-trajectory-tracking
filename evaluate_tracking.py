import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. Load simulation results
# ==========================================

results_dir = Path("results")

csv_path = results_dir / "circle_tracking.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        "Run track_circle.py before evaluating."
    )

df = pd.read_csv(csv_path)

time = df["time"].to_numpy()

desired = df[
    ["desired_x", "desired_y", "desired_z"]
].to_numpy()

actual = df[
    ["actual_x", "actual_y", "actual_z"]
].to_numpy()


# ==========================================
# 2. Validate the data
# ==========================================

if len(time) < 3:
    raise ValueError("At least three samples are required.")

if not np.all(np.diff(time) > 0):
    raise ValueError("Time values must be increasing.")

if not np.all(np.isfinite(actual)):
    raise ValueError("Actual positions contain invalid values.")

if not np.all(np.isfinite(desired)):
    raise ValueError("Desired positions contain invalid values.")


# ==========================================
# 3. Calculate tracking errors
# ==========================================

position_error = desired - actual

error_magnitude = np.linalg.norm(
    position_error,
    axis=1
)

mean_error = np.mean(error_magnitude)

rmse = np.sqrt(
    np.mean(error_magnitude ** 2)
)

max_error = np.max(error_magnitude)

p95_error = np.percentile(
    error_magnitude,
    95
)


# ==========================================
# 4. Calculate end-effector velocities
# ==========================================

# Numerical derivative of actual position
velocity = np.gradient(
    actual,
    time,
    axis=0
)

speed = np.linalg.norm(
    velocity,
    axis=1
)


# ==========================================
# 5. Calculate end-effector acceleration
# ==========================================

acceleration = np.gradient(
    velocity,
    time,
    axis=0
)

acceleration_magnitude = np.linalg.norm(
    acceleration,
    axis=1
)


# ==========================================
# 6. Print evaluation results
# ==========================================

print("\nFRANKA PANDA TRACKING EVALUATION")
print("=" * 45)

print(f"Mean error: {mean_error * 1000:.3f} mm")
print(f"RMSE: {rmse * 1000:.3f} mm")

print(f"Maximum error: {max_error * 1000:.3f} mm")

print(f"95th percentile error: {p95_error * 1000:.3f} mm")

print(f"Mean hand speed: {np.mean(speed):.4f} m/s")

print(
    f"Mean hand acceleration: "
    f"{np.mean(acceleration_magnitude):.4f} m/s²"
)


# ==========================================
# 7. Plot desired vs actual coordinates
# ==========================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(10, 9),
    sharex=True
)

coordinate_names = ["X", "Y", "Z"]

for i, ax in enumerate(axes):

    ax.plot(
        time,
        desired[:, i],
        label="Desired",
        linewidth=2
    )

    ax.plot(
        time,
        actual[:, i],
        label="Actual",
        linestyle="--"
    )

    ax.set_ylabel(f"{coordinate_names[i]} (m)")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time (s)")

fig.suptitle(
    "Franka Panda: Desired vs Actual Coordinates"
)

plt.tight_layout()

plt.savefig(
    results_dir / "coordinate_tracking.png",
    dpi=300
)


# ==========================================
# 8. Plot end-effector speed
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(time, speed)

plt.xlabel("Time (s)")
plt.ylabel("Hand speed (m/s)")

plt.title("End-Effector Speed Over Time")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_dir / "hand_speed.png",
    dpi=300
)


# ==========================================
# 9. Plot end-effector acceleration
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    acceleration_magnitude
)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")

plt.title("End-Effector Acceleration Over Time")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_dir / "hand_acceleration.png",
    dpi=300
)

plt.show()

print("\nEvaluation complete!")