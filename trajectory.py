import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Define trajectory parameters
centre_x = 0.5
centre_y = 0.0
centre_z = 0.6
radius = 0.05
duration = 10
frequency = 50

def generate_circle(
        centre_x,
        centre_y,
        centre_z,
        radius,
        duration,
        frequency
):
    # Number of samples
    num_samples = int(duration * frequency)

    # Generate time values
    time = np.linspace(
        0, 
        duration,
        num_samples, 
        endpoint=False
    )

    # Angular velocity
    omega = (2 * np.pi) / duration

    x = centre_x + radius * np.cos(omega * time)
    y = centre_y + radius * np.sin(omega * time)
    z = np.full_like(time, centre_z)

    return time, x, y, z

# Generate circular trajectory
time, x, y, z = generate_circle(
    centre_x,
    centre_y,
    centre_z,
    radius,
    duration,
    frequency
)

# Test unit
distance_from_center = np.sqrt(
    (x - centre_x)**2 +
    (y - centre_y)**2
)

assert np.allclose(
    distance_from_center,
    radius,
    atol=1e-10
), "Trajectory radius verification failed!"

# Verify constant height
assert np.allclose(
    z, 
    centre_z
), "Trajectory height verification failed!"

# Verify number of samples
assert len(time) == 500

print("Trajectory mathematical verification passed!")

print("Circular trajectory generated!")
print(f"Number of samples: {len(time)}")
print(f"Duration: {duration} seconds")
print(f"Frequency: {frequency} Hz")

# Create results folder
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Save trajectory data as CSV
trajectory_data = pd.DataFrame({
    "time": time,
    "x": x,
    "y": y,
    "z": z
})

csv_path = results_dir / "circular_trajectory.csv" 
trajectory_data.to_csv(
    csv_path,
    index=False
)

print(f"Trajectory saved to: {csv_path}")

fig = plt.figure(figsize=(9,7))

ax = fig.add_subplot(
    111, 
    projection="3d"
)

ax.plot(
    x, 
    y, 
    z, 
    linewidth=2,
    label="Desired trajectory"
)

# Mark starting position
ax.scatter(
    x[0],
    y[0],
    z[0],
    s=60, 
    label="Start"
)

# Mark centre
ax.scatter(
    centre_x,
    centre_y, 
    centre_z, 
    marker="x",
    s=80,
    label="Centre"
)

ax.set_title(
    "Franka Panda: Desired Circular Trajectory"
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.legend()

# Equal scaling across axes
ax.set_box_aspect((1, 1, 1))

plt.tight_layout()


# ==========================================
# 7. Save figure
# ==========================================

plot_path = results_dir / "circular_trajectory.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Plot saved to: {plot_path}")

plt.show()