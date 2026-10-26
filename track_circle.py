import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Simulation settings
DURATION = 10.0
CONTROL_FREQUENCY = 50
CONTROL_DT = 1.0 / CONTROL_FREQUENCY

RADIUS = 0.04

# One complete revolution in 10 seconds
OMEGA = 2 * np.pi / DURATION

KP = 3.0
DAMPING = 0.05

MAX_CART_SPEED = 0.12
MAX_JOINT_SPEED = 0.8

# Load Franka Panda

model_path = (
    Path.home()
    /"robotics-projects"
    /"mujoco_menagerie"
    /"franka_emika_panda"
    /"scene.xml"
)

if not model_path.exists():
    raise FileNotFoundError(model_path)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

if home_id == -1:
    raise RuntimeError("Home keyframe not found")

mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

# Identify robot joints and hand

hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

if hand_id == -1:
    raise RuntimeError("Hand body not found")

def get_id(object_type, name):
    object_id = mujoco.mj_name2id(
        model, object_type, name
    )

    if object_id == -1:
        raise RuntimeError(f"Missing model element: {name}")

    return object_id

joint_ids = np.array([
    get_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
    for i in range(1, 8)
])

actuator_ids = np.array([
    get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
    for i in range(1, 8)
])

qpos_ids = model.jnt_qposadr[joint_ids]
dof_ids = model.jnt_dofadr[joint_ids]

joint_limits = model.jnt_range[joint_ids]

# Initial joint position commands
q_command = data.qpos[qpos_ids].copy()

data.ctrl[actuator_ids] = q_command

# Generate a circular trajectory

initial_pos = data.xpos[hand_id].copy()
center = initial_pos.copy()
center[0] -= RADIUS

def desired_trajectory(t):

    theta = OMEGA * t

    desired_pos = center + np.array([
        RADIUS * np.cos(theta),
        RADIUS * np.sin(theta),
        0.0
    ])

    desired_velocity = np.array([
        -RADIUS * OMEGA * np.sin(theta),
        RADIUS * OMEGA * np.cos(theta),
        0.0
    ])

    return desired_pos, desired_velocity

# Calculate Jacobian-based joint commands

def calculate_joint_velocity(
        desired_pos,
        desired_velocity
):
    current_pos = data.xpos[hand_id].copy()

    error = desired_pos - current_pos

    # Feedforward + feedback controller
    cart_velocity = desired_velocity + KP * error
    speed = np.linalg.norm(cart_velocity)

    if speed > MAX_CART_SPEED:
        cart_velocity *= MAX_CART_SPEED / speed

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    mujoco.mj_jacBody(
        model,
        data,
        jac_pos,
        jac_rot,
        hand_id
    )

    J = jac_pos[:, dof_ids]

    # Dampled least-squares inverse kinematics
    A = J @ J.T + DAMPING**2 * np.eye(3)

    q_velocity = J.T @ np.linalg.solve(
        A,
        cart_velocity
    )

    q_velocity = np.clip(
        q_velocity,
        -MAX_JOINT_SPEED,
        MAX_JOINT_SPEED   
    )

    return q_velocity

# Prepare simulation

physics_dt = model.opt.timestep

steps_per_control = round(
    CONTROL_DT / physics_dt
)

if steps_per_control < 1 or not np.isclose(
    steps_per_control * physics_dt,
    CONTROL_DT
):
    raise ValueError(
        "Control interval must be an integer multiple "
        "of the MuJoCo physics timestep."
    )

num_steps = int(DURATION * CONTROL_FREQUENCY)

time_history = []
desired_history = []
actual_history = []
error_history = []

# Run circular tracking

print("Starting circular trajectory tracking...")
print("Initial hand position:", initial_pos)
print("Circle centre:", center)

for step in range(num_steps):
    t = step * CONTROL_DT

    # Get target position and velocity
    desired_pos, desired_velocity = (
        desired_trajectory(t)
    )

    # Calculate IK joint velocity
    q_velocity = calculate_joint_velocity(
        desired_pos,
        desired_velocity
    )

    # Update joint positioni commands
    q_command = q_command + q_velocity * CONTROL_DT

    # Respect joint limiits
    q_command = np.clip(
        q_command,
        joint_limits[:, 0],
        joint_limits[:, 1]
    )

    # Send commands to MuJoCo
    data.ctrl[actuator_ids] = q_command

    # Advance physics 
    for _ in range(steps_per_control):
        mujoco.mj_step(model, data)

    # Record actual position
    actual_pos = data.xpos[hand_id].copy()

    # Compare at the same simulation timestamp
    desired_at_actual_time, _ = desired_trajectory(
        data.time
    )

    error = np.linalg.norm(
        desired_at_actual_time - actual_pos
    )

    time_history.append(data.time)
    desired_history.append(desired_at_actual_time.copy())
    actual_history.append(actual_pos.copy())
    error_history.append(error)

    if step % 50 == 0:
        print(
            f"Time: {data.time:.2f}s | "
            f"Tracking error: {error:.5f} m"
        )

# Evaluate tracking accuracy
time_history = np.array(time_history)
desired_history = np.array(desired_history)
actual_history = np.array(actual_history)
error_history = np.array(error_history)

mean_error = np.mean(error_history)

rmse = np.sqrt(np.mean(error_history**2))

max_error = np.max(error_history)

print("\nSIMULATION COMPLETE")

print(f"Mean tracking error: {mean_error:.6f} m")
print(f"RMSE: {rmse:.6f} m")
print(f"Maximum error: {max_error:.6f} m")


# ==========================================
# 9. Save results to CSV
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
    results_dir / "circle_tracking.csv",
    index=False
)


# ==========================================
# 10. Plot desired vs actual trajectory
# ==========================================

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(111, projection="3d")

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
    label="Actual trajectory",
    linewidth=2
)

ax.scatter(
    *initial_pos,
    label="Initial position",
    s=40
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title("Franka Panda: Circular Trajectory Tracking")

ax.set_xlim(center[0] - 0.07, center[0] + 0.07)
ax.set_ylim(center[1] - 0.07, center[1] + 0.07)
ax.set_zlim(center[2] - 0.07, center[2] + 0.07)

ax.set_box_aspect((1, 1, 1))

ax.legend()

plt.tight_layout()

plt.savefig(
    results_dir / "circle_tracking_3d.png",
    dpi=300
)


# ==========================================
# 11. Plot tracking error over time
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    time_history,
    error_history * 1000
)

plt.xlabel("Time (s)")
plt.ylabel("Position error (mm)")

plt.title("End-Effector Tracking Error Over Time")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir / "circle_tracking_error.png",
    dpi=300
)

plt.show()

print("\nResults saved successfully!")
