import argparse
import time
from pathlib import Path
import mujoco
import numpy as np

# Load Franka Panda
model_path = (
    Path.home()
    /"robotics-projects"
    /"mujoco_menagerie"
    /"franka_emika_panda"
    /"scene.xml"
)

if not model_path.exists():
    raise FileNotFoundError(
        f"Robot model not found: {model_path}"
    )

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Reset robot to its home configuration
home_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

if home_id == -1:
    raise RuntimeError("Home keyframe not found")

mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

# Identify joints and robot hand
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

if hand_id == -1:
    raise RuntimeError("Robot hand not found")

joint_ids = [
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"joint{i}"
    )
    for i in range(1, 8)
]

if any(joint_id == -1 for joint_id in joint_ids):
    raise RuntimeError("One or more arm joints were not found")

qpos_ids = np.array([
    model.jnt_qposadr[joint_id]
    for joint_id in joint_ids
])

dof_ids = np.array([
    model.jnt_dofadr[joint_id]
    for joint_id in joint_ids
])

joint_limits = model.jnt_range[joint_ids]

# The official Panda model uses actuator1 to actuator7
# for its 7 arm joints
actuator_ids = np.array([
    mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"actuator{i}"
    )
    for i in range(1, 8)
])

if np.any(actuator_ids == -1):
    raise RuntimeError("Arm actuators not found")

# Define starting and target positions
intial_pos = data.xpos[hand_id].copy()

target_pos = intial_pos + np.array([
    0.00,
    0.03,
    0.60
])

print("Initial hand position:", intial_pos)
print("Target hand position:", target_pos)

# Controller settings
CONTROL_DT = 0.02 # 50Hz
SIM_DURATION = 5.0 # seconds

KP = 2.0 # Position correction gain
DAMPING = 0.05 # Jacobian damping

MAX_CART_SPEED = 0.10 # metres per second
MAX_JOINT_SPEED = 0.50 # radians per second

# Store the commanded joint positions
q_command = data.qpos[qpos_ids].copy()

# Track the robot's previous state
previous_pos = intial_pos.copy()

# Inverse kinematics control step
def control_step():

    global q_command

    # Current hand position
    current_pos = data.xpos[hand_id].copy()

    # Cartesian position error
    error = target_pos - current_pos

    # Desired hand velocity
    desired_velocity = KP * error

    # Limit Cartesian speed
    speed = np.linalg.norm(desired_velocity)

    if speed > MAX_CART_SPEED:
        desired_velocity *= MAX_CART_SPEED / speed

    # Calculate the 6 x nv Jacobian
    jacobian_pos = np.zeros((3, model.nv))
    jacobian_rot = np.zeros((3, model.nv))

    mujoco.mj_jacBody(
        model, 
        data, 
        jacobian_pos,
        jacobian_rot,
        hand_id
    )

    # Extract the seven arm joint columns
    J = jacobian_pos[:, dof_ids]

    # Damped least-squares inverse kinematics
    A = J @ J.T + (DAMPING ** 2) * np.eye(3)

    joint_velocity = (
        J.T @ np.linalg.solve(A, desired_velocity)
    )

    # Limit joint velocities 
    joint_velocity = np.clip(
        joint_velocity, 
        -MAX_JOINT_SPEED,
        MAX_JOINT_SPEED
    )

    # Increment commanded joint posisitons
    q_command += joint_velocity * CONTROL_DT

    # Respect joint limits
    q_command = np.clip(
        q_command,
        joint_limits[:, 0],
        joint_limits[:, 1]
    )

    # Send joint positions targets to actuators
    data.ctrl[actuator_ids] = q_command

    return np.linalg.norm(error)

# Run physics simulation
def run_sim(viewer=None):

    number_of_steps = int(SIM_DURATION / CONTROL_DT)
    physics_steps = max(
        1, 
        round(CONTROL_DT / model.opt.timestep)
    )

    next_control_time = time.perf_counter()

    for step in range(number_of_steps):
        error = control_step()

        # Advance MuJoCo physics
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)

        if step % 25 == 0:
            print(
                f"Time: {data.time:.2f} s |"
                f"Delay: {error:.2f} m"
            )

        if viewer is not None:
            viewer.sync()

            # Slow simulation to approximately real time 
            next_control_time += (
                physics_steps * model.opt.timestep
            )

            sleep_time = (
                next_control_time - time.perf_counter()
            )

            if sleep_time > 0:
                time.sleep(sleep_time)

            if not viewer.is_running():
                break

        # Final evaluation
        mujoco.mj_forward(model, data)

        final_pos = data.xpos[hand_id].copy()

        final_error = np.linalg.norm(
            target_pos - final_pos
        )

        print("\nSIMULATION COMPLETE")
        print("Initial position:", intial_pos)
        print("Final position:", final_pos)

        print(f"Final tracking error: {final_error:.6f} m")

# Choose whether to open the viewer
parser = argparse.ArgumentParser()

parser.add_argument(
    "--viewer",
    action="store_true",
    help="Display the MuJoCo simuation"
)

args = parser.parse_args()

if args.viewer:

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Synchronise the initial robot configuration
        viewer.sync()

        # Run the controller
        run_sim(viewer)

        # Synchronise the final robot configuration
        if viewer.is_running():
            viewer.sync()

else:

    run_sim()