import mujoco
from pathlib import Path
import numpy as np

# Locate the Franka Panda robot model
model_path = (
    Path.home()
    /"robotics-projects"
    /"mujoco_menagerie"
    /"franka_emika_panda"
    /"scene.xml"
)

# Load the model 
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Initialise the home configuration
home_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_KEY, 
    "home"
)

if home_id == -1:
    raise RuntimeError("Home keyframe not found")

mujoco.mj_resetDataKeyframe(model, data, home_id)

# Calculate the robot's kinematics
mujoco.mj_forward(model, data)

# Find the robot's hand
hand_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_BODY, 
    "hand"
)

if hand_id == -1:
    raise RuntimeError("Robot hand not found")

# Get the hand's 3D position
initial_pos = data.xpos[hand_id].copy()

print("INITIAL HAND POSITION")
print(f"X: {initial_pos[0]:.4f} m")
print(f"X: {initial_pos[1]:.4f} m")
print(f"X: {initial_pos[2]:.4f} m")

# Change the first joint angle
# Initialise joint configuration
joint_id = mujoco.mj_name2id(
    model, 
    mujoco.mjtObj.mjOBJ_JOINT, 
    "joint1"
)

qpos_index = model.jnt_qposadr[joint_id]

data.qpos[qpos_index] = np.deg2rad(30)

# Recalculate forward kinematics
mujoco.mj_forward(model, data)

# Get the new hand position
new_hand_pos = data.xpos[hand_id].copy()

print("NEW HAND POSITION")
print(f"X: {new_hand_pos[0]:.4f} m")
print(f"X: {new_hand_pos[1]:.4f} m")
print(f"X: {new_hand_pos[2]:.4f} m")

# Calculate how far the hand moved
distance = np.linalg.norm(new_hand_pos - initial_pos)

print(f"\nDistance between positions: {distance:.4f} m")