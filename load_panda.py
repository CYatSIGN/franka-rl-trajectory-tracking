import mujoco
from pathlib import Path

# Step 1: Locate the robot model
project_directory = Path.home() / "robotics-projects"

model_path = (
    project_directory
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

# Step 2: Load the robot model
model = mujoco.MjModel.from_xml_path(str(model_path))

# Step 3: Create simulation data
data = mujoco.MjData(model)

# Step 4: Move the robot to its home configuration
home_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

if home_id == -1:
    raise RuntimeError("Home configuration not found")

mujoco.mj_resetDataKeyframe(model, data, home_id)

# Step 5: Calculate robot positions
mujoco.mj_forward(model, data)

# Step 6: Display model information
print("Franka Panda successfully loaded!")
print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)

# Step 7: Display each arm joint
print("\nRobot arm joint positions:")

for i in range(1, 8):

    joint_name = f"joint{i}"

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise RuntimeError(f"Joint not found: {joint_name}")

    qpos_index = model.jnt_qposadr[joint_id]

    joint_position = data.qpos[qpos_index]

    print(f"{joint_name}: {joint_position:.4f} rad")

print("\nRobot model loaded successfully!")