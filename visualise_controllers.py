import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from stable_baselines3 import PPO
from panda_rl_env import PandaTrackingEnv


# ==========================================
# 1. Command-line options
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--controller",
    choices=["ik", "ppo"],
    required=True
)

parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed: 1.0 = real time, 0.5 = half speed"
)

args = parser.parse_args()

if args.speed <= 0:
    parser.error("--speed must be greater than zero")


# ==========================================
# 2. Create environment
# ==========================================

env = PandaTrackingEnv()

obs, _ = env.reset(seed=123)

model = env.model
data = env.data

KP = 3.0
DAMPING = 0.05

MAX_CART_SPEED = 0.12
MAX_JOINT_SPEED = 0.8


# ==========================================
# 3. Load PPO if selected
# ==========================================

policy = None

if args.controller == "ppo":

    policy = PPO.load(
        "models/panda_ppo.zip",
        device="cpu"
    )

    if policy.observation_space.shape != (
        env.observation_space.shape
    ):
        raise ValueError(
            "PPO model and environment observation "
            "dimensions do not match."
        )


# ==========================================
# 4. Classical IK controller
# ==========================================

q_command = data.qpos[env.qpos_ids].copy()


def classical_ik_step():

    global q_command

    target, target_velocity = (
        env.desired_trajectory(data.time)
    )

    actual_position = data.xpos[
        env.hand_id
    ].copy()

    position_error = target - actual_position

    cart_velocity = (
        target_velocity + KP * position_error
    )

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
        env.hand_id
    )

    J = jac_pos[:, env.dof_ids]

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

    q_command += q_velocity * env.control_dt

    q_command = np.clip(
        q_command,
        env.joint_limits[:, 0],
        env.joint_limits[:, 1]
    )

    data.ctrl[env.actuator_ids] = q_command

    for _ in range(env.physics_steps):
        mujoco.mj_step(model, data)


# ==========================================
# 5. Add target and hand markers
# ==========================================

def update_markers(viewer):

    target, _ = env.desired_trajectory(data.time)

    actual = data.xpos[env.hand_id].copy()

    # Blue sphere = target position
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.008, 0, 0],
        pos=target,
        mat=np.eye(3).flatten(),
        rgba=[0.1, 0.4, 1.0, 0.9]
    )

    # Orange sphere = actual hand position
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.006, 0, 0],
        pos=actual,
        mat=np.eye(3).flatten(),
        rgba=[1.0, 0.45, 0.0, 0.9]
    )

    viewer.user_scn.ngeom = 2


# ==========================================
# 6. Launch viewer
# ==========================================

print(f"Controller: {args.controller.upper()}")
print("Blue sphere: desired target")
print("Orange sphere: actual hand")
print("Simulation duration: 10 seconds")
print("Close the viewer window to stop early.")

errors = []

with mujoco.viewer.launch_passive(
    model, data
) as viewer:

    update_markers(viewer)
    viewer.sync()

    for step in range(env.max_steps):

        if not viewer.is_running():
            break

        start = time.perf_counter()

        if args.controller == "ppo":

            action, _ = policy.predict(
                obs,
                deterministic=True
            )

            obs, _, terminated, truncated, info = (
                env.step(action)
            )

        else:

            classical_ik_step()

        target, _ = env.desired_trajectory(data.time)

        actual = data.xpos[env.hand_id].copy()

        error = np.linalg.norm(target - actual)

        errors.append(error)

        update_markers(viewer)
        viewer.sync()

        if step % 50 == 0:
            print(
                f"Time: {data.time:.2f}s | "
                f"Tracking error: {error * 1000:.3f} mm",
                flush=True
            )

        if args.controller == "ppo" and (
            terminated or truncated
        ):
            break

        # Slow simulation to a watchable speed
        elapsed = time.perf_counter() - start
        delay = env.control_dt / args.speed - elapsed

        if delay > 0:
            time.sleep(delay)

    print("\nSimulation finished.", flush=True)

    if errors:
        print(
            f"Mean tracking error: "
            f"{np.mean(errors) * 1000:.3f} mm",
            flush=True
        )

    # Give the user a moment to inspect the final pose
    if viewer.is_running():
        time.sleep(1.0)

env.close()

print("Program finished.", flush=True)