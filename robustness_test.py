import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco

from pathlib import Path
from stable_baselines3 import PPO

from panda_rl_env import PandaTrackingEnv


# ==========================================
# 1. Experiment settings
# ==========================================

NOISE_LEVELS_MM = [0, 1, 3]
SEEDS = [42, 123, 456]

KP = 3.0
DAMPING = 0.05

MAX_CART_SPEED = 0.12
MAX_JOINT_SPEED = 0.8

MODEL_PATH = Path("models/panda_ppo.zip")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. Load original PPO model
# ==========================================

model = PPO.load(
    str(MODEL_PATH),
    device="cpu"
)

if model.observation_space.shape != (36,):
    raise ValueError(
        "Expected the original 36-observation PPO model."
    )

print("Original PPO model loaded.")


# ==========================================
# 3. Add noise to PPO observations
# ==========================================

def add_observation_noise(obs, noise_std, rng):

    noisy_obs = obs.copy()

    # Hand position is stored at indices 14:17
    noisy_obs[14:17] += rng.normal(
        0,
        noise_std,
        size=3
    )

    # Recalculate tracking error using noisy position
    noisy_obs[26:29] = (
        noisy_obs[20:23] - noisy_obs[14:17]
    )

    return noisy_obs


# ==========================================
# 4. Evaluate PPO
# ==========================================

def evaluate_ppo(noise_std, seed):

    env = PandaTrackingEnv()

    if env.observation_space.shape != (36,):
        env.close()
        raise ValueError(
            "panda_rl_env.py must use 36 observations."
        )

    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=seed)

    errors = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        noisy_obs = add_observation_noise(
            obs,
            noise_std,
            rng
        )

        action, _ = model.predict(
            noisy_obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        # Record actual error, not noisy error
        errors.append(info["tracking_error"])

    env.close()

    return np.array(errors)


# ==========================================
# 5. Evaluate classical IK controller
# ==========================================

def evaluate_ik(noise_std, seed):

    env = PandaTrackingEnv()

    rng = np.random.default_rng(seed)

    data = env.data
    mj_model = env.model

    env.reset(seed=seed)

    q_command = data.qpos[
        env.qpos_ids
    ].copy()

    errors = []

    for step in range(env.max_steps):

        # Desired position and velocity
        target, target_velocity = (
            env.desired_trajectory(data.time)
        )

        # True hand position
        actual_position = data.xpos[
            env.hand_id
        ].copy()

        # Sensor measurement with added noise
        measured_position = (
            actual_position
            + rng.normal(0, noise_std, size=3)
        )

        # Feedback error based on noisy measurement
        position_error = (
            target - measured_position
        )

        # Feedforward + feedback
        cart_velocity = (
            target_velocity + KP * position_error
        )

        speed = np.linalg.norm(cart_velocity)

        if speed > MAX_CART_SPEED:
            cart_velocity *= MAX_CART_SPEED / speed

        # Calculate Jacobian
        jac_pos = np.zeros((3, mj_model.nv))
        jac_rot = np.zeros((3, mj_model.nv))

        mujoco.mj_jacBody(
            mj_model,
            data,
            jac_pos,
            jac_rot,
            env.hand_id
        )

        J = jac_pos[:, env.dof_ids]

        # Damped least-squares IK
        A = (
            J @ J.T
            + DAMPING**2 * np.eye(3)
        )

        q_velocity = J.T @ np.linalg.solve(
            A,
            cart_velocity
        )

        q_velocity = np.clip(
            q_velocity,
            -MAX_JOINT_SPEED,
            MAX_JOINT_SPEED
        )

        # Update position commands
        q_command += (
            q_velocity * env.control_dt
        )

        q_command = np.clip(
            q_command,
            env.joint_limits[:, 0],
            env.joint_limits[:, 1]
        )

        data.ctrl[env.actuator_ids] = q_command

        # Advance physics
        for _ in range(env.physics_steps):
            mujoco.mj_step(mj_model, data)

        # Compare against true target position
        target_after_step, _ = (
            env.desired_trajectory(data.time)
        )

        actual_after_step = data.xpos[
            env.hand_id
        ].copy()

        error = np.linalg.norm(
            target_after_step - actual_after_step
        )

        errors.append(error)

    env.close()

    return np.array(errors)


# ==========================================
# 6. Run robustness experiments
# ==========================================

records = []

for noise_mm in NOISE_LEVELS_MM:

    noise_std = noise_mm / 1000.0

    for seed in SEEDS:

        for controller_name, evaluator in [
            ("Classical IK", evaluate_ik),
            ("PPO", evaluate_ppo),
        ]:

            errors = evaluator(noise_std, seed)

            records.append({
                "controller": controller_name,
                "noise_mm": noise_mm,
                "seed": seed,
                "mean_error_mm": np.mean(errors) * 1000,
                "rmse_mm": np.sqrt(
                    np.mean(errors**2)
                ) * 1000,
                "max_error_mm": np.max(errors) * 1000,
            })

            print(
                f"{controller_name:12s} | "
                f"Noise: {noise_mm} mm | "
                f"Seed: {seed} | "
                f"Mean error: {np.mean(errors)*1000:.3f} mm"
            )


# ==========================================
# 7. Save detailed results
# ==========================================

df = pd.DataFrame(records)

df.to_csv(
    RESULTS_DIR / "robustness_trials.csv",
    index=False
)


# ==========================================
# 8. Summarise results
# ==========================================

summary = (
    df.groupby(["controller", "noise_mm"])
    .agg(
        mean_error_mm=("mean_error_mm", "mean"),
        rmse_mm=("rmse_mm", "mean"),
        max_error_mm=("max_error_mm", "mean"),
    )
    .reset_index()
)

print("\nROBUSTNESS TEST RESULTS")
print("=" * 60)

print(summary.round(3).to_string(index=False))

summary.to_csv(
    RESULTS_DIR / "robustness_summary.csv",
    index=False
)


# ==========================================
# 9. Plot robustness comparison
# ==========================================

plt.figure(figsize=(9, 5))

for controller_name in ["Classical IK", "PPO"]:

    controller_data = summary[
        summary["controller"] == controller_name
    ]

    plt.plot(
        controller_data["noise_mm"],
        controller_data["mean_error_mm"],
        marker="o",
        linewidth=2,
        label=controller_name
    )

plt.xlabel("Position measurement noise (mm, standard deviation)")
plt.ylabel("Mean tracking error (mm)")

plt.title("Franka Panda: Robustness to Position Measurement Noise")

plt.xticks(NOISE_LEVELS_MM)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "robustness_comparison.png",
    dpi=300
)

plt.show()

print("\nRobustness testing complete!")