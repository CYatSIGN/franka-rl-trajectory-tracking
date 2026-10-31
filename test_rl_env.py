import numpy as np

from stable_baselines3.common.env_checker import check_env

from panda_rl_env import PandaTrackingEnv


# ==========================================
# 1. Create the environment
# ==========================================

env = PandaTrackingEnv()

print("Panda RL environment created!")


# ==========================================
# 2. Check observation and action spaces
# ==========================================

print("\nOBSERVATION SPACE")
print(env.observation_space)

print("\nACTION SPACE")
print(env.action_space)


# ==========================================
# 3. Validate the Gymnasium interface
# ==========================================

print("\nChecking environment...")

check_env(env, warn=True)

print("Environment validation passed!")


# ==========================================
# 4. Reset the robot
# ==========================================

obs, info = env.reset(seed=42)

print("\nInitial observation shape:", obs.shape)

assert obs.shape == (43,)

print("Initial tracking error:", np.linalg.norm(obs[26:29]))


# ==========================================
# 5. Run one episode with random actions
# ==========================================

total_reward = 0.0
tracking_errors = []

terminated = False
truncated = False

step_count = 0

while not (terminated or truncated):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward

    tracking_errors.append(
        info["tracking_error"]
    )

    step_count += 1

    if step_count % 100 == 0:

        print(
            f"Step {step_count} | "
            f"Error: {info['tracking_error']:.4f} m"
        )


# ==========================================
# 6. Show results
# ==========================================

print("\nRANDOM POLICY TEST COMPLETE")

print("Steps completed:", step_count)

print("Total reward:", total_reward)

print(
    "Mean tracking error:",
    np.mean(tracking_errors),
    "m"
)

env.close()