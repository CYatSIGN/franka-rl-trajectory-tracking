import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np

from pathlib import Path


class PandaTrackingEnv(gym.Env):

    """
    Reinforcement learning environment for
    Franka Panda circular trajectory tracking.
    """

    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # =====================================
        # 1. Simulation settings
        # =====================================

        self.duration = 10.0
        self.control_frequency = 50

        self.control_dt = 1.0 / self.control_frequency

        self.radius = 0.04
        self.omega = 2 * np.pi / self.duration

        self.max_steps = int(
            self.duration * self.control_frequency
        )

        # Maximum joint-position change per step
        self.max_action_delta = 0.01

        # =====================================
        # 2. Load Franka Panda
        # =====================================

        model_path = (
            Path.home()
            / "robotics-projects"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "scene.xml"
        )

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.model = mujoco.MjModel.from_xml_path(
            str(model_path)
        )

        self.data = mujoco.MjData(self.model)

        # =====================================
        # 3. Find home keyframe
        # =====================================

        self.home_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home"
        )

        if self.home_id == -1:
            raise RuntimeError("Home keyframe not found")

        # =====================================
        # 4. Identify robot components
        # =====================================

        def find_id(object_type, name):

            object_id = mujoco.mj_name2id(
                self.model,
                object_type,
                name
            )

            if object_id == -1:
                raise RuntimeError(
                    f"Model element not found: {name}"
                )

            return object_id

        self.hand_id = find_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "hand"
        )

        self.joint_ids = np.array([
            find_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                f"joint{i}"
            )
            for i in range(1, 8)
        ])

        self.actuator_ids = np.array([
            find_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"actuator{i}"
            )
            for i in range(1, 8)
        ])

        self.qpos_ids = self.model.jnt_qposadr[
            self.joint_ids
        ]

        self.dof_ids = self.model.jnt_dofadr[
            self.joint_ids
        ]

        self.joint_limits = self.model.jnt_range[
            self.joint_ids
        ]

        # =====================================
        # 5. Physics stepping
        # =====================================

        physics_dt = self.model.opt.timestep

        self.physics_steps = round(
            self.control_dt / physics_dt
        )

        if self.physics_steps < 1 or not np.isclose(
            self.physics_steps * physics_dt,
            self.control_dt
        ):
            raise ValueError(
                "Control timestep must be an integer "
                "multiple of the physics timestep."
            )

        # =====================================
        # 6. Action space
        # =====================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # =====================================
        # 7. Observation space
        # =====================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(43,),
            dtype=np.float32
        )

        self.current_step = 0

        self.previous_action = np.zeros(
            7, dtype=np.float32
        )

        self.q_command = np.zeros(7)

        self.center = np.zeros(3)

        # Initialise simulation state
        self.reset()

    # =========================================
    # 8. Desired circular trajectory
    # =========================================

    def desired_trajectory(self, t):

        theta = self.omega * t

        position = self.center + np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            0.0
        ])

        velocity = np.array([
            -self.radius * self.omega * np.sin(theta),
             self.radius * self.omega * np.cos(theta),
             0.0
        ])

        return position, velocity

    # =========================================
    # 9. Construct observation
    # =========================================

    def _get_obs(self):

        q = self.data.qpos[
            self.qpos_ids
        ].copy()

        q_velocity = self.data.qvel[
            self.dof_ids
        ].copy()

        hand_position = self.data.xpos[
            self.hand_id
        ].copy()

        # Calculate hand velocity from Jacobian
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        mujoco.mj_jacBody(
            self.model,
            self.data,
            jac_pos,
            jac_rot,
            self.hand_id
        )

        hand_velocity = jac_pos @ self.data.qvel

        target_position, target_velocity = (
            self.desired_trajectory(self.data.time)
        )

        tracking_error = (
            target_position - hand_position
        )

        observation = np.concatenate([
            q,
            q_velocity,
            hand_position,
            hand_velocity,
            target_position,
            target_velocity,
            tracking_error,
            self.q_command,
            self.previous_action
        ])

        return observation.astype(np.float32)

    # =========================================
    # 10. Reset environment
    # =========================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.home_id
        )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.current_step = 0

        self.previous_action = np.zeros(
            7, dtype=np.float32
        )

        self.q_command = self.data.qpos[
            self.qpos_ids
        ].copy()

        # Match the initial joint targets
        self.data.ctrl[
            self.actuator_ids
        ] = self.q_command

        # Keep the initial target reachable
        initial_position = self.data.xpos[
            self.hand_id
        ].copy()

        self.center = initial_position.copy()
        self.center[0] -= self.radius

        return self._get_obs(), {}

    # =========================================
    # 11. Apply action
    # =========================================

    def step(self, action):

        action = np.asarray(
            action, dtype=np.float32
        )

        action = np.clip(
            action, -1.0, 1.0
        )

        action_change = (
            action - self.previous_action
        )

        # Convert normalised action to
        # joint-position increment
        joint_delta = (
            action * self.max_action_delta
        )

        self.q_command += joint_delta

        self.q_command = np.clip(
            self.q_command,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )

        # Send position targets to MuJoCo
        self.data.ctrl[
            self.actuator_ids
        ] = self.q_command

        # Simulate physics
        for _ in range(self.physics_steps):

            mujoco.mj_step(
                self.model,
                self.data
            )

        self.current_step += 1

        # Get new observation
        obs = self._get_obs()

        tracking_error = obs[26:29]

        position_error = np.linalg.norm(
            tracking_error
        )

        joint_velocity = self.data.qvel[
            self.dof_ids
        ]

        # =====================================
        # 12. Reward function
        # =====================================

        reward = (
            -10.0 * position_error
            -0.02 * np.sum(action_change ** 2)
            -0.002 * np.sum(joint_velocity ** 2)
        )

        # =====================================
        # 13. Episode termination
        # =====================================

        state_is_valid = (
            np.all(np.isfinite(obs))
            and np.isfinite(reward)
        )

        terminated = not state_is_valid

        truncated = (
            self.current_step >= self.max_steps
        )

        if terminated:
            reward = -100.0

        # Store the latest action
        self.previous_action = action.copy()

        # Update action component for the
        # next observation
        obs[-7:] = self.previous_action

        info = {
            "tracking_error": float(position_error),
            "simulation_time": float(self.data.time)
        }

        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            info
        )

    # =========================================
    # 14. Close environment
    # =========================================

    def close(self):
        pass