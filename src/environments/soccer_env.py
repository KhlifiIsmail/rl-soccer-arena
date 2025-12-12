"""Soccer Gym environment with 3D PyBullet physics and multi-agent support.

This is the main environment that integrates field, agents, and ball into a complete
reinforcement learning environment compatible with Gym and Stable-Baselines3.
"""

from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from src.environments.agent import AgentConfig, SoccerAgent, Team
from src.environments.ball import BallConfig, SoccerBall
from src.environments.field import FieldDimensions, SoccerField


class SoccerEnv(gym.Env):
    """3D Soccer Gym environment with two competing agents.

    Environment features:
    - 3D physics simulation via PyBullet
    - Two agents (blue vs red) with continuous actions
    - One ball with realistic physics
    - Episode termination on goal or max timesteps
    - Reward shaping with goal rewards and ball proximity

    Observation space (per agent):
        - Own position (3): [x, y, z]
        - Own velocity (3): [vx, vy, vz]
        - Own orientation (1): [yaw]
        - Ball position (3): [x, y, z]
        - Ball velocity (3): [vx, vy, vz]
        - Opponent position (3): [x, y, z]
        - Opponent velocity (3): [vx, vy, vz]
        - Distance to own goal (1)
        - Distance to opponent goal (1)
        Total: 21 dimensions

    Action space (per agent):
        - Forward/backward (1): [-1, 1]
        - Strafe left/right (1): [-1, 1]
        - Turn left/right (1): [-1, 1]
        Total: 3 dimensions (continuous)

    Reward:
        - +10.0: Scoring a goal
        - -10.0: Conceding a goal
        - +0.01: Ball proximity reward (inversely proportional to distance)
        - Small time penalty to encourage faster play
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 2000,
        time_step: float = 1.0 / 240.0,
        field_config: FieldDimensions | None = None,
        agent_config: AgentConfig | None = None,
        ball_config: BallConfig | None = None,
        reward_goal_scored: float = 10.0,
        reward_goal_conceded: float = -10.0,
        reward_ball_proximity_scale: float = 0.01,
        reward_time_penalty: float = -0.001,
    ) -> None:
        """Initialize Soccer environment.

        Args:
            render_mode: Rendering mode ('human' for GUI, 'rgb_array' for headless).
            max_episode_steps: Maximum timesteps per episode.
            time_step: Physics simulation timestep.
            field_config: Field dimensions configuration.
            agent_config: Agent physical configuration.
            ball_config: Ball physical configuration.
            reward_goal_scored: Reward for scoring a goal.
            reward_goal_conceded: Penalty for conceding a goal.
            reward_ball_proximity_scale: Scaling factor for ball proximity reward.
            reward_time_penalty: Small penalty per timestep.
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.time_step = time_step

        # Configurations
        self.field_config = field_config or FieldDimensions()
        self.agent_config = agent_config or AgentConfig()
        self.ball_config = ball_config or BallConfig()

        # Reward parameters
        self.reward_goal_scored = reward_goal_scored
        self.reward_goal_conceded = reward_goal_conceded
        self.reward_ball_proximity_scale = reward_ball_proximity_scale
        self.reward_time_penalty = reward_time_penalty

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # PyBullet setup
        self.physics_client: int | None = None
        self.field: SoccerField | None = None
        self.blue_agent: SoccerAgent | None = None
        self.red_agent: SoccerAgent | None = None
        self.ball: SoccerBall | None = None

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        self.blue_goals = 0
        self.red_goals = 0

        # Previous state for reward shaping
        self.prev_ball_distance_blue = 0.0
        self.prev_ball_distance_red = 0.0

        # Initialize PyBullet
        self._init_pybullet()

    def _init_pybullet(self) -> None:
        """Initialize PyBullet physics client and create entities."""
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Configure PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)

        # Configure camera for better viewing
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=25.0,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.physics_client
            )

        # Create field
        self.field = SoccerField(
            physics_client=self.physics_client,
            dimensions=self.field_config
        )

        # Create agents
        blue_start_pos = np.array([-self.field_config.half_length + 3.0, 0.0, 0.5])
        red_start_pos = np.array([self.field_config.half_length - 3.0, 0.0, 0.5])

        self.blue_agent = SoccerAgent(
            team=Team.BLUE,
            physics_client=self.physics_client,
            start_position=blue_start_pos,
            start_orientation=0.0,
            config=self.agent_config
        )

        self.red_agent = SoccerAgent(
            team=Team.RED,
            physics_client=self.physics_client,
            start_position=red_start_pos,
            start_orientation=np.pi,  # Face opposite direction
            config=self.agent_config
        )

        # Create ball
        ball_start_pos = np.array([0.0, 0.0, 0.5])
        self.ball = SoccerBall(
            physics_client=self.physics_client,
            start_position=ball_start_pos,
            config=self.ball_config
        )

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # Reset entities
        if self.field:
            self.field.reset()
        if self.blue_agent:
            self.blue_agent.reset()
        if self.red_agent:
            self.red_agent.reset()
        if self.ball:
            self.ball.reset()

        # Reset episode state
        self.current_step = 0
        self.episode_count += 1

        # Initialize previous distances
        self.prev_ball_distance_blue = self.blue_agent.get_distance_to(self.ball.get_position())
        self.prev_ball_distance_red = self.red_agent.get_distance_to(self.ball.get_position())

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep in environment.

        Args:
            action: Action for blue agent (controlling agent).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Apply blue agent action
        self.blue_agent.apply_action(action)

        # Red agent uses opponent policy (set externally or random)
        # For now, use simple heuristic: move towards ball
        red_action = self._get_red_agent_action()
        self.red_agent.apply_action(red_action)

        # Step physics simulation
        p.stepSimulation(physicsClientId=self.physics_client)

        self.current_step += 1

        # Check termination conditions
        terminated, goal_reward = self._check_goal()
        truncated = self.current_step >= self.max_episode_steps

        # Calculate reward
        reward = self._calculate_reward(goal_reward)

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info["goal_scored"] = goal_reward > 0
        info["goal_conceded"] = goal_reward < 0

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation for blue agent.

        Returns:
            Observation array (21 dimensions).
        """
        # Blue agent self state (7)
        blue_pos = self.blue_agent.get_position()
        blue_vel = self.blue_agent.get_velocity()
        blue_yaw = self.blue_agent.get_orientation()

        # Ball state (6)
        ball_pos = self.ball.get_position()
        ball_vel = self.ball.get_velocity()

        # Red agent state (6)
        red_pos = self.red_agent.get_position()
        red_vel = self.red_agent.get_velocity()

        # Goal distances (2)
        dist_to_blue_goal = np.linalg.norm(blue_pos - self.field_config.blue_goal_center)
        dist_to_red_goal = np.linalg.norm(blue_pos - self.field_config.red_goal_center)

        observation = np.concatenate([
            blue_pos,
            blue_vel,
            [blue_yaw],
            ball_pos,
            ball_vel,
            red_pos,
            red_vel,
            [dist_to_blue_goal, dist_to_red_goal]
        ]).astype(np.float32)

        return observation

    def _get_red_agent_action(self) -> np.ndarray:
        """Get action for red agent (simple heuristic or learned policy).

        This is a placeholder for opponent policy. In self-play, this will be
        replaced with a previous version of the trained agent.

        Returns:
            Action array (3 dimensions).
        """
        # Simple heuristic: move towards ball
        ball_pos = self.ball.get_position()
        red_pos = self.red_agent.get_position()

        direction = ball_pos - red_pos
        distance = np.linalg.norm(direction[:2])  # Only x,y

        if distance > 0.1:
            direction_normalized = direction / distance
            red_forward = self.red_agent.get_forward_vector()

            # Calculate angle to ball
            angle_to_ball = np.arctan2(direction_normalized[1], direction_normalized[0])
            red_yaw = self.red_agent.get_orientation()
            angle_diff = angle_to_ball - red_yaw

            # Normalize angle to [-pi, pi]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            # Simple control
            forward = 0.8 if abs(angle_diff) < np.pi / 4 else 0.3
            turn = np.clip(angle_diff * 2.0, -1.0, 1.0)

            return np.array([forward, 0.0, turn])
        else:
            # Kick ball towards blue goal
            return np.array([1.0, 0.0, 0.0])

    def _check_goal(self) -> Tuple[bool, float]:
        """Check if a goal has been scored.

        Returns:
            Tuple of (terminated, goal_reward).
        """
        ball_pos = self.ball.get_position()

        if self.field_config.is_in_blue_goal(ball_pos):
            # Red scored on blue
            self.red_goals += 1
            return True, self.reward_goal_conceded

        if self.field_config.is_in_red_goal(ball_pos):
            # Blue scored on red
            self.blue_goals += 1
            return True, self.reward_goal_scored

        return False, 0.0

    def _calculate_reward(self, goal_reward: float) -> float:
        """Calculate reward for current step.

        Args:
            goal_reward: Reward from goal scoring (if any).

        Returns:
            Total reward for this timestep.
        """
        reward = goal_reward

        # Ball proximity reward (encourages agent to chase ball)
        ball_pos = self.ball.get_position()
        current_distance = self.blue_agent.get_distance_to(ball_pos)

        # Reward for getting closer to ball
        proximity_reward = (self.prev_ball_distance_blue - current_distance) * self.reward_ball_proximity_scale
        reward += proximity_reward

        # Update previous distance
        self.prev_ball_distance_blue = current_distance

        # Small time penalty
        reward += self.reward_time_penalty

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary.

        Returns:
            Info dictionary with episode statistics.
        """
        return {
            "episode": self.episode_count,
            "step": self.current_step,
            "blue_goals": self.blue_goals,
            "red_goals": self.red_goals,
            "ball_position": self.ball.get_position().copy(),
            "ball_velocity": self.ball.get_velocity().copy(),
            "blue_position": self.blue_agent.get_position().copy(),
            "red_position": self.red_agent.get_position().copy(),
        }

    def render(self) -> np.ndarray | None:
        """Render environment.

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "rgb_array":
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=25.0,
                yaw=0,
                pitch=-45,
                roll=0,
                upAxisIndex=2
            )

            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )

            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.physics_client
            )

            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (480, 640, 4))
            rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

            return rgb_array

        return None

    def close(self) -> None:
        """Clean up resources."""
        if self.ball:
            self.ball.cleanup()
        if self.blue_agent:
            self.blue_agent.cleanup()
        if self.red_agent:
            self.red_agent.cleanup()
        if self.field:
            self.field.cleanup()

        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = None
