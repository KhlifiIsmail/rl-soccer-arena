"""Soccer Gym environment - SELF-PLAY MODE."""

from typing import Any, Dict, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from src.environments.agent import AgentConfig, SoccerAgent, Team
from src.environments.ball import BallConfig, SoccerBall
from src.environments.field import FieldDimensions, SoccerField


class SoccerEnv(gym.Env):
    """3D Soccer environment with self-play support."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 2000,
        time_step: float = 0.01,
        field_config: FieldDimensions | None = None,
        agent_config: AgentConfig | None = None,
        ball_config: BallConfig | None = None,
        reward_goal_scored: float = 100.0,
        reward_goal_conceded: float = -50.0,
        reward_own_goal: float = -100.0,
        reward_ball_touch: float = 2.0,
        reward_ball_to_goal: float = 5.0,
        reward_no_action: float = 0.0,
    ) -> None:
        """Initialize Soccer environment."""
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.time_step = time_step

        self.field_config = field_config or FieldDimensions()
        self.agent_config = agent_config or AgentConfig()
        self.ball_config = ball_config or BallConfig()

        self.reward_goal_scored = reward_goal_scored
        self.reward_goal_conceded = reward_goal_conceded
        self.reward_own_goal = reward_own_goal
        self.reward_ball_touch = reward_ball_touch
        self.reward_ball_to_goal = reward_ball_to_goal
        self.reward_no_action = reward_no_action

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.physics_client: int | None = None
        self.field: SoccerField | None = None
        self.blue_agent: SoccerAgent | None = None
        self.red_agent: SoccerAgent | None = None
        self.ball: SoccerBall | None = None

        self.current_step = 0
        self.episode_count = 0
        self.blue_goals = 0
        self.red_goals = 0

        self.prev_ball_distance = 999.0
        self.prev_ball_to_red_goal_dist = 999.0
        self.blue_touched_ball_last = False

        # Self-play opponent policy
        self.opponent_policy: Optional[Any] = None

        self._init_pybullet()

    def set_opponent_policy(self, policy: Any) -> None:
        """Set opponent policy for self-play."""
        self.opponent_policy = policy

    def _init_pybullet(self) -> None:
        """Initialize PyBullet."""
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=25.0,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.physics_client
            )

        self.field = SoccerField(
            physics_client=self.physics_client,
            dimensions=self.field_config
        )

        blue_start_pos = np.array([-self.field_config.half_length + 3.0, 0.0, 0.5])
        red_start_pos = np.array([self.field_config.half_length - 3.0, 0.0, 0.5])

        self.blue_agent = SoccerAgent(
            team=Team.BLUE,
            physics_client=self.physics_client,
            start_position=blue_start_pos,
            config=self.agent_config
        )

        self.red_agent = SoccerAgent(
            team=Team.RED,
            physics_client=self.physics_client,
            start_position=red_start_pos,
            config=self.agent_config
        )

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
        """Reset environment."""
        super().reset(seed=seed)

        if self.field:
            self.field.reset()
        if self.blue_agent:
            self.blue_agent.reset()
        if self.red_agent:
            self.red_agent.reset()
        if self.ball:
            self.ball.reset()

        self.current_step = 0
        self.episode_count += 1
        self.blue_goals = 0
        self.red_goals = 0

        ball_pos = self.ball.get_position()
        blue_pos = self.blue_agent.get_position()
        
        self.prev_ball_distance = np.linalg.norm(blue_pos - ball_pos)
        self.prev_ball_to_red_goal_dist = np.linalg.norm(ball_pos - self.field_config.red_goal_center)
        self.blue_touched_ball_last = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep."""
        self.blue_agent.apply_action(action)

        red_action = self._get_red_agent_action()
        self.red_agent.apply_action(red_action)

        p.stepSimulation(physicsClientId=self.physics_client)

        self.current_step += 1

        terminated, goal_reward, own_goal = self._check_goal()
        truncated = self.current_step >= self.max_episode_steps

        reward = self._calculate_reward(goal_reward, own_goal)

        observation = self._get_observation()
        info = self._get_info()
        info["goal_scored"] = goal_reward > 0 and not own_goal
        info["goal_conceded"] = goal_reward < 0 or own_goal

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get observation."""
        blue_pos = self.blue_agent.get_position()
        blue_vel = self.blue_agent.get_velocity()

        ball_pos = self.ball.get_position()
        ball_vel = self.ball.get_velocity()

        red_pos = self.red_agent.get_position()
        red_vel = self.red_agent.get_velocity()

        observation = np.concatenate([
            blue_pos,
            blue_vel,
            ball_pos,
            ball_vel,
            red_pos,
            red_vel,
        ]).astype(np.float32)

        return observation

    def _get_red_observation(self) -> np.ndarray:
        """Get observation from red agent's perspective (flipped)."""
        red_pos = self.red_agent.get_position()
        red_vel = self.red_agent.get_velocity()

        ball_pos = self.ball.get_position()
        ball_vel = self.ball.get_velocity()

        blue_pos = self.blue_agent.get_position()
        blue_vel = self.blue_agent.get_velocity()

        # Flip x-coordinates for red's perspective
        observation = np.concatenate([
            [-red_pos[0], red_pos[1], red_pos[2]],
            [-red_vel[0], red_vel[1], red_vel[2]],
            [-ball_pos[0], ball_pos[1], ball_pos[2]],
            [-ball_vel[0], ball_vel[1], ball_vel[2]],
            [-blue_pos[0], blue_pos[1], blue_pos[2]],
            [-blue_vel[0], blue_vel[1], blue_vel[2]],
        ]).astype(np.float32)

        return observation

    def _get_red_agent_action(self) -> np.ndarray:
        """Get red agent action using opponent policy or random."""
        if self.opponent_policy is not None:
            try:
                red_obs = self._get_red_observation()
                action, _ = self.opponent_policy.predict(red_obs, deterministic=False)
                # Flip x-axis action back
                return np.array([-action[0], action[1]], dtype=np.float32)
            except Exception as e:
                # Fallback to random if policy fails
                return np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
        else:
            # Random opponent until self-play kicks in
            if np.random.random() < 0.3:
                return np.array([0.0, 0.0], dtype=np.float32)
            return np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)

    def _check_goal(self) -> Tuple[bool, float, bool]:
        """Check goals - detect own goals."""
        ball_pos = self.ball.get_position()

        if self.field_config.is_in_red_goal(ball_pos):
            self.blue_goals += 1
            return True, self.reward_goal_scored, False

        if self.field_config.is_in_blue_goal(ball_pos):
            self.red_goals += 1
            if self.blue_touched_ball_last:
                return True, self.reward_own_goal, True
            else:
                return True, self.reward_goal_conceded, False

        return False, 0.0, False

    def _calculate_reward(self, goal_reward: float, own_goal: bool) -> float:
        """Calculate reward - POSITIVE FOCUSED."""
        reward = goal_reward

        ball_pos = self.ball.get_position()
        blue_pos = self.blue_agent.get_position()

        # REWARD FOR GETTING CLOSER TO BALL
        current_distance = np.linalg.norm(blue_pos - ball_pos)
        distance_improvement = self.prev_ball_distance - current_distance
        
        if distance_improvement > 0:
            reward += distance_improvement * 0.5
        
        self.prev_ball_distance = current_distance

        # BIG REWARD FOR TOUCHING BALL
        if current_distance < 1.0:
            reward += self.reward_ball_touch
            self.blue_touched_ball_last = True
        else:
            self.blue_touched_ball_last = False

        # REWARD FOR MOVING BALL TOWARDS RED GOAL
        current_ball_to_goal = np.linalg.norm(ball_pos - self.field_config.red_goal_center)
        ball_progress = self.prev_ball_to_red_goal_dist - current_ball_to_goal
        
        if ball_progress > 0 and current_distance < 2.0:
            reward += ball_progress * self.reward_ball_to_goal

        self.prev_ball_to_red_goal_dist = current_ball_to_goal

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
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
        """Render environment."""
        if self.render_mode == "rgb_array":
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
            rgb_array = rgb_array[:, :, :3]

            return rgb_array

        return None

    def close(self) -> None:
        """Clean up."""
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