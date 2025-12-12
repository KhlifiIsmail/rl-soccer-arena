"""Lightweight 2D soccer environment (no PyBullet) for faster training.

This kinematic environment models two agents and a ball on a 2D field:
- Continuous actions: forward, strafe, turn (yaw rate)
- Simple point-mass dynamics with friction and speed limits
- Ball moves in 2D, gains impulse on contact/kick, bounces off walls
- Episode ends on goal or when max steps reached

Observation (15 dims):
    [blue_pos (2), blue_vel (2), blue_yaw (1),
     ball_pos (2), ball_vel (2),
     red_pos (2), red_vel (2),
     dist_to_blue_goal, dist_to_red_goal]
Action (3 dims): [forward, strafe, turn], each in [-1, 1]
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SoccerEnv2D(gym.Env):
    """2D kinematic soccer environment."""

    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 2000,
        time_step: float = 1.0 / 60.0,
        field_length: float = 30.0,
        field_width: float = 20.0,
        goal_width: float = 4.0,
        reward_goal_scored: float = 10.0,
        reward_goal_conceded: float = -10.0,
        reward_ball_proximity_scale: float = 0.01,
        reward_time_penalty: float = -0.001,
        reward_ball_progress_scale: float = 0.05,
        reward_possession_bonus: float = 0.05,
        possession_distance_threshold: float = 1.5,
        max_speed_agent: float = 8.0,
        max_accel_agent: float = 20.0,
        max_speed_ball: float = 20.0,
        ball_friction: float = 0.98,
        agent_friction: float = 0.95,
        kick_impulse: float = 12.0,
        contact_radius: float = 1.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode  # unused
        self.dt = time_step
        self.max_episode_steps = max_episode_steps
        self.field_length = field_length
        self.field_width = field_width
        self.goal_width = goal_width

        # Rewards
        self.reward_goal_scored = reward_goal_scored
        self.reward_goal_conceded = reward_goal_conceded
        self.reward_ball_proximity_scale = reward_ball_proximity_scale
        self.reward_time_penalty = reward_time_penalty
        self.reward_ball_progress_scale = reward_ball_progress_scale
        self.reward_possession_bonus = reward_possession_bonus
        self.possession_distance_threshold = possession_distance_threshold

        # Dynamics parameters
        self.max_speed_agent = max_speed_agent
        self.max_accel_agent = max_accel_agent
        self.max_speed_ball = max_speed_ball
        self.ball_friction = ball_friction
        self.agent_friction = agent_friction
        self.kick_impulse = kick_impulse
        self.contact_radius = contact_radius

        # State
        self.current_step = 0
        self.episode_index = 0
        self.blue_pos = np.zeros(2, dtype=np.float32)
        self.blue_vel = np.zeros(2, dtype=np.float32)
        self.blue_yaw = 0.0
        self.red_pos = np.zeros(2, dtype=np.float32)
        self.red_vel = np.zeros(2, dtype=np.float32)
        self.red_yaw = 0.0
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)

        # For shaping
        self.prev_ball_distance_blue = 0.0
        self.prev_ball_to_red_goal = 0.0

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

    @property
    def half_length(self) -> float:
        return self.field_length / 2.0

    @property
    def half_width(self) -> float:
        return self.field_width / 2.0

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.current_step = 0
        self.episode_index += 1

        # Start positions
        self.blue_pos = np.array([-self.half_length + 3.0, 0.0], dtype=np.float32)
        self.blue_vel = np.zeros(2, dtype=np.float32)
        self.blue_yaw = 0.0

        self.red_pos = np.array([self.half_length - 3.0, 0.0], dtype=np.float32)
        self.red_vel = np.zeros(2, dtype=np.float32)
        self.red_yaw = math.pi

        self.ball_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)

        self.prev_ball_distance_blue = self._distance(self.blue_pos, self.ball_pos)
        self.prev_ball_to_red_goal = self._ball_to_red_goal_dist()

        obs = self._get_observation()
        info = self._get_info(goal_scored=False, goal_conceded=False)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        # Blue action
        self._apply_action(self.blue_pos, self.blue_vel, action, is_blue=True)

        # Red heuristic action
        red_action = self._get_red_action()
        self._apply_action(self.red_pos, self.red_vel, red_action, is_blue=False)

        # Apply friction
        self.blue_vel *= self.agent_friction
        self.red_vel *= self.agent_friction
        self.ball_vel *= self.ball_friction

        # Integrate positions
        self.blue_pos += self.blue_vel * self.dt
        self.red_pos += self.red_vel * self.dt
        self.ball_pos += self.ball_vel * self.dt

        # Boundary handling
        self._clamp_entity(self.blue_pos, self.blue_vel)
        self._clamp_entity(self.red_pos, self.red_vel)
        self._clamp_ball()

        # Agent-ball interaction (kick/impulse)
        self._apply_contact_impulse()

        self.current_step += 1

        terminated, goal_reward = self._check_goal()
        truncated = self.current_step >= self.max_episode_steps

        reward = self._calculate_reward(goal_reward)

        obs = self._get_observation()
        info = self._get_info(goal_scored=goal_reward > 0, goal_conceded=goal_reward < 0)

        return obs, reward, terminated, truncated, info

    def _apply_action(self, pos: np.ndarray, vel: np.ndarray, action: np.ndarray, is_blue: bool) -> None:
        forward, strafe, turn = action
        if is_blue:
            self.blue_yaw += float(turn) * self.dt * 2.5
            yaw = self.blue_yaw
        else:
            self.red_yaw += float(turn) * self.dt * 2.5
            yaw = self.red_yaw

        # World-frame accel from body-frame action
        c, s = math.cos(yaw), math.sin(yaw)
        forward_vec = np.array([c, s], dtype=np.float32)
        strafe_vec = np.array([-s, c], dtype=np.float32)
        accel = forward_vec * float(forward) + strafe_vec * float(strafe)
        accel = accel * self.max_accel_agent

        vel += accel * self.dt
        speed = np.linalg.norm(vel)
        if speed > self.max_speed_agent:
            vel *= self.max_speed_agent / speed

    def _apply_contact_impulse(self) -> None:
        # Blue to ball
        self._kick_if_close(self.blue_pos, self.blue_yaw, self.ball_pos, self.ball_vel)
        # Red to ball
        self._kick_if_close(self.red_pos, self.red_yaw, self.ball_pos, self.ball_vel, towards_blue=True)

    def _kick_if_close(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        towards_blue: bool = False,
    ) -> None:
        delta = ball_pos - agent_pos
        dist = np.linalg.norm(delta)
        if dist < self.contact_radius and dist > 1e-6:
            direction = delta / dist
            if towards_blue:
                # Kick toward blue goal (negative x)
                direction = np.array([-1.0, 0.0], dtype=np.float32)
            # Blend facing direction and direct-to-ball direction for realism
            facing = np.array([math.cos(agent_yaw), math.sin(agent_yaw)], dtype=np.float32)
            kick_dir = 0.5 * facing + 0.5 * direction
            norm = np.linalg.norm(kick_dir)
            if norm > 1e-6:
                kick_dir /= norm
                ball_vel += kick_dir * self.kick_impulse
                speed = np.linalg.norm(ball_vel)
                if speed > self.max_speed_ball:
                    ball_vel *= self.max_speed_ball / speed

    def _clamp_entity(self, pos: np.ndarray, vel: np.ndarray) -> None:
        # Clamp to field; bounce on walls by inverting velocity component
        bounced = False
        if pos[0] < -self.half_length:
            pos[0] = -self.half_length
            vel[0] = abs(vel[0])
            bounced = True
        elif pos[0] > self.half_length:
            pos[0] = self.half_length
            vel[0] = -abs(vel[0])
            bounced = True

        if pos[1] < -self.half_width:
            pos[1] = -self.half_width
            vel[1] = abs(vel[1])
            bounced = True
        elif pos[1] > self.half_width:
            pos[1] = self.half_width
            vel[1] = -abs(vel[1])
            bounced = True

        if bounced:
            vel *= 0.8  # dampen on bounce

    def _clamp_ball(self) -> None:
        self._clamp_entity(self.ball_pos, self.ball_vel)

    def _check_goal(self) -> Tuple[bool, float]:
        # Red goal at +half_length, Blue goal at -half_length
        goal_y_ok = abs(self.ball_pos[1]) < (self.goal_width / 2.0)
        if self.ball_pos[0] > self.half_length and goal_y_ok:
            return True, self.reward_goal_scored  # blue scores
        if self.ball_pos[0] < -self.half_length and goal_y_ok:
            return True, self.reward_goal_conceded  # blue concedes
        return False, 0.0

    def _calculate_reward(self, goal_reward: float) -> float:
        reward = goal_reward

        # Ball proximity shaping
        dist_ball = self._distance(self.blue_pos, self.ball_pos)
        reward += (self.prev_ball_distance_blue - dist_ball) * self.reward_ball_proximity_scale
        self.prev_ball_distance_blue = dist_ball

        # Progress to red goal
        ball_to_red = self._ball_to_red_goal_dist()
        reward += (self.prev_ball_to_red_goal - ball_to_red) * self.reward_ball_progress_scale
        self.prev_ball_to_red_goal = ball_to_red

        # Possession bonus
        if dist_ball < self.possession_distance_threshold:
            reward += self.reward_possession_bonus

        # Time penalty
        reward += self.reward_time_penalty

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        dist_to_blue_goal = self._distance(self.blue_pos, np.array([-self.half_length, 0.0], dtype=np.float32))
        dist_to_red_goal = self._distance(self.blue_pos, np.array([self.half_length, 0.0], dtype=np.float32))
        obs = np.concatenate([
            self.blue_pos,
            self.blue_vel,
            np.array([self.blue_yaw], dtype=np.float32),
            self.ball_pos,
            self.ball_vel,
            self.red_pos,
            self.red_vel,
            np.array([dist_to_blue_goal, dist_to_red_goal], dtype=np.float32),
        ]).astype(np.float32)
        return obs

    def _get_red_action(self) -> np.ndarray:
        # Heuristic: move toward ball, slight facing alignment; small strafe to cut angles.
        delta = self.ball_pos - self.red_pos
        angle_to_ball = math.atan2(delta[1], delta[0])
        angle_diff = self._wrap_pi(angle_to_ball - self.red_yaw)
        forward = 1.0 if abs(angle_diff) < math.pi / 3 else 0.5
        turn = np.clip(angle_diff * 1.5, -1.0, 1.0)
        strafe = 0.2 * np.clip(delta[1], -1.0, 1.0)
        return np.array([forward, strafe, turn], dtype=np.float32)

    def _get_info(self, goal_scored: bool, goal_conceded: bool) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "step": self.current_step,
            "goal_scored": goal_scored,
            "goal_conceded": goal_conceded,
            "ball_position": self.ball_pos.copy(),
            "ball_velocity": self.ball_vel.copy(),
            "blue_position": self.blue_pos.copy(),
            "red_position": self.red_pos.copy(),
        }

    def render(self) -> None:
        # No rendering implemented for the 2D kinematic env.
        return None

    def close(self) -> None:
        return None

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _ball_to_red_goal_dist(self) -> float:
        red_goal = np.array([self.half_length, 0.0], dtype=np.float32)
        return self._distance(self.ball_pos, red_goal)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi
