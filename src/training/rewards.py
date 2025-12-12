"""Reward shaping functions for soccer environment.

This module provides customizable reward functions for training soccer agents.
Reward shaping helps guide learning by providing intermediate feedback signals.
"""

from typing import Dict

import numpy as np


class RewardShaper:
    """Reward shaping for soccer environment.

    Provides structured rewards to encourage desired behaviors:
    - Goal scoring/conceding (terminal rewards)
    - Ball possession and control
    - Positioning and spatial awareness
    - Energy efficiency

    Attributes:
        goal_scored: Reward for scoring a goal.
        goal_conceded: Penalty for conceding a goal.
        ball_proximity_scale: Scale for ball proximity reward.
        ball_possession_bonus: Bonus for being close to ball.
        goal_approach_scale: Scale for approaching opponent goal with ball.
        defensive_positioning: Reward for good defensive positioning.
        time_penalty: Small penalty per timestep.
    """

    def __init__(
        self,
        goal_scored: float = 10.0,
        goal_conceded: float = -10.0,
        ball_proximity_scale: float = 0.01,
        ball_possession_bonus: float = 0.05,
        goal_approach_scale: float = 0.02,
        defensive_positioning: float = 0.01,
        time_penalty: float = -0.001,
    ) -> None:
        """Initialize reward shaper.

        Args:
            goal_scored: Reward for scoring a goal.
            goal_conceded: Penalty for conceding a goal.
            ball_proximity_scale: Scale for getting closer to ball.
            ball_possession_bonus: Bonus when very close to ball.
            goal_approach_scale: Scale for moving ball toward goal.
            defensive_positioning: Reward for staying between ball and own goal.
            time_penalty: Small penalty per timestep.
        """
        self.goal_scored = goal_scored
        self.goal_conceded = goal_conceded
        self.ball_proximity_scale = ball_proximity_scale
        self.ball_possession_bonus = ball_possession_bonus
        self.goal_approach_scale = goal_approach_scale
        self.defensive_positioning = defensive_positioning
        self.time_penalty = time_penalty

    def calculate_reward(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        opponent_pos: np.ndarray,
        own_goal_pos: np.ndarray,
        opponent_goal_pos: np.ndarray,
        prev_ball_distance: float,
        goal_scored: bool = False,
        goal_conceded: bool = False,
    ) -> Dict[str, float]:
        """Calculate shaped reward for current state.

        Args:
            agent_pos: Agent position [x, y, z].
            agent_vel: Agent velocity [vx, vy, vz].
            ball_pos: Ball position [x, y, z].
            ball_vel: Ball velocity [vx, vy, vz].
            opponent_pos: Opponent position [x, y, z].
            own_goal_pos: Own goal center position [x, y, z].
            opponent_goal_pos: Opponent goal center position [x, y, z].
            prev_ball_distance: Previous distance to ball.
            goal_scored: Whether agent scored a goal.
            goal_conceded: Whether agent conceded a goal.

        Returns:
            Dictionary of reward components and total reward.
        """
        rewards = {}

        # Terminal rewards
        if goal_scored:
            rewards["goal_scored"] = self.goal_scored
        if goal_conceded:
            rewards["goal_conceded"] = self.goal_conceded

        # Ball proximity reward (encourage chasing ball)
        current_ball_distance = np.linalg.norm(agent_pos - ball_pos)
        proximity_reward = (prev_ball_distance - current_ball_distance) * self.ball_proximity_scale
        rewards["ball_proximity"] = proximity_reward

        # Ball possession bonus (very close to ball)
        if current_ball_distance < 1.0:
            rewards["ball_possession"] = self.ball_possession_bonus
        else:
            rewards["ball_possession"] = 0.0

        # Goal approach reward (moving ball towards opponent goal)
        ball_to_goal_distance = np.linalg.norm(ball_pos - opponent_goal_pos)
        if current_ball_distance < 2.0:  # Only if close to ball
            # Reward for ball moving toward opponent goal
            ball_to_goal_direction = (opponent_goal_pos - ball_pos) / (ball_to_goal_distance + 1e-8)
            ball_velocity_toward_goal = np.dot(ball_vel, ball_to_goal_direction)
            rewards["goal_approach"] = ball_velocity_toward_goal * self.goal_approach_scale
        else:
            rewards["goal_approach"] = 0.0

        # Defensive positioning (stay between ball and own goal when far from ball)
        if current_ball_distance > 3.0:
            agent_to_ball = ball_pos - agent_pos
            agent_to_own_goal = own_goal_pos - agent_pos

            # Check if agent is between ball and goal
            if np.dot(agent_to_ball, agent_to_own_goal) < 0:
                rewards["defensive_position"] = self.defensive_positioning
            else:
                rewards["defensive_position"] = -self.defensive_positioning
        else:
            rewards["defensive_position"] = 0.0

        # Time penalty (encourage faster play)
        rewards["time_penalty"] = self.time_penalty

        # Total reward
        rewards["total"] = sum(rewards.values())

        return rewards


class SparseRewardShaper(RewardShaper):
    """Sparse reward shaper (only goal rewards).

    Provides only terminal rewards for goals, no intermediate shaping.
    Useful for benchmarking against dense reward shaping.
    """

    def __init__(self, goal_scored: float = 1.0, goal_conceded: float = -1.0) -> None:
        """Initialize sparse reward shaper.

        Args:
            goal_scored: Reward for scoring a goal.
            goal_conceded: Penalty for conceding a goal.
        """
        super().__init__(
            goal_scored=goal_scored,
            goal_conceded=goal_conceded,
            ball_proximity_scale=0.0,
            ball_possession_bonus=0.0,
            goal_approach_scale=0.0,
            defensive_positioning=0.0,
            time_penalty=0.0,
        )


class CurriculumRewardShaper(RewardShaper):
    """Curriculum learning reward shaper with adaptive weights.

    Gradually shifts from dense shaping rewards to sparse terminal rewards
    as training progresses.

    Attributes:
        curriculum_stage: Current curriculum stage (0-1).
        initial_weights: Initial reward weights (dense shaping).
        final_weights: Final reward weights (mostly sparse).
    """

    def __init__(self) -> None:
        """Initialize curriculum reward shaper."""
        super().__init__()
        self.curriculum_stage = 0.0  # 0 = start, 1 = end

    def update_curriculum(self, stage: float) -> None:
        """Update curriculum stage.

        Args:
            stage: Curriculum progress (0.0 to 1.0).
        """
        self.curriculum_stage = np.clip(stage, 0.0, 1.0)

        # Reduce shaping rewards as curriculum progresses
        scale = 1.0 - self.curriculum_stage

        self.ball_proximity_scale = 0.01 * scale
        self.ball_possession_bonus = 0.05 * scale
        self.goal_approach_scale = 0.02 * scale
        self.defensive_positioning = 0.01 * scale
