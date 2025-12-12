"""Metrics tracking and statistics computation.

Provides utilities for tracking training metrics, computing statistics,
and generating performance reports.
"""

from collections import deque
from typing import Any, Dict, List

import numpy as np


class MetricsTracker:
    """Tracks metrics during training.

    Attributes:
        metrics: Dictionary of metric names to value lists.
        window_size: Size of rolling window for statistics.
    """

    def __init__(self, window_size: int = 100) -> None:
        """Initialize metrics tracker.

        Args:
            window_size: Size of rolling window for computing statistics.
        """
        self.metrics: Dict[str, deque] = {}
        self.window_size = window_size

    def add(self, name: str, value: float) -> None:
        """Add metric value.

        Args:
            name: Metric name.
            value: Metric value.
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)

        self.metrics[name].append(value)

    def get_mean(self, name: str) -> float:
        """Get mean of metric over window.

        Args:
            name: Metric name.

        Returns:
            Mean value.
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0

        return float(np.mean(self.metrics[name]))

    def get_std(self, name: str) -> float:
        """Get standard deviation of metric.

        Args:
            name: Metric name.

        Returns:
            Standard deviation.
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0

        return float(np.std(self.metrics[name]))

    def get_last(self, name: str) -> float:
        """Get last value of metric.

        Args:
            name: Metric name.

        Returns:
            Last value.
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0

        return float(self.metrics[name][-1])

    def get_all(self, name: str) -> List[float]:
        """Get all values of metric.

        Args:
            name: Metric name.

        Returns:
            List of all values in window.
        """
        if name not in self.metrics:
            return []

        return list(self.metrics[name])

    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for metric.

        Args:
            name: Metric name.

        Returns:
            Dictionary with mean, std, min, max, last.
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "last": 0.0,
                "count": 0,
            }

        values = np.array(self.metrics[name])

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "last": float(values[-1]),
            "count": len(values),
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics.

        Returns:
            Dictionary mapping metric names to their summaries.
        """
        return {name: self.get_summary(name) for name in self.metrics.keys()}

    def reset(self, name: str | None = None) -> None:
        """Reset metric(s).

        Args:
            name: Metric name to reset, or None to reset all.
        """
        if name is None:
            self.metrics.clear()
        elif name in self.metrics:
            self.metrics[name].clear()


class EpisodeStatistics:
    """Compute statistics over episodes.

    Tracks episode rewards, lengths, goals, and computes aggregated statistics.

    Attributes:
        episode_rewards: List of episode rewards.
        episode_lengths: List of episode lengths.
        goals_scored: List of goals scored per episode.
        goals_conceded: List of goals conceded per episode.
    """

    def __init__(self) -> None:
        """Initialize episode statistics."""
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.goals_scored: List[int] = []
        self.goals_conceded: List[int] = []

    def add_episode(
        self,
        reward: float,
        length: int,
        goals_scored: int = 0,
        goals_conceded: int = 0,
    ) -> None:
        """Add episode statistics.

        Args:
            reward: Total episode reward.
            length: Episode length (timesteps).
            goals_scored: Goals scored in episode.
            goals_conceded: Goals conceded in episode.
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.goals_scored.append(goals_scored)
        self.goals_conceded.append(goals_conceded)

    def get_stats(self, window: int | None = None) -> Dict[str, float]:
        """Get episode statistics.

        Args:
            window: Number of recent episodes to consider (None for all).

        Returns:
            Dictionary of statistics.
        """
        if len(self.episode_rewards) == 0:
            return {}

        if window is not None:
            rewards = self.episode_rewards[-window:]
            lengths = self.episode_lengths[-window:]
            scored = self.goals_scored[-window:]
            conceded = self.goals_conceded[-window:]
        else:
            rewards = self.episode_rewards
            lengths = self.episode_lengths
            scored = self.goals_scored
            conceded = self.goals_conceded

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "total_goals_scored": sum(scored),
            "total_goals_conceded": sum(conceded),
            "goal_difference": sum(scored) - sum(conceded),
            "win_rate": float(sum(1 for s, c in zip(scored, conceded) if s > c) / len(scored)),
            "num_episodes": len(rewards),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.goals_scored.clear()
        self.goals_conceded.clear()
