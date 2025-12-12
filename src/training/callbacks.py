"""Custom Stable-Baselines3 callbacks for training monitoring and checkpointing.

This module provides callbacks for:
- Model checkpointing based on performance metrics
- TensorBoard logging of custom metrics
- Training progress monitoring
- Early stopping
"""

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class SoccerMetricsCallback(BaseCallback):
    """Callback for logging soccer-specific metrics to TensorBoard.

    Tracks and logs:
    - Win rate (goals scored vs conceded)
    - Average episode length
    - Goal scoring frequency
    - Ball control statistics

    Attributes:
        log_freq: Frequency (in steps) to compute and log metrics.
        episode_goals_scored: Goals scored per episode.
        episode_goals_conceded: Goals conceded per episode.
        episode_lengths: Episode lengths.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        """Initialize metrics callback.

        Args:
            log_freq: Frequency to log metrics (in training steps).
            verbose: Verbosity level (0=silent, 1=info).
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Episode statistics
        self.episode_goals_scored: list[int] = []
        self.episode_goals_conceded: list[int] = []
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []

        # Current episode tracking
        self.current_goals_scored = 0
        self.current_goals_conceded = 0

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training, False to stop.
        """
        # Check for episode completion
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                # Episode finished
                ep_info = info["episode"]
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])

            # Track goals
            if info.get("goal_scored", False):
                self.current_goals_scored += 1

            if info.get("goal_conceded", False):
                self.current_goals_conceded += 1

        # Log metrics periodically
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Compute and log metrics to TensorBoard."""
        if len(self.episode_rewards) == 0:
            return

        # Calculate statistics
        mean_reward = np.mean(self.episode_rewards)
        mean_episode_length = np.mean(self.episode_lengths)

        # Goal statistics
        total_goals_scored = sum(self.episode_goals_scored) if self.episode_goals_scored else 0
        total_goals_conceded = sum(self.episode_goals_conceded) if self.episode_goals_conceded else 0

        # Win rate (goals scored - goals conceded)
        goal_difference = total_goals_scored - total_goals_conceded

        # Log to TensorBoard
        self.logger.record("metrics/mean_reward", mean_reward)
        self.logger.record("metrics/mean_episode_length", mean_episode_length)
        self.logger.record("metrics/total_goals_scored", total_goals_scored)
        self.logger.record("metrics/total_goals_conceded", total_goals_conceded)
        self.logger.record("metrics/goal_difference", goal_difference)
        self.logger.record("metrics/episodes_completed", len(self.episode_rewards))

        if self.verbose >= 1:
            print(f"[{self.num_timesteps} steps] Reward: {mean_reward:.2f}, "
                  f"Goals: +{total_goals_scored}/-{total_goals_conceded}, "
                  f"Ep Length: {mean_episode_length:.0f}")

        # Clear episode buffers
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_goals_scored.clear()
        self.episode_goals_conceded.clear()


class BestModelCallback(BaseCallback):
    """Callback for saving best model based on performance metric.

    Tracks performance metric (e.g., win rate, mean reward) and saves
    model when new best is achieved.

    Attributes:
        save_path: Path to save best model.
        metric_name: Name of metric to track.
        best_metric: Best metric value seen so far.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_path: str | Path,
        metric_name: str = "mean_reward",
        verbose: int = 1,
    ) -> None:
        """Initialize best model callback.

        Args:
            save_path: Path to save best model checkpoint.
            metric_name: Metric to track for best model.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.best_metric = -np.inf

        # Episode tracking
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Called at each training step.

        Returns:
            True to continue training.
        """
        # Track episode rewards
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)

        # Check for improvement periodically
        if len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards)

            if mean_reward > self.best_metric:
                self.best_metric = mean_reward

                # Save best model
                self.model.save(self.save_path)

                if self.verbose >= 1:
                    print(f"\n[{self.num_timesteps} steps] New best model! "
                          f"Mean reward: {mean_reward:.2f}")

            # Clear buffer
            self.episode_rewards.clear()

        return True


class ProgressBarCallback(BaseCallback):
    """Callback for displaying training progress.

    Shows progress bar with key metrics during training.

    Attributes:
        total_timesteps: Total training timesteps.
        pbar: Progress bar (tqdm).
    """

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        """Initialize progress bar callback.

        Args:
            total_timesteps: Total training timesteps.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        """Called when training starts."""
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_timesteps, desc="Training")
        except ImportError:
            self.pbar = None

    def _on_step(self) -> bool:
        """Update progress bar.

        Returns:
            True to continue training.
        """
        if self.pbar is not None:
            self.pbar.update(1)

        return True

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.pbar is not None:
            self.pbar.close()


class EarlyStoppingCallback(BaseCallback):
    """Callback for early stopping based on lack of improvement.

    Stops training if metric doesn't improve for specified number of checks.

    Attributes:
        patience: Number of checks without improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        check_freq: Frequency (in steps) to check for improvement.
        best_metric: Best metric value seen.
        wait_count: Number of checks without improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        check_freq: int = 10_000,
        verbose: int = 1,
    ) -> None:
        """Initialize early stopping callback.

        Args:
            patience: Number of checks without improvement before stopping.
            min_delta: Minimum improvement to reset patience counter.
            check_freq: Frequency to check for improvement (in steps).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq

        self.best_metric = -np.inf
        self.wait_count = 0

        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Check for improvement.

        Returns:
            True to continue, False to stop training.
        """
        # Collect episode rewards
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)

        # Check periodically
        if self.num_timesteps % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)

            if mean_reward > self.best_metric + self.min_delta:
                # Improvement found
                self.best_metric = mean_reward
                self.wait_count = 0
            else:
                # No improvement
                self.wait_count += 1

                if self.wait_count >= self.patience:
                    if self.verbose >= 1:
                        print(f"\n[{self.num_timesteps} steps] Early stopping triggered. "
                              f"No improvement for {self.patience} checks.")
                    return False  # Stop training

            self.episode_rewards.clear()

        return True
