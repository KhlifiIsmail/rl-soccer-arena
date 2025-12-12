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
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class SoccerMetricsCallback(BaseCallback):
    """Callback for logging soccer-specific metrics to TensorBoard."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_goals_scored: list[int] = []
        self.episode_goals_conceded: list[int] = []
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []
        self.current_goals_scored = 0
        self.current_goals_conceded = 0

    def _on_step(self) -> bool:
        """Called at each training step. Returns True to continue training."""
        infos = self.locals.get("infos", [])

        for info in infos:
            if not isinstance(info, dict):
                continue
                
            if "episode" in info:
                ep_info = info["episode"]
                if isinstance(ep_info, dict):
                    if "r" in ep_info:
                        self.episode_rewards.append(ep_info["r"])
                    if "l" in ep_info:
                        self.episode_lengths.append(ep_info["l"])

            if info.get("goal_scored", False):
                self.current_goals_scored += 1

            if info.get("goal_conceded", False):
                self.current_goals_conceded += 1

        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self._log_metrics()

        return True  # ALWAYS return True to continue

    def _log_metrics(self) -> None:
        if len(self.episode_rewards) == 0:
            return

        mean_reward = np.mean(self.episode_rewards)
        mean_episode_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        total_goals_scored = sum(self.episode_goals_scored) if self.episode_goals_scored else 0
        total_goals_conceded = sum(self.episode_goals_conceded) if self.episode_goals_conceded else 0
        goal_difference = total_goals_scored - total_goals_conceded

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

        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_goals_scored.clear()
        self.episode_goals_conceded.clear()


class BestModelCallback(BaseCallback):
    """Callback for saving best model based on performance metric."""

    def __init__(
        self,
        save_path: str | Path,
        metric_name: str = "mean_reward",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.best_metric = -np.inf
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Called at each training step. Returns True to continue training."""
        infos = self.locals.get("infos", [])

        for info in infos:
            if not isinstance(info, dict):
                continue
                
            if "episode" in info:
                ep_info = info["episode"]
                if isinstance(ep_info, dict) and "r" in ep_info:
                    ep_reward = ep_info["r"]
                    self.episode_rewards.append(ep_reward)

        if len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards)

            if mean_reward > self.best_metric:
                self.best_metric = mean_reward

                try:
                    policy_path = self.save_path.parent / f"best_policy_{self.num_timesteps}.pth"
                    torch.save({
                        'policy_state_dict': self.model.policy.state_dict(),
                        'timesteps': self.num_timesteps,
                        'mean_reward': mean_reward
                    }, policy_path)
                    
                    if self.verbose >= 1:
                        print(f"\n[{self.num_timesteps} steps] New best model saved! "
                              f"Mean reward: {mean_reward:.2f}")
                        print(f"Saved to: {policy_path}")
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"Warning: Could not save model: {e}")

            self.episode_rewards.clear()

        return True  # ALWAYS return True to continue


class ProgressBarCallback(BaseCallback):
    """Callback for displaying training progress."""

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_timesteps, desc="Training")
        except ImportError:
            self.pbar = None

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True  # ALWAYS return True to continue

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


class EarlyStoppingCallback(BaseCallback):
    """Callback for early stopping based on lack of improvement."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        check_freq: int = 10_000,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.best_metric = -np.inf
        self.wait_count = 0
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        """Check for improvement. Returns False ONLY if early stopping triggered."""
        infos = self.locals.get("infos", [])

        for info in infos:
            if not isinstance(info, dict):
                continue
                
            if "episode" in info:
                ep_info = info["episode"]
                if isinstance(ep_info, dict) and "r" in ep_info:
                    ep_reward = ep_info["r"]
                    self.episode_rewards.append(ep_reward)

        if self.num_timesteps % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)

            if mean_reward > self.best_metric + self.min_delta:
                self.best_metric = mean_reward
                self.wait_count = 0
            else:
                self.wait_count += 1

                if self.wait_count >= self.patience:
                    if self.verbose >= 1:
                        print(f"\n[{self.num_timesteps} steps] Early stopping triggered. "
                              f"No improvement for {self.patience} checks.")
                    return False  # Stop training

            self.episode_rewards.clear()

        return True  # Continue training