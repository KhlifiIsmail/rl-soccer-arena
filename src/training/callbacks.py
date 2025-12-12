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

    def __init__(self, log_freq: int = 2048, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_goals_scored: list[int] = []
        self.episode_goals_conceded: list[int] = []

    def _on_step(self) -> bool:
        """Called at each training step."""
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if not isinstance(info, dict):
                continue
            
            if "episode" in info:
                ep_info = info["episode"]
                if isinstance(ep_info, dict):
                    if "r" in ep_info:
                        self.episode_rewards.append(float(ep_info["r"]))
                    if "l" in ep_info:
                        self.episode_lengths.append(int(ep_info["l"]))
            
            if info.get("goal_scored", False):
                if len(self.episode_goals_scored) == 0 or len(self.episode_rewards) > len(self.episode_goals_scored):
                    self.episode_goals_scored.append(1)
                else:
                    self.episode_goals_scored[-1] += 1
            
            if info.get("goal_conceded", False):
                if len(self.episode_goals_conceded) == 0 or len(self.episode_rewards) > len(self.episode_goals_conceded):
                    self.episode_goals_conceded.append(1)
                else:
                    self.episode_goals_conceded[-1] += 1
        
        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self._log_metrics()
        
        return True

    def _log_metrics(self) -> None:
        """Log collected metrics - FORCE IT TO LOG."""
        if len(self.episode_rewards) == 0:
            return
        
        mean_reward = np.mean(self.episode_rewards)
        mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        total_goals_scored = sum(self.episode_goals_scored) if self.episode_goals_scored else 0
        total_goals_conceded = sum(self.episode_goals_conceded) if self.episode_goals_conceded else 0
        
        # FORCE LOGGING - bypass any filters
        try:
            self.logger.record("soccer/mean_reward", float(mean_reward))
            self.logger.record("soccer/mean_episode_length", float(mean_length))
            self.logger.record("soccer/goals_scored", int(total_goals_scored))
            self.logger.record("soccer/goals_conceded", int(total_goals_conceded))
            self.logger.record("soccer/goal_difference", int(total_goals_scored - total_goals_conceded))
            self.logger.record("soccer/episodes", len(self.episode_rewards))
            self.logger.dump(step=self.num_timesteps)
        except Exception as e:
            print(f"WARNING: Failed to log metrics: {e}")
        
        if self.verbose >= 1:
            print(f"\n[{self.num_timesteps}] Reward: {mean_reward:.2f} | "
                  f"Goals: {total_goals_scored}-{total_goals_conceded} | "
                  f"Ep Len: {mean_length:.0f}")
        
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_goals_scored.clear()
        self.episode_goals_conceded.clear()


class BestModelCallback(BaseCallback):
    """Save best model based on mean reward."""

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
        """Check if we should save a new best model."""
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if not isinstance(info, dict):
                continue
            
            if "episode" in info:
                ep_info = info["episode"]
                if isinstance(ep_info, dict) and "r" in ep_info:
                    self.episode_rewards.append(float(ep_info["r"]))
        
        if len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards)
            
            if mean_reward > self.best_metric:
                self.best_metric = mean_reward
                
                policy_path = self.save_path.parent / f"best_policy_{self.num_timesteps}.pth"
                
                try:
                    torch.save({
                        'policy_state_dict': self.model.policy.state_dict(),
                        'timesteps': self.num_timesteps,
                        'mean_reward': mean_reward
                    }, policy_path)
                    
                    if self.verbose >= 1:
                        print(f"\n{'='*60}")
                        print(f"NEW BEST MODEL at {self.num_timesteps} steps!")
                        print(f"Mean Reward: {mean_reward:.2f}")
                        print(f"Saved to: {policy_path}")
                        print(f"{'='*60}\n")
                        
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"Warning: Failed to save model: {e}")
            
            self.episode_rewards.clear()
        
        return True


class ProgressBarCallback(BaseCallback):
    """Display training progress bar - FIXED VERSION."""

    def __init__(self, total_timesteps: int, n_envs: int = 8, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.pbar = None

    def _on_training_start(self) -> None:
        """Initialize progress bar."""
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")
        except ImportError:
            if self.verbose >= 1:
                print("tqdm not installed, progress bar disabled")
            self.pbar = None

    def _on_step(self) -> bool:
        """Update progress bar - FIXED TO ACCOUNT FOR MULTIPLE ENVS."""
        if self.pbar is not None:
            self.pbar.update(self.n_envs)
        return True

    def _on_training_end(self) -> None:
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()


class EarlyStoppingCallback(BaseCallback):
    """Stop training if no improvement."""

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
        """Check for early stopping."""
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if not isinstance(info, dict):
                continue
            
            if "episode" in info:
                ep_info = info["episode"]
                if isinstance(ep_info, dict) and "r" in ep_info:
                    self.episode_rewards.append(float(ep_info["r"]))
        
        if self.num_timesteps % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            
            if mean_reward > self.best_metric + self.min_delta:
                self.best_metric = mean_reward
                self.wait_count = 0
            else:
                self.wait_count += 1
                
                if self.wait_count >= self.patience:
                    if self.verbose >= 1:
                        print(f"\n{'='*60}")
                        print(f"EARLY STOPPING at {self.num_timesteps} steps")
                        print(f"No improvement for {self.patience} checks")
                        print(f"{'='*60}\n")
                    return False
            
            self.episode_rewards.clear()
        
        return True