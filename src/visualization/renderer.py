"""3D visualization and rendering for soccer environment.

Provides utilities for visualizing trained agents in 3D using PyBullet's GUI.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from src.environments.soccer_env import SoccerEnv


class SoccerRenderer:
    """3D renderer for soccer matches using PyBullet GUI.

    Attributes:
        env: Soccer environment.
        model: Trained model to visualize.
        fps: Target frames per second.
    """

    def __init__(
        self,
        model: PPO,
        fps: int = 60,
        env_config: dict | None = None,
    ) -> None:
        """Initialize soccer renderer.

        Args:
            model: Trained PPO model.
            fps: Target frames per second for rendering.
            env_config: Environment configuration.
        """
        self.model = model
        self.fps = fps
        self.frame_time = 1.0 / fps

        # Create environment with rendering enabled
        env_config = env_config or {}
        self.env = SoccerEnv(
            render_mode="human",
            **env_config
        )

    def render_episode(
        self,
        max_steps: int | None = None,
        deterministic: bool = True,
    ) -> dict:
        """Render a single episode.

        Args:
            max_steps: Maximum steps (None for full episode).
            deterministic: Use deterministic policy.

        Returns:
            Episode statistics dictionary.
        """
        obs, info = self.env.reset()

        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1

            # Frame rate limiting
            time.sleep(self.frame_time)

            if max_steps is not None and steps >= max_steps:
                break

        return {
            "episode_reward": episode_reward,
            "episode_length": steps,
            "blue_goals": info.get("blue_goals", 0),
            "red_goals": info.get("red_goals", 0),
        }

    def render_multiple_episodes(
        self,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> list[dict]:
        """Render multiple episodes.

        Args:
            n_episodes: Number of episodes to render.
            deterministic: Use deterministic policy.

        Returns:
            List of episode statistics.
        """
        episode_stats = []

        for episode in range(n_episodes):
            print(f"\nRendering episode {episode + 1}/{n_episodes}")
            stats = self.render_episode(deterministic=deterministic)

            print(f"Reward: {stats['episode_reward']:.2f}, "
                  f"Length: {stats['episode_length']}, "
                  f"Goals: {stats['blue_goals']}-{stats['red_goals']}")

            episode_stats.append(stats)

        return episode_stats

    def close(self) -> None:
        """Close environment and cleanup."""
        if self.env:
            self.env.close()
