"""Self-play opponent management for training competitive agents.

This module implements self-play training where agents train against previous
versions of themselves, promoting robust and diverse strategies.
"""

import os
import random
from pathlib import Path
from typing import List

import numpy as np
from stable_baselines3 import PPO


class OpponentPool:
    """Pool of opponent policies for self-play training.

    Maintains a collection of past agent checkpoints to use as opponents
    during training. Supports various opponent selection strategies.

    Attributes:
        pool_size: Maximum number of opponents to keep in pool.
        checkpoint_dir: Directory containing opponent checkpoints.
        opponents: List of opponent model paths.
        selection_strategy: Strategy for selecting opponents ('latest', 'random', 'uniform').
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        pool_size: int = 10,
        selection_strategy: str = "uniform",
    ) -> None:
        """Initialize opponent pool.

        Args:
            checkpoint_dir: Directory to store/load opponent checkpoints.
            pool_size: Maximum number of opponents in pool.
            selection_strategy: How to select opponents ('latest', 'random', 'uniform').
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.pool_size = pool_size
        self.selection_strategy = selection_strategy
        self.opponents: List[Path] = []

        self._load_existing_opponents()

    def _load_existing_opponents(self) -> None:
        """Load existing opponent checkpoints from disk."""
        if not self.checkpoint_dir.exists():
            return

        # Find all opponent checkpoint files
        opponent_files = sorted(
            self.checkpoint_dir.glob("opponent_*.zip"),
            key=lambda p: p.stat().st_mtime
        )

        # Keep only most recent ones up to pool_size
        self.opponents = list(opponent_files[-self.pool_size:])

    def add_opponent(self, model: PPO, timesteps: int) -> None:
        """Add new opponent to pool.

        Args:
            model: Trained model to add as opponent.
            timesteps: Training timesteps when this opponent was saved.
        """
        # Create opponent checkpoint filename
        opponent_path = self.checkpoint_dir / f"opponent_{timesteps}.zip"

        # Save model
        model.save(opponent_path)

        # Add to pool
        self.opponents.append(opponent_path)

        # Remove oldest if pool is full
        if len(self.opponents) > self.pool_size:
            oldest = self.opponents.pop(0)
            if oldest.exists():
                oldest.unlink()  # Delete file

    def select_opponent(self) -> Path | None:
        """Select opponent from pool based on strategy.

        Returns:
            Path to selected opponent checkpoint, or None if pool is empty.
        """
        if not self.opponents:
            return None

        if self.selection_strategy == "latest":
            # Always use most recent opponent
            return self.opponents[-1]

        elif self.selection_strategy == "random":
            # Uniformly random selection
            return random.choice(self.opponents)

        elif self.selection_strategy == "uniform":
            # Weighted selection favoring recent opponents
            weights = np.arange(1, len(self.opponents) + 1)
            weights = weights / weights.sum()
            idx = np.random.choice(len(self.opponents), p=weights)
            return self.opponents[idx]

        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def load_opponent(self, opponent_path: Path) -> PPO:
        """Load opponent model from checkpoint.

        Args:
            opponent_path: Path to opponent checkpoint.

        Returns:
            Loaded PPO model.
        """
        return PPO.load(opponent_path)

    def get_pool_size(self) -> int:
        """Get current number of opponents in pool.

        Returns:
            Number of opponents.
        """
        return len(self.opponents)

    def clear_pool(self) -> None:
        """Clear all opponents from pool and delete checkpoints."""
        for opponent_path in self.opponents:
            if opponent_path.exists():
                opponent_path.unlink()

        self.opponents.clear()


class SelfPlayCallback:
    """Callback for managing self-play during training.

    Periodically adds current agent to opponent pool and updates
    the opponent used in the environment.

    Attributes:
        opponent_pool: Pool of opponent policies.
        save_freq: Frequency (in timesteps) to save new opponents.
        opponent_update_freq: Frequency to update current opponent.
    """

    def __init__(
        self,
        opponent_pool: OpponentPool,
        save_freq: int = 100_000,
        opponent_update_freq: int = 10_000,
    ) -> None:
        """Initialize self-play callback.

        Args:
            opponent_pool: Opponent pool manager.
            save_freq: How often to save current model as opponent (timesteps).
            opponent_update_freq: How often to switch opponent (timesteps).
        """
        self.opponent_pool = opponent_pool
        self.save_freq = save_freq
        self.opponent_update_freq = opponent_update_freq

        self.last_save_timestep = 0
        self.last_update_timestep = 0
        self.current_opponent: PPO | None = None

    def on_step(self, model: PPO, timesteps: int) -> None:
        """Called at each training step.

        Args:
            model: Current training model.
            timesteps: Total training timesteps so far.
        """
        # Save current model as opponent
        if timesteps - self.last_save_timestep >= self.save_freq:
            self.opponent_pool.add_opponent(model, timesteps)
            self.last_save_timestep = timesteps

        # Update current opponent
        if timesteps - self.last_update_timestep >= self.opponent_update_freq:
            self._update_opponent()
            self.last_update_timestep = timesteps

    def _update_opponent(self) -> None:
        """Update current opponent from pool."""
        opponent_path = self.opponent_pool.select_opponent()

        if opponent_path is not None:
            self.current_opponent = self.opponent_pool.load_opponent(opponent_path)

    def get_opponent_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from current opponent.

        Args:
            observation: Environment observation.

        Returns:
            Action from opponent policy, or random action if no opponent.
        """
        if self.current_opponent is None:
            # No opponent yet, return random action
            return np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)

        action, _ = self.current_opponent.predict(observation, deterministic=False)
        return action


class SelfPlayManager:
    """High-level manager for self-play training.

    Coordinates opponent pool, callback, and training process.

    Attributes:
        opponent_pool: Pool of opponent policies.
        callback: Self-play callback for training.
        use_self_play: Whether self-play is enabled.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        pool_size: int = 10,
        save_freq: int = 100_000,
        opponent_update_freq: int = 10_000,
        selection_strategy: str = "uniform",
    ) -> None:
        """Initialize self-play manager.

        Args:
            checkpoint_dir: Directory for opponent checkpoints.
            pool_size: Maximum opponents in pool.
            save_freq: Frequency to save new opponents (timesteps).
            opponent_update_freq: Frequency to switch opponents (timesteps).
            selection_strategy: Opponent selection strategy.
        """
        self.opponent_pool = OpponentPool(
            checkpoint_dir=checkpoint_dir,
            pool_size=pool_size,
            selection_strategy=selection_strategy,
        )

        self.callback = SelfPlayCallback(
            opponent_pool=self.opponent_pool,
            save_freq=save_freq,
            opponent_update_freq=opponent_update_freq,
        )

        self.use_self_play = True

    def on_training_step(self, model: PPO, timesteps: int) -> None:
        """Called during training to update self-play state.

        Args:
            model: Current training model.
            timesteps: Total training timesteps.
        """
        if self.use_self_play:
            self.callback.on_step(model, timesteps)

    def get_opponent_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from opponent for current observation.

        Args:
            observation: Environment observation.

        Returns:
            Opponent action.
        """
        if self.use_self_play:
            return self.callback.get_opponent_action(observation)
        else:
            # Return random action if self-play disabled
            return np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)

    def get_stats(self) -> dict:
        """Get self-play statistics.

        Returns:
            Dictionary with self-play stats.
        """
        return {
            "pool_size": self.opponent_pool.get_pool_size(),
            "use_self_play": self.use_self_play,
            "selection_strategy": self.opponent_pool.selection_strategy,
        }
