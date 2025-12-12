"""Configuration management with Hydra integration.

This module provides utilities for loading, validating, and managing
training configurations using Hydra and OmegaConf.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Environment configuration.

    Attributes:
        max_episode_steps: Maximum steps per episode.
        time_step: Physics simulation timestep.
        mode: Environment variant ("3d" or "2d").
        reward_goal_scored: Reward for scoring.
        reward_goal_conceded: Penalty for conceding.
        reward_ball_proximity_scale: Ball proximity reward scale.
        reward_time_penalty: Time penalty per step.
        reward_ball_progress_scale: Reward scale for moving ball toward opponent goal.
        reward_possession_bonus: Bonus for maintaining close ball possession.
        possession_distance_threshold: Distance threshold for possession bonus.
    """

    max_episode_steps: int = 2000
    time_step: float = 1.0 / 240.0
    mode: str = "3d"
    reward_goal_scored: float = 10.0
    reward_goal_conceded: float = -10.0
    reward_ball_proximity_scale: float = 0.01
    reward_time_penalty: float = -0.001
    reward_ball_progress_scale: float = 0.05
    reward_possession_bonus: float = 0.05
    possession_distance_threshold: float = 1.5


@dataclass
class ModelConfig:
    """Model configuration.

    Attributes:
        learning_rate: Learning rate for optimizer.
        n_steps: Steps per update.
        batch_size: Minibatch size.
        n_epochs: Epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        vf_coef: Value function coefficient.
        max_grad_norm: Maximum gradient norm.
        device: Device ('auto', 'cpu', 'cuda').
    """

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        total_timesteps: Total training timesteps.
        n_envs: Number of parallel environments.
        use_subprocess: Use subprocesses for parallel envs.
        log_freq: Logging frequency (steps).
        save_freq: Checkpoint save frequency (steps).
        use_early_stopping: Enable early stopping.
        early_stopping_patience: Early stopping patience.
        early_stopping_check_freq: Early stopping check frequency.
    """

    total_timesteps: int = 1_000_000
    n_envs: int = 4
    use_subprocess: bool = False
    log_freq: int = 1000
    save_freq: int = 10_000
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_check_freq: int = 10_000


@dataclass
class SelfPlayConfig:
    """Self-play configuration.

    Attributes:
        enabled: Enable self-play training.
        pool_size: Opponent pool size.
        save_freq: Frequency to save opponents (steps).
        update_freq: Frequency to update opponent (steps).
        selection_strategy: Strategy for selecting opponents.
    """

    enabled: bool = True
    pool_size: int = 10
    save_freq: int = 100_000
    update_freq: int = 10_000
    selection_strategy: str = "uniform"


@dataclass
class Config:
    """Complete training configuration.

    Attributes:
        env: Environment configuration.
        model: Model configuration.
        training: Training configuration.
        self_play: Self-play configuration.
        seed: Random seed.
        output_dir: Output directory.
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    seed: int = 42
    output_dir: str = "outputs"


class ConfigManager:
    """Manager for loading and validating configurations.

    Provides utilities for:
    - Loading YAML configs
    - Merging configs with defaults
    - Validation
    - Converting to dictionaries

    Attributes:
        config: Current configuration object.
    """

    def __init__(self, config: Config | DictConfig | None = None) -> None:
        """Initialize config manager.

        Args:
            config: Configuration object or None for defaults.
        """
        if config is None:
            self.config = Config()
        elif isinstance(config, DictConfig):
            self.config = OmegaConf.to_object(config)
        else:
            self.config = config

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ConfigManager":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            ConfigManager instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert to OmegaConf
        omega_conf = OmegaConf.create(config_dict)

        # Merge with defaults
        default_conf = OmegaConf.structured(Config)
        merged = OmegaConf.merge(default_conf, omega_conf)

        # Validate
        OmegaConf.to_object(merged)

        logger.info(f"Loaded configuration from {config_path}")

        return cls(merged)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigManager":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            ConfigManager instance.
        """
        omega_conf = OmegaConf.create(config_dict)
        default_conf = OmegaConf.structured(Config)
        merged = OmegaConf.merge(default_conf, omega_conf)

        return cls(merged)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary.
        """
        return OmegaConf.to_container(
            OmegaConf.structured(self.config),
            resolve=True
        )

    def save(self, save_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            save_path: Path to save configuration.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {save_path}")

    def validate(self) -> bool:
        """Validate configuration values.

        Returns:
            True if valid, False otherwise.

        Raises:
            ValueError: If configuration is invalid.
        """
        config_dict = self.to_dict()

        # Validate environment config
        if config_dict["env"]["max_episode_steps"] <= 0:
            raise ValueError("max_episode_steps must be positive")

        # Validate model config
        if config_dict["model"]["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")

        if config_dict["model"]["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")

        # Validate training config
        if config_dict["training"]["total_timesteps"] <= 0:
            raise ValueError("total_timesteps must be positive")

        if config_dict["training"]["n_envs"] <= 0:
            raise ValueError("n_envs must be positive")

        # Validate self-play config
        if config_dict["self_play"]["pool_size"] <= 0:
            raise ValueError("pool_size must be positive")

        logger.info("Configuration validation passed")

        return True

    def get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration as dictionary.

        Returns:
            Environment config dict.
        """
        return OmegaConf.to_container(
            OmegaConf.structured(self.config.env),
            resolve=True
        )

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary.

        Returns:
            Model config dict.
        """
        return OmegaConf.to_container(
            OmegaConf.structured(self.config.model),
            resolve=True
        )

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary.

        Returns:
            Training config dict.
        """
        return OmegaConf.to_container(
            OmegaConf.structured(self.config.training),
            resolve=True
        )

    def get_self_play_config(self) -> Dict[str, Any]:
        """Get self-play configuration as dictionary.

        Returns:
            Self-play config dict.
        """
        return OmegaConf.to_container(
            OmegaConf.structured(self.config.self_play),
            resolve=True
        )
