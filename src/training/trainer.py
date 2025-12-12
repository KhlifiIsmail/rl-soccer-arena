"""Main training orchestrator for soccer RL agents.

This module coordinates the complete training process including:
- Environment setup
- Model initialization
- Self-play management
- Training loops
- Checkpointing and logging
"""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from src.training.vec_wrappers import CleanEpisodeInfoVecWrapper
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.environments.soccer_env import SoccerEnv
from src.environments.soccer_env_2d import SoccerEnv2D
from src.training.callbacks import (
    BestModelCallback,
    EarlyStoppingCallback,
    ProgressBarCallback,
    SoccerMetricsCallback,
)
from src.training.self_play import SelfPlayManager


logger = logging.getLogger(__name__)


class SoccerTrainer:
    """Training orchestrator for soccer RL agents.

    Manages complete training pipeline:
    - Environment creation and vectorization
    - PPO agent initialization
    - Self-play opponent management
    - Training execution with callbacks
    - Model checkpointing

    Attributes:
        config: Training configuration dictionary.
        env: Vectorized training environment.
        model: PPO model being trained.
        self_play_manager: Self-play opponent manager.
        output_dir: Directory for outputs (models, logs).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str | Path = "outputs",
    ) -> None:
        """Initialize soccer trainer.

        Args:
            config: Training configuration dictionary.
            output_dir: Directory for saving outputs.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.env = None
        self.model = None
        self.self_play_manager = None

        # Setup logging
        self._setup_logging()

        logger.info("Initialized SoccerTrainer")
        logger.info(f"Output directory: {self.output_dir}")

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_file = self.output_dir / "training.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    def create_env(self, n_envs: int = 4, use_subprocess: bool = False) -> DummyVecEnv | SubprocVecEnv:
        """Create vectorized training environment.

        Args:
            n_envs: Number of parallel environments.
            use_subprocess: Use subprocesses (True) or single process (False).

        Returns:
            Vectorized environment.
        """
        env_config = self.config.get("env", {})
        mode = env_config.get("mode", "3d").lower()

        def make_env():
            def _init():
                if mode == "2d":
                    return SoccerEnv2D(
                        render_mode=None,
                        max_episode_steps=env_config.get("max_episode_steps", 2000),
                        time_step=env_config.get("time_step", 1.0 / 60.0),
                        reward_goal_scored=env_config.get("reward_goal_scored", 10.0),
                        reward_goal_conceded=env_config.get("reward_goal_conceded", -10.0),
                        reward_ball_proximity_scale=env_config.get("reward_ball_proximity_scale", 0.01),
                        reward_time_penalty=env_config.get("reward_time_penalty", -0.001),
                        reward_ball_progress_scale=env_config.get("reward_ball_progress_scale", 0.05),
                        reward_possession_bonus=env_config.get("reward_possession_bonus", 0.05),
                        possession_distance_threshold=env_config.get("possession_distance_threshold", 1.5),
                    )
                # Default: 3D PyBullet env
                return SoccerEnv(
                    render_mode=None,  # No rendering during training
                    max_episode_steps=env_config.get("max_episode_steps", 2000),
                    time_step=env_config.get("time_step", 1.0 / 240.0),
                    reward_goal_scored=env_config.get("reward_goal_scored", 10.0),
                    reward_goal_conceded=env_config.get("reward_goal_conceded", -10.0),
                    reward_ball_proximity_scale=env_config.get("reward_ball_proximity_scale", 0.01),
                    reward_time_penalty=env_config.get("reward_time_penalty", -0.001),
                    reward_ball_progress_scale=env_config.get("reward_ball_progress_scale", 0.05),
                    reward_possession_bonus=env_config.get("reward_possession_bonus", 0.05),
                    possession_distance_threshold=env_config.get("possession_distance_threshold", 1.5),
                )
            return _init

        if use_subprocess and n_envs > 1:
            env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        else:
            env = DummyVecEnv([make_env() for _ in range(n_envs)])

        logger.info(f"Created {n_envs} parallel environments")

        return env

    def create_model(self, env: DummyVecEnv | SubprocVecEnv) -> PPO:
        """Create PPO model.

        Args:
            env: Training environment.

        Returns:
            PPO model.
        """
        model_config = self.config.get("model", {})

        # Device selection
        device = model_config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {device}")

        # Create PPO model
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=model_config.get("learning_rate", 3e-4),
            n_steps=model_config.get("n_steps", 2048),
            batch_size=model_config.get("batch_size", 64),
            n_epochs=model_config.get("n_epochs", 10),
            gamma=model_config.get("gamma", 0.99),
            gae_lambda=model_config.get("gae_lambda", 0.95),
            clip_range=model_config.get("clip_range", 0.2),
            ent_coef=model_config.get("ent_coef", 0.01),
            vf_coef=model_config.get("vf_coef", 0.5),
            max_grad_norm=model_config.get("max_grad_norm", 0.5),
            tensorboard_log=str(self.output_dir / "tensorboard"),
            device=device,
            verbose=1,
        )

        logger.info("Created PPO model")
        logger.info(f"Learning rate: {model_config.get('learning_rate', 3e-4)}")
        logger.info(f"Batch size: {model_config.get('batch_size', 64)}")

        return model

    def create_callbacks(self) -> CallbackList:
        """Create training callbacks.

        Returns:
            Callback list for training.
        """
        training_config = self.config.get("training", {})

        callbacks = [
            # Metrics logging
            SoccerMetricsCallback(
                log_freq=training_config.get("log_freq", 1000),
                verbose=1
            ),

            # Best model saving
            BestModelCallback(
                save_path=self.output_dir / "checkpoints" / "best_model.zip",
                verbose=1
            ),

            # Progress bar
            ProgressBarCallback(
                total_timesteps=training_config.get("total_timesteps", 1_000_000),
                verbose=0
            ),
        ]

        # Optional early stopping
        if training_config.get("use_early_stopping", False):
            callbacks.append(
                EarlyStoppingCallback(
                    patience=training_config.get("early_stopping_patience", 10),
                    check_freq=training_config.get("early_stopping_check_freq", 10_000),
                    verbose=1
                )
            )

        return CallbackList(callbacks)

    def setup_self_play(self) -> SelfPlayManager:
        """Setup self-play opponent management.

        Returns:
            Self-play manager.
        """
        self_play_config = self.config.get("self_play", {})

        if not self_play_config.get("enabled", True):
            logger.info("Self-play disabled")
            return None

        manager = SelfPlayManager(
            checkpoint_dir=self.output_dir / "opponents",
            pool_size=self_play_config.get("pool_size", 10),
            save_freq=self_play_config.get("save_freq", 100_000),
            opponent_update_freq=self_play_config.get("update_freq", 10_000),
            selection_strategy=self_play_config.get("selection_strategy", "uniform"),
        )

        logger.info("Self-play enabled")
        logger.info(f"Pool size: {self_play_config.get('pool_size', 10)}")
        logger.info(f"Save frequency: {self_play_config.get('save_freq', 100_000)} steps")

        return manager

    def train(self) -> PPO:
        """Execute training.

        Returns:
            Trained PPO model.
        """
        training_config = self.config.get("training", {})

        # Create environment
        self.env = self.create_env(
            n_envs=training_config.get("n_envs", 4),
            use_subprocess=training_config.get("use_subprocess", False),
        )
        self.env = CleanEpisodeInfoVecWrapper(self.env, rename_to="episode_scalar")
        self.env = VecMonitor(self.env)
        # Create model
        self.model = self.create_model(self.env)

        # Setup self-play
        self.self_play_manager = self.setup_self_play()

        # Create callbacks
        callbacks = self.create_callbacks()

        # Training loop
        total_timesteps = training_config.get("total_timesteps", 1_000_000)
        logger.info(f"Starting training for {total_timesteps} timesteps")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name="ppo_soccer",
                reset_num_timesteps=False,
            )

            logger.info("Training completed successfully")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            raise

        finally:
            # Save final model
            final_model_path = self.output_dir / "checkpoints" / "final_model.zip"
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(final_model_path)
            logger.info(f"Saved final model to {final_model_path}")

            # Cleanup
            if self.env:
                self.env.close()

        return self.model

    def load_checkpoint(self, checkpoint_path: str | Path) -> PPO:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Loaded PPO model.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create environment for model
        if self.env is None:
            self.env = self.create_env(n_envs=1)

        # Load model
        self.model = PPO.load(checkpoint_path, env=self.env)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return self.model

    def save_checkpoint(self, checkpoint_name: str = "checkpoint") -> Path:
        """Save current model checkpoint.

        Args:
            checkpoint_name: Name for checkpoint file.

        Returns:
            Path to saved checkpoint.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        checkpoint_path = self.output_dir / "checkpoints" / f"{checkpoint_name}.zip"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path
