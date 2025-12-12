"""Main training orchestrator for soccer RL agents."""

import logging
from pathlib import Path
from typing import Any, Dict

# CRITICAL: Apply SB3 fix BEFORE importing PPO
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import src.training.sb3_fix
    print("[TRAINER] SB3 fix loaded successfully")
except ImportError as e:
    print(f"[TRAINER] Warning: Could not load SB3 fix: {e}")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.environments.soccer_env import SoccerEnv
from src.training.callbacks import (
    BestModelCallback,
    EarlyStoppingCallback,
    ProgressBarCallback,
    SoccerMetricsCallback,
)
from src.training.self_play import SelfPlayManager


logger = logging.getLogger(__name__)


class SoccerTrainer:
    """Training orchestrator for soccer RL agents."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str | Path = "outputs",
    ) -> None:
        """Initialize soccer trainer."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.env = None
        self.model = None
        self.self_play_manager = None

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

    def create_env(self, n_envs: int = 8, use_subprocess: bool = False):
        """Create vectorized training environment."""
        env_config = self.config.get("env", {})

        def make_env():
            def _init():
                return SoccerEnv(
                    render_mode=None,
                    max_episode_steps=env_config.get("max_episode_steps", 1000),
                    time_step=env_config.get("time_step", 0.01),
                    reward_goal_scored=env_config.get("reward_goal_scored", 1000.0),
                    reward_goal_conceded=env_config.get("reward_goal_conceded", -1000.0),
                    reward_own_goal=env_config.get("reward_own_goal", -2000.0),
                    reward_ball_touch=env_config.get("reward_ball_touch", 5.0),
                    reward_ball_to_goal=env_config.get("reward_ball_to_goal", 10.0),
                    reward_no_action=env_config.get("reward_no_action", -1.0),
                )
            return _init

        if use_subprocess and n_envs > 1:
            env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        else:
            env = DummyVecEnv([make_env() for _ in range(n_envs)])

        logger.info(f"Created {n_envs} parallel environments")
        return env

    def create_model(self, env):
        """Create PPO model."""
        model_config = self.config.get("model", {})

        device = model_config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {device}")

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=model_config.get("learning_rate", 3e-4),
            n_steps=model_config.get("n_steps", 512),
            batch_size=model_config.get("batch_size", 128),
            n_epochs=model_config.get("n_epochs", 10),
            gamma=model_config.get("gamma", 0.99),
            gae_lambda=model_config.get("gae_lambda", 0.95),
            clip_range=model_config.get("clip_range", 0.2),
            ent_coef=model_config.get("ent_coef", 0.1),
            vf_coef=model_config.get("vf_coef", 0.5),
            max_grad_norm=model_config.get("max_grad_norm", 0.5),
            tensorboard_log=str(self.output_dir / "tensorboard"),
            device=device,
            verbose=1,
        )

        logger.info("Created PPO model")
        return model

    def create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        training_config = self.config.get("training", {})

        callbacks = [
            SoccerMetricsCallback(
                log_freq=training_config.get("log_freq", 2048),
                verbose=1
            ),
            BestModelCallback(
                save_path=self.output_dir / "checkpoints" / "best_model.zip",
                verbose=1
            ),
            ProgressBarCallback(
                total_timesteps=training_config.get("total_timesteps", 2_000_000),
                n_envs=training_config.get("n_envs", 8),
                verbose=0
            ),
        ]

        if training_config.get("use_early_stopping", False):
            callbacks.append(
                EarlyStoppingCallback(
                    patience=training_config.get("early_stopping_patience", 10),
                    check_freq=training_config.get("early_stopping_check_freq", 10_000),
                    verbose=1
                )
            )

        return CallbackList(callbacks)

    def setup_self_play(self):
        """Setup self-play opponent management."""
        self_play_config = self.config.get("self_play", {})

        if not self_play_config.get("enabled", False):
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
        return manager

    def train(self) -> PPO:
        """Execute training."""
        training_config = self.config.get("training", {})

        self.env = self.create_env(
            n_envs=training_config.get("n_envs", 8),
            use_subprocess=training_config.get("use_subprocess", False),
        )

        self.model = self.create_model(self.env)

        self.self_play_manager = self.setup_self_play()

        callbacks = self.create_callbacks()

        total_timesteps = training_config.get("total_timesteps", 2_000_000)
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
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

        finally:
            final_path = self.output_dir / "checkpoints" / "final_policy.pth"
            final_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                torch.save({
                    'policy_state_dict': self.model.policy.state_dict(),
                    'timesteps': self.model.num_timesteps,
                }, final_path)
                logger.info(f"Saved final model to {final_path}")
            except Exception as e:
                logger.error(f"Failed to save final model: {e}")
            finally:
                if self.env:
                    try:
                        self.env.close()
                    except:
                        pass

        return self.model

    def load_checkpoint(self, checkpoint_path: str | Path) -> PPO:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if self.env is None:
            self.env = self.create_env(n_envs=1)

        self.model = PPO.load(checkpoint_path, env=self.env)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return self.model

    def save_checkpoint(self, checkpoint_name: str = "checkpoint") -> Path:
        """Save current model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save")

        checkpoint_path = self.output_dir / "checkpoints" / f"{checkpoint_name}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save({
                'policy_state_dict': self.model.policy.state_dict(),
                'timesteps': self.model.num_timesteps,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

        return checkpoint_path