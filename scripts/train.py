"""Training entry point for RL Soccer Arena.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --output outputs/exp1
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import SoccerTrainer
from src.utils.config import ConfigManager
from src.utils.logger import log_system_info, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train RL soccer agents with self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration YAML file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config_manager = ConfigManager.from_yaml(args.config)
    config_dict = config_manager.to_dict()

    # Override config with command line arguments
    if args.output:
        config_dict["output_dir"] = args.output

    if args.seed is not None:
        config_dict["seed"] = args.seed

    if args.timesteps is not None:
        config_dict["training"]["total_timesteps"] = args.timesteps

    # Setup output directory
    output_dir = Path(config_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger(
        name="rl_soccer_arena",
        log_file=output_dir / "training.log",
        level="INFO",
    )

    logger.info("="  * 60)
    logger.info("RL SOCCER ARENA - TRAINING")
    logger.info("=" * 60)

    # Log system information
    log_system_info(logger)

    # Save configuration
    config_save_path = output_dir / "config.yaml"
    config_manager.save(config_save_path)
    logger.info(f"Saved configuration to: {config_save_path}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = SoccerTrainer(
        config=config_dict,
        output_dir=output_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training
    try:
        logger.info("Starting training...")
        model = trainer.train()
        logger.info("Training completed successfully!")

        # Print final checkpoint location
        final_checkpoint = output_dir / "checkpoints" / "final_model.zip"
        logger.info(f"Final model saved to: {final_checkpoint}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
