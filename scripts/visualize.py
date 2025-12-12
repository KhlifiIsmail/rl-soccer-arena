"""Visualization entry point for RL Soccer Arena.

Usage:
    python scripts/visualize.py --checkpoint outputs/checkpoints/best_model.zip
    python scripts/visualize.py --checkpoint outputs/checkpoints/best_model.zip --episodes 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.renderer import SoccerRenderer
from src.utils.logger import setup_logger
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize trained RL soccer agents in 3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint to visualize",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target frames per second for rendering",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration)",
    )

    return parser.parse_args()


def main() -> None:
    """Main visualization function."""
    args = parse_args()

    # Setup logging
    logger = setup_logger(name="rl_soccer_arena", level="INFO")

    logger.info("=" * 60)
    logger.info("RL SOCCER ARENA - VISUALIZATION")
    logger.info("=" * 60)

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model: {checkpoint_path}")
    try:
        model = PPO.load(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Create renderer
    logger.info("Initializing 3D renderer...")
    renderer = SoccerRenderer(
        model=model,
        fps=args.fps,
    )

    # Render episodes
    try:
        logger.info(f"Rendering {args.episodes} episodes...")
        logger.info("Close the PyBullet window to stop visualization.")

        episode_stats = renderer.render_multiple_episodes(
            n_episodes=args.episodes,
            deterministic=args.deterministic,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("VISUALIZATION SUMMARY")
        print("=" * 60)

        total_reward = sum(ep["episode_reward"] for ep in episode_stats)
        avg_reward = total_reward / len(episode_stats)
        total_goals = sum(ep["blue_goals"] for ep in episode_stats)

        print(f"Episodes: {len(episode_stats)}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Total Goals Scored: {total_goals}")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        renderer.close()
        logger.info("Visualization complete")


if __name__ == "__main__":
    main()
