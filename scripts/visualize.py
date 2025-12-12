"""Visualization entry point for RL Soccer Arena.

Usage:
    python scripts/visualize.py --checkpoint outputs/checkpoints/best_policy.pth --episodes 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.renderer import SoccerRenderer
from src.utils.logger import setup_logger
from stable_baselines3 import PPO
import torch


def load_model_from_pth(pth_path: Path):
    """Load model from .pth policy weights file WITHOUT creating environment."""
    from src.environments.soccer_env import SoccerEnv
    
    # Create HEADLESS environment (no GUI)
    env = SoccerEnv(render_mode=None)
    
    # Create new model
    model = PPO("MlpPolicy", env, device="cpu")  # Use CPU to avoid issues
    
    # Load policy weights
    checkpoint = torch.load(pth_path, map_location="cpu")
    model.policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Close the headless environment immediately
    env.close()
    
    print(f"Loaded policy from {pth_path}")
    print(f"Training timesteps: {checkpoint.get('timesteps', 'unknown')}")
    
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize trained RL soccer agents in 3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.zip or .pth)",
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
        if checkpoint_path.suffix == ".pth":
            model = load_model_from_pth(checkpoint_path)
        else:
            model = PPO.load(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create renderer (this will create the GUI environment)
    logger.info("Initializing 3D renderer...")
    try:
        renderer = SoccerRenderer(
            model=model,
            fps=args.fps,
        )
    except Exception as e:
        logger.error(f"Failed to create renderer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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
        total_conceded = sum(ep["red_goals"] for ep in episode_stats)

        print(f"Episodes: {len(episode_stats)}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Total Goals Scored: {total_goals}")
        print(f"Total Goals Conceded: {total_conceded}")
        print(f"Goal Difference: {total_goals - total_conceded:+d}")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        try:
            renderer.close()
        except:
            pass
        logger.info("Visualization complete")


if __name__ == "__main__":
    main()