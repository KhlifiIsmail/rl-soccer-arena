"""Evaluation entry point for RL Soccer Arena.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.zip
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.zip --episodes 200
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import AgentEvaluator
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL soccer agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint to evaluate",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--env-mode",
        type=str,
        default="3d",
        choices=["2d", "3d"],
        help="Environment mode to match the trained model",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file for results (JSON)",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    logger = setup_logger(name="rl_soccer_arena", level="INFO")

    logger.info("=" * 60)
    logger.info("RL SOCCER ARENA - EVALUATION")
    logger.info("=" * 60)

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Create evaluator
    logger.info(f"Loading model: {checkpoint_path}")
    evaluator = AgentEvaluator(
        model_path=checkpoint_path,
        env_mode=args.env_mode,
        n_eval_episodes=args.episodes,
    )

    # Run evaluation
    try:
        logger.info(f"Evaluating for {args.episodes} episodes...")
        stats = evaluator.evaluate(deterministic=args.deterministic)

        # Print results
        evaluator.print_results()

        # Save results if output file specified
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Saved results to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
