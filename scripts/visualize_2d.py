"""Visualize a trained policy in the 2D kinematic soccer environment.

Usage:
    python scripts/visualize_2d.py --checkpoint outputs/checkpoints/final_model.zip --steps 500
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from src.environments.soccer_env_2d import SoccerEnv2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 2D soccer policy rollout")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.zip)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Max steps to roll out",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    env = SoccerEnv2D(max_episode_steps=args.steps)
    model = PPO.load(ckpt, env=env)

    obs, info = env.reset()

    xs_b, ys_b, xs_r, ys_r, xs_ball, ys_ball = [], [], [], [], [], []
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, terminated, truncated, info = env.step(action)

        xs_b.append(info["blue_position"][0])
        ys_b.append(info["blue_position"][1])
        xs_r.append(info["red_position"][0])
        ys_r.append(info["red_position"][1])
        xs_ball.append(info["ball_position"][0])
        ys_ball.append(info["ball_position"][1])

        if terminated or truncated:
            break

    # Plot trajectories
    plt.figure(figsize=(8, 5))
    plt.plot(xs_b, ys_b, label="Blue (agent)")
    plt.plot(xs_r, ys_r, label="Red (opponent)")
    plt.plot(xs_ball, ys_ball, label="Ball", color="orange")
    plt.scatter([xs_b[0]], [ys_b[0]], marker="o", color="blue", label="Blue start")
    plt.scatter([xs_r[0]], [ys_r[0]], marker="o", color="red", label="Red start")
    plt.scatter([xs_ball[0]], [ys_ball[0]], marker="o", color="orange", label="Ball start")

    plt.axvline(env.half_length, color="green", linestyle="--", label="Red goal line")
    plt.axvline(-env.half_length, color="purple", linestyle="--", label="Blue goal line")
    plt.axhline(env.goal_width / 2, color="gray", linestyle=":", linewidth=0.8)
    plt.axhline(-env.goal_width / 2, color="gray", linestyle=":", linewidth=0.8)

    plt.title(f"2D Soccer Rollout ({step+1} steps)")
    plt.legend()
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
