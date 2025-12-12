"""Live 2D visualization of a trained policy in the kinematic env.

Usage:
    python scripts/visualize_2d_live.py --checkpoint outputs/checkpoints/final_model.zip --steps 500
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from src.environments.soccer_env_2d import SoccerEnv2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live 2D visualization")
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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-env.half_length - 1, env.half_length + 1)
    ax.set_ylim(-env.half_width - 1, env.half_width + 1)
    ax.set_aspect("equal")
    ax.set_title("2D Soccer Live Rollout")
    ax.axvline(env.half_length, color="green", linestyle="--", label="Red goal line")
    ax.axvline(-env.half_length, color="purple", linestyle="--", label="Blue goal line")
    ax.axhline(env.goal_width / 2, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(-env.goal_width / 2, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right")

    blue_dot, = ax.plot([], [], "bo", label="Blue")
    red_dot, = ax.plot([], [], "ro", label="Red")
    ball_dot, = ax.plot([], [], "o", color="orange", label="Ball")

    def init():
        blue_dot.set_data([], [])
        red_dot.set_data([], [])
        ball_dot.set_data([], [])
        return blue_dot, red_dot, ball_dot

    def update(frame):
        nonlocal obs
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, terminated, truncated, info = env.step(action)

        blue_dot.set_data(info["blue_position"][0], info["blue_position"][1])
        red_dot.set_data(info["red_position"][0], info["red_position"][1])
        ball_dot.set_data(info["ball_position"][0], info["ball_position"][1])

        if terminated or truncated:
            anim.event_source.stop()
        return blue_dot, red_dot, ball_dot

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=args.steps, interval=50, blit=True
    )
    plt.show()


if __name__ == "__main__":
    main()
