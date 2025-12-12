"""Agent evaluation utilities.

Provides comprehensive evaluation of trained agents including:
- Performance metrics
- Win rates
- Goal statistics
- Consistency analysis
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm

from src.environments.soccer_env import SoccerEnv
from src.environments.soccer_env_2d import SoccerEnv2D


class AgentEvaluator:
    """Evaluate trained soccer agents.

    Attributes:
        model: Trained PPO model to evaluate.
        env: Evaluation environment.
        n_eval_episodes: Number of evaluation episodes.
    """

    def __init__(
        self,
        model_path: str | Path,
        env_config: Dict[str, Any] | None = None,
        env_mode: str = "3d",
        n_eval_episodes: int = 100,
    ) -> None:
        """Initialize agent evaluator.

        Args:
            model_path: Path to trained model checkpoint.
            env_config: Environment configuration.
            n_eval_episodes: Number of episodes for evaluation.
        """
        self.model_path = Path(model_path)
        self.n_eval_episodes = n_eval_episodes

        # Load model
        env_config = env_config or {}
        env_mode = env_mode.lower()
        if env_mode == "2d":
            self.env = SoccerEnv2D(render_mode=None, **env_config)
        else:
            self.env = SoccerEnv(render_mode=None, **env_config)
        self.model = PPO.load(self.model_path, env=self.env)

        # Results storage
        self.results: List[Dict[str, Any]] = []

    def evaluate(self, deterministic: bool = True) -> Dict[str, Any]:
        """Run evaluation.

        Args:
            deterministic: Use deterministic policy.

        Returns:
            Dictionary of evaluation metrics.
        """
        print(f"Evaluating model: {self.model_path.name}")
        print(f"Running {self.n_eval_episodes} episodes...")

        self.results.clear()

        for episode in tqdm(range(self.n_eval_episodes), desc="Evaluating"):
            episode_result = self._run_episode(deterministic=deterministic)
            self.results.append(episode_result)

        # Compute statistics
        stats = self._compute_statistics()

        return stats

    def _run_episode(self, deterministic: bool) -> Dict[str, Any]:
        """Run single evaluation episode.

        Args:
            deterministic: Use deterministic policy.

        Returns:
            Episode results.
        """
        obs, info = self.env.reset()

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        return {
            "reward": episode_reward,
            "length": episode_length,
            "blue_goals": info.get("blue_goals", 0),
            "red_goals": info.get("red_goals", 0),
            "goal_scored": info.get("blue_goals", 0) > 0,
            "goal_conceded": info.get("red_goals", 0) > 0,
            "won": info.get("blue_goals", 0) > info.get("red_goals", 0),
        }

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute evaluation statistics.

        Returns:
            Statistics dictionary.
        """
        rewards = [r["reward"] for r in self.results]
        lengths = [r["length"] for r in self.results]
        wins = sum(r["won"] for r in self.results)
        goals_scored = sum(r["blue_goals"] for r in self.results)
        goals_conceded = sum(r["red_goals"] for r in self.results)

        return {
            "n_episodes": len(self.results),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
            "win_rate": float(wins / len(self.results)),
            "total_goals_scored": goals_scored,
            "total_goals_conceded": goals_conceded,
            "goal_difference": goals_scored - goals_conceded,
            "avg_goals_per_episode": float(goals_scored / len(self.results)),
        }

    def print_results(self) -> None:
        """Print evaluation results."""
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return

        stats = self._compute_statistics()

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes: {stats['n_episodes']}")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Mean Episode Length: {stats['mean_episode_length']:.1f}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        print(f"Goals Scored: {stats['total_goals_scored']}")
        print(f"Goals Conceded: {stats['total_goals_conceded']}")
        print(f"Goal Difference: {stats['goal_difference']:+d}")
        print(f"Avg Goals/Episode: {stats['avg_goals_per_episode']:.2f}")
        print("=" * 60 + "\n")

    def close(self) -> None:
        """Close environment."""
        if self.env:
            self.env.close()
