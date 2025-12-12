"""Replay viewer for saved matches.

Load and replay saved match trajectories for analysis and visualization.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class MatchRecorder:
    """Record match trajectories for later replay.

    Attributes:
        trajectory: List of state snapshots.
        metadata: Match metadata.
    """

    def __init__(self) -> None:
        """Initialize match recorder."""
        self.trajectory: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def record_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Record single timestep.

        Args:
            observation: Environment observation.
            action: Action taken.
            reward: Reward received.
            info: Info dictionary.
        """
        self.trajectory.append({
            "observation": observation.copy(),
            "action": action.copy(),
            "reward": reward,
            "info": info.copy(),
        })

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set match metadata.

        Args:
            metadata: Metadata dictionary.
        """
        self.metadata = metadata

    def save(self, save_path: str | Path) -> None:
        """Save recording to file.

        Args:
            save_path: Path to save file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trajectory": self.trajectory,
            "metadata": self.metadata,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def reset(self) -> None:
        """Reset recorder."""
        self.trajectory.clear()
        self.metadata.clear()


class ReplayViewer:
    """View saved match replays.

    Attributes:
        trajectory: Loaded match trajectory.
        metadata: Match metadata.
    """

    def __init__(self, replay_path: str | Path) -> None:
        """Initialize replay viewer.

        Args:
            replay_path: Path to saved replay file.
        """
        self.replay_path = Path(replay_path)
        self.trajectory: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

        self._load_replay()

    def _load_replay(self) -> None:
        """Load replay from file."""
        with open(self.replay_path, "rb") as f:
            data = pickle.load(f)

        self.trajectory = data["trajectory"]
        self.metadata = data["metadata"]

    def get_summary(self) -> Dict[str, Any]:
        """Get replay summary.

        Returns:
            Summary dictionary.
        """
        if not self.trajectory:
            return {}

        total_reward = sum(step["reward"] for step in self.trajectory)
        final_info = self.trajectory[-1]["info"]

        return {
            "length": len(self.trajectory),
            "total_reward": total_reward,
            "blue_goals": final_info.get("blue_goals", 0),
            "red_goals": final_info.get("red_goals", 0),
            "metadata": self.metadata,
        }

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get full trajectory.

        Returns:
            List of trajectory steps.
        """
        return self.trajectory
