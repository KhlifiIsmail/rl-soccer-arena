"""Statistical analysis utilities for evaluation results."""

from typing import Dict, List

import numpy as np
import pandas as pd


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for data.

    Args:
        data: List of values.
        confidence: Confidence level (default 95%).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(data) == 0:
        return (0.0, 0.0)

    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))

    # Z-score for 95% confidence
    z_score = 1.96 if confidence == 0.95 else 2.576

    margin = z_score * std_err

    return (mean - margin, mean + margin)


def compare_agents(
    results_a: Dict[str, float],
    results_b: Dict[str, float],
) -> Dict[str, str]:
    """Compare two agents statistically.

    Args:
        results_a: Agent A results.
        results_b: Agent B results.

    Returns:
        Comparison dictionary.
    """
    comparison = {}

    for metric in ["mean_reward", "win_rate", "avg_goals_per_episode"]:
        if metric in results_a and metric in results_b:
            diff = results_a[metric] - results_b[metric]
            pct_diff = (diff / abs(results_b[metric])) * 100 if results_b[metric] != 0 else 0

            comparison[metric] = {
                "agent_a": results_a[metric],
                "agent_b": results_b[metric],
                "difference": diff,
                "percent_difference": pct_diff,
                "better": "A" if diff > 0 else "B" if diff < 0 else "Tie",
            }

    return comparison
