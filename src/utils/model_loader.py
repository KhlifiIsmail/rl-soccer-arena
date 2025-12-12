"""Utility to load models from policy weights when full model can't be saved."""

from pathlib import Path
import torch
from stable_baselines3 import PPO
from src.environments.soccer_env import SoccerEnv


def load_policy_weights(policy_path: str | Path, env=None):
    """Load model from saved policy weights.
    
    Args:
        policy_path: Path to .pth policy weights file
        env: Environment (will create new one if None)
        
    Returns:
        PPO model with loaded weights
    """
    policy_path = Path(policy_path)
    
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy weights not found: {policy_path}")
    
    # Create environment if not provided
    if env is None:
        env = SoccerEnv(render_mode=None)
    
    # Create new model
    model = PPO("MlpPolicy", env, device="cuda")
    
    # Load policy weights
    model.policy.load_state_dict(torch.load(policy_path))
    
    print(f"Loaded policy weights from: {policy_path}")
    
    return model