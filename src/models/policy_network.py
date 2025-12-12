"""Custom policy network architectures for soccer agents.

This module provides specialized policy and value networks optimized
for soccer gameplay with customizable architectures.
"""

from typing import Dict, List, Tuple, Type

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.models.feature_extractor import SoccerFeatureExtractor


class SoccerActorCriticPolicy(ActorCriticPolicy):
    """Custom Actor-Critic policy for soccer environment.

    Extends SB3's ActorCriticPolicy with custom network architecture
    and initialization optimized for soccer gameplay.

    Attributes:
        features_extractor: Feature extraction network.
        policy_net: Actor network (action policy).
        value_net: Critic network (value function).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: List[Dict[str, List[int]]] | None = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = SoccerFeatureExtractor,
        features_extractor_kwargs: Dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize soccer actor-critic policy.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            lr_schedule: Learning rate schedule.
            net_arch: Network architecture specification.
            activation_fn: Activation function.
            features_extractor_class: Feature extractor class.
            features_extractor_kwargs: Kwargs for feature extractor.
            *args: Additional arguments for parent class.
            **kwargs: Additional keyword arguments for parent class.
        """
        # Default network architecture
        if net_arch is None:
            net_arch = [
                dict(pi=[256, 256], vf=[256, 256])
            ]

        # Default feature extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(features_dim=256)

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs,
        )

        # Custom initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Custom weight initialization for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


class SeparateActorCriticNet(nn.Module):
    """Separate networks for actor and critic with shared feature extractor.

    Uses a shared feature extractor followed by separate actor and critic heads.
    This architecture allows for better learning dynamics.

    Attributes:
        feature_extractor: Shared feature extraction network.
        actor_head: Policy (actor) network head.
        critic_head: Value (critic) network head.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        action_dim: int = 3,
        hidden_dims: List[int] = [256, 256],
    ) -> None:
        """Initialize separate actor-critic network.

        Args:
            feature_dim: Feature dimension from extractor.
            action_dim: Action space dimension.
            hidden_dims: Hidden layer dimensions for actor/critic heads.
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Actor head
        actor_layers = []
        input_dim = feature_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        # Output layer for actor (mean of action distribution)
        actor_layers.append(nn.Linear(input_dim, action_dim))
        self.actor_head = nn.Sequential(*actor_layers)

        # Critic head
        critic_layers = []
        input_dim = feature_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        # Output layer for critic (value)
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic_head = nn.Sequential(*critic_layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor and critic.

        Args:
            features: Extracted features [batch_size, feature_dim].

        Returns:
            Tuple of (actions, values).
        """
        actions = self.actor_head(features)
        values = self.critic_head(features)

        return actions, values


class ResidualBlock(nn.Module):
    """Residual block for deeper policy networks.

    Adds skip connections to allow training deeper networks.

    Attributes:
        layers: Main transformation layers.
        activation: Activation function.
    """

    def __init__(self, dim: int, activation_fn: Type[nn.Module] = nn.ReLU) -> None:
        """Initialize residual block.

        Args:
            dim: Input/output dimension.
            activation_fn: Activation function class.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation_fn(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.activation = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output with residual connection.
        """
        return self.activation(x + self.layers(x))


class DeepSoccerPolicy(nn.Module):
    """Deep policy network with residual connections.

    Uses residual blocks for better gradient flow in deep networks.
    Suitable for complex environments requiring deeper representations.

    Attributes:
        input_layer: Input projection layer.
        residual_blocks: Stack of residual blocks.
        actor_head: Policy output head.
        critic_head: Value output head.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        action_dim: int = 3,
        n_residual_blocks: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize deep soccer policy.

        Args:
            feature_dim: Input feature dimension.
            action_dim: Action space dimension.
            n_residual_blocks: Number of residual blocks.
            hidden_dim: Hidden dimension for residual blocks.
        """
        super().__init__()

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_residual_blocks)
        ])

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through deep policy.

        Args:
            features: Input features [batch_size, feature_dim].

        Returns:
            Tuple of (actions, values).
        """
        # Project input
        x = self.input_layer(features)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Compute actor and critic outputs
        actions = self.actor_head(x)
        values = self.critic_head(x)

        return actions, values
