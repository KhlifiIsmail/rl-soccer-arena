"""Custom feature extractors for processing soccer observations.

This module provides neural network components for extracting features
from raw soccer environment observations before policy/value computation.
"""

from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SoccerFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for soccer observations.

    Processes soccer-specific observations with structured encoding:
    - Separate processing for agent state, ball state, opponent state
    - Spatial relationship encoding
    - Feature fusion with attention mechanism

    Attributes:
        features_dim: Output feature dimension.
        agent_encoder: Network for encoding agent self-state.
        ball_encoder: Network for encoding ball state.
        opponent_encoder: Network for encoding opponent state.
        fusion_net: Network for fusing encoded features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize soccer feature extractor.

        Args:
            observation_space: Gym observation space.
            features_dim: Output feature dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__(observation_space, features_dim)

        # Observation breakdown:
        # Agent: position(3) + velocity(3) + orientation(1) = 7
        # Ball: position(3) + velocity(3) = 6
        # Opponent: position(3) + velocity(3) = 6
        # Goal distances: 2
        # Total: 21

        self.agent_dim = 7
        self.ball_dim = 6
        self.opponent_dim = 6
        self.spatial_dim = 2

        # Agent state encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Ball state encoder
        self.ball_encoder = nn.Sequential(
            nn.Linear(self.ball_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Opponent state encoder
        self.opponent_encoder = nn.Sequential(
            nn.Linear(self.opponent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Spatial information encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Feature fusion network
        fusion_input_dim = hidden_dim * 3 + hidden_dim // 2
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations.

        Args:
            observations: Batch of observations [batch_size, 21].

        Returns:
            Extracted features [batch_size, features_dim].
        """
        # Split observations into components
        agent_obs = observations[:, :7]
        ball_obs = observations[:, 7:13]
        opponent_obs = observations[:, 13:19]
        spatial_obs = observations[:, 19:21]

        # Encode each component
        agent_features = self.agent_encoder(agent_obs)
        ball_features = self.ball_encoder(ball_obs)
        opponent_features = self.opponent_encoder(opponent_obs)
        spatial_features = self.spatial_encoder(spatial_obs)

        # Concatenate all features
        combined = torch.cat([
            agent_features,
            ball_features,
            opponent_features,
            spatial_features
        ], dim=1)

        # Fuse features
        features = self.fusion_net(combined)

        return features


class SimpleSoccerFeatureExtractor(BaseFeaturesExtractor):
    """Simplified feature extractor for soccer (faster, less complex).

    Uses simple MLP without structured decomposition.
    Good baseline for comparison with structured extractor.

    Attributes:
        features_dim: Output feature dimension.
        network: Simple feedforward network.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ) -> None:
        """Initialize simple feature extractor.

        Args:
            observation_space: Gym observation space.
            features_dim: Output feature dimension.
        """
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations.

        Args:
            observations: Batch of observations.

        Returns:
            Extracted features.
        """
        return self.network(observations)


class AttentionSoccerFeatureExtractor(BaseFeaturesExtractor):
    """Advanced feature extractor with attention mechanism.

    Uses attention to focus on relevant aspects of game state:
    - Attention over ball, opponent, spatial features
    - Dynamic weighting based on situation

    Attributes:
        features_dim: Output feature dimension.
        encoders: Dictionary of component encoders.
        attention: Attention mechanism.
        fusion_net: Final fusion network.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        hidden_dim: int = 128,
        n_heads: int = 4,
    ) -> None:
        """Initialize attention-based feature extractor.

        Args:
            observation_space: Gym observation space.
            features_dim: Output feature dimension.
            hidden_dim: Hidden dimension for encoders.
            n_heads: Number of attention heads.
        """
        super().__init__(observation_space, features_dim)

        # Component dimensions
        self.agent_dim = 7
        self.ball_dim = 6
        self.opponent_dim = 6
        self.spatial_dim = 2

        # Encoders
        self.agent_encoder = nn.Linear(self.agent_dim, hidden_dim)
        self.ball_encoder = nn.Linear(self.ball_dim, hidden_dim)
        self.opponent_encoder = nn.Linear(self.opponent_dim, hidden_dim)
        self.spatial_encoder = nn.Linear(self.spatial_dim, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features with attention.

        Args:
            observations: Batch of observations [batch_size, 21].

        Returns:
            Extracted features [batch_size, features_dim].
        """
        batch_size = observations.shape[0]

        # Split and encode components
        agent_obs = observations[:, :7]
        ball_obs = observations[:, 7:13]
        opponent_obs = observations[:, 13:19]
        spatial_obs = observations[:, 19:21]

        agent_feat = self.agent_encoder(agent_obs).unsqueeze(1)
        ball_feat = self.ball_encoder(ball_obs).unsqueeze(1)
        opponent_feat = self.opponent_encoder(opponent_obs).unsqueeze(1)
        spatial_feat = self.spatial_encoder(spatial_obs).unsqueeze(1)

        # Stack features: [batch_size, 4, hidden_dim]
        features = torch.cat([
            agent_feat,
            ball_feat,
            opponent_feat,
            spatial_feat
        ], dim=1)

        # Apply self-attention
        attended_features, _ = self.attention(features, features, features)

        # Aggregate attended features (mean pooling)
        aggregated = attended_features.mean(dim=1)

        # Project to output dimension
        output = self.output_proj(aggregated)

        return output
