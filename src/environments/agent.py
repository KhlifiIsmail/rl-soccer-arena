"""Soccer agent implementation - NO ROTATION, DASH AND KICK ONLY."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import pybullet as p


class Team(Enum):
    """Team enumeration."""
    BLUE = 0
    RED = 1


@dataclass(frozen=True)
class AgentConfig:
    """Agent physical configuration."""
    radius: float = 0.5
    height: float = 1.0
    mass: float = 10.0
    max_force: float = 500.0  # SUPER FAST
    lateral_friction: float = 0.9
    restitution: float = 0.5


class SoccerAgent:
    """Soccer agent - moves in 8 directions, no rotation."""

    def __init__(
        self,
        team: Team,
        physics_client: int,
        start_position: np.ndarray,
        start_orientation: float = 0.0,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize soccer agent."""
        self.team = team
        self.physics_client = physics_client
        self.config = config or AgentConfig()
        self.start_position = start_position.copy()
        self.start_orientation = start_orientation

        self.body_id = self._create_body()
        self.reset()

    def _create_body(self) -> int:
        """Create agent body - SPHERE so it doesn't rotate."""
        # Use SPHERE instead of cylinder - no rotation issues
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.config.radius,
            physicsClientId=self.physics_client
        )

        color = [0.2, 0.4, 1.0, 1.0] if self.team == Team.BLUE else [1.0, 0.2, 0.2, 1.0]
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.config.radius,
            rgbaColor=color,
            physicsClientId=self.physics_client
        )

        body_id = p.createMultiBody(
            baseMass=self.config.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.start_position,
            physicsClientId=self.physics_client
        )

        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=self.config.lateral_friction,
            restitution=self.config.restitution,
            linearDamping=0.05,
            angularDamping=0.99,  # HIGH damping to prevent spinning
            physicsClientId=self.physics_client
        )

        return body_id

    def reset(self) -> None:
        """Reset agent to starting position."""
        p.resetBasePositionAndOrientation(
            self.body_id,
            self.start_position,
            [0, 0, 0, 1],
            physicsClientId=self.physics_client
        )

        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.physics_client
        )

    def get_position(self) -> np.ndarray:
        """Get agent's current position."""
        position, _ = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(position)

    def get_velocity(self) -> np.ndarray:
        """Get agent's current velocity."""
        velocity, _ = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(velocity)

    def get_orientation(self) -> float:
        """Get orientation - not used anymore but kept for compatibility."""
        return 0.0

    def get_angular_velocity(self) -> float:
        """Get angular velocity - not used."""
        return 0.0

    def get_forward_vector(self) -> np.ndarray:
        """Not used anymore."""
        return np.array([1.0, 0.0, 0.0])

    def get_right_vector(self) -> np.ndarray:
        """Not used anymore."""
        return np.array([0.0, 1.0, 0.0])

    def apply_action(self, action: np.ndarray) -> None:
        """
        Apply action - SIMPLE 2D MOVEMENT.
        action[0] = x direction (-1 to 1)
        action[1] = y direction (-1 to 1)
        """
        x_dir = np.clip(action[0], -1.0, 1.0)
        y_dir = np.clip(action[1], -1.0, 1.0)

        # Direct force in world coordinates - NO ROTATION
        force = np.array([
            x_dir * self.config.max_force,
            y_dir * self.config.max_force,
            0.0
        ])

        p.applyExternalForce(
            self.body_id,
            -1,
            forceObj=force,
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )

    def get_observation(self) -> np.ndarray:
        """Get agent's observation."""
        position = self.get_position()
        velocity = self.get_velocity()
        return np.concatenate([position, velocity])

    def get_distance_to(self, position: np.ndarray) -> float:
        """Calculate distance to target."""
        return np.linalg.norm(self.get_position() - position)

    def cleanup(self) -> None:
        """Remove agent body."""
        try:
            p.removeBody(self.body_id, physicsClientId=self.physics_client)
        except p.error:
            pass