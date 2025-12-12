"""Soccer ball implementation with 3D PyBullet physics.

This module defines the soccer ball entity with realistic physics,
collision handling, and state tracking.
"""

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class BallConfig:
    """Ball physical configuration.

    Attributes:
        radius: Ball radius in meters.
        mass: Ball mass in kg.
        restitution: Bounciness (0=no bounce, 1=perfect bounce).
        lateral_friction: Lateral friction coefficient.
        rolling_friction: Rolling friction coefficient.
        spinning_friction: Spinning friction coefficient.
    """

    radius: float = 0.22  # Standard soccer ball radius (~22cm)
    mass: float = 0.45  # Standard soccer ball mass (~450g)
    restitution: float = 0.7
    lateral_friction: float = 0.5
    rolling_friction: float = 0.01
    spinning_friction: float = 0.01


class SoccerBall:
    """3D Soccer ball with PyBullet physics.

    Represents the soccer ball with realistic physics:
    - Spherical collision shape
    - Realistic mass and friction
    - Bouncing and rolling behavior
    - Collision detection with agents and walls

    Attributes:
        config: Physical configuration parameters.
        physics_client: PyBullet physics client ID.
        body_id: PyBullet body ID for this ball.
    """

    def __init__(
        self,
        physics_client: int,
        start_position: np.ndarray,
        config: BallConfig | None = None,
    ) -> None:
        """Initialize soccer ball.

        Args:
            physics_client: PyBullet physics client ID.
            start_position: Initial position [x, y, z].
            config: Ball configuration, uses defaults if None.
        """
        self.physics_client = physics_client
        self.config = config or BallConfig()
        self.start_position = start_position.copy()

        self.body_id = self._create_body()
        self.reset()

    def _create_body(self) -> int:
        """Create ball body in PyBullet.

        Returns:
            PyBullet body ID.
        """
        # Collision shape (sphere)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.config.radius,
            physicsClientId=self.physics_client
        )

        # Visual shape (classic soccer ball - black and white)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.config.radius,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],  # White base
            physicsClientId=self.physics_client
        )

        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=self.config.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.start_position,
            physicsClientId=self.physics_client
        )

        # Set dynamics properties for realistic ball physics
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=self.config.lateral_friction,
            rollingFriction=self.config.rolling_friction,
            spinningFriction=self.config.spinning_friction,
            restitution=self.config.restitution,
            linearDamping=0.04,  # Air resistance
            angularDamping=0.04,
            physicsClientId=self.physics_client
        )

        return body_id

    def reset(self, position: np.ndarray | None = None) -> None:
        """Reset ball to specified or starting position.

        Args:
            position: Reset position, uses start_position if None.
        """
        reset_pos = position if position is not None else self.start_position

        p.resetBasePositionAndOrientation(
            self.body_id,
            reset_pos,
            [0, 0, 0, 1],  # No rotation
            physicsClientId=self.physics_client
        )

        # Reset velocities
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.physics_client
        )

    def get_position(self) -> np.ndarray:
        """Get ball's current position.

        Returns:
            Position array [x, y, z].
        """
        position, _ = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(position)

    def get_velocity(self) -> np.ndarray:
        """Get ball's current linear velocity.

        Returns:
            Velocity array [vx, vy, vz].
        """
        velocity, _ = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(velocity)

    def get_angular_velocity(self) -> np.ndarray:
        """Get ball's current angular velocity.

        Returns:
            Angular velocity array [wx, wy, wz].
        """
        _, angular_velocity = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(angular_velocity)

    def get_speed(self) -> float:
        """Get ball's current speed (magnitude of velocity).

        Returns:
            Speed in meters/second.
        """
        return np.linalg.norm(self.get_velocity())

    def is_stationary(self, threshold: float = 0.05) -> bool:
        """Check if ball is nearly stationary.

        Args:
            threshold: Velocity threshold for considering ball stationary.

        Returns:
            True if ball speed is below threshold.
        """
        return self.get_speed() < threshold

    def apply_impulse(self, impulse: np.ndarray, position: np.ndarray | None = None) -> None:
        """Apply impulse to ball (instantaneous force).

        Args:
            impulse: Impulse vector [fx, fy, fz] in N⋅s.
            position: Point of application in world coordinates, uses center if None.
        """
        if position is None:
            position = self.get_position()

        p.applyExternalForce(
            self.body_id,
            -1,
            forceObj=impulse,
            posObj=position,
            flags=p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )

    def get_distance_to(self, position: np.ndarray) -> float:
        """Calculate distance to target position.

        Args:
            position: Target position [x, y, z].

        Returns:
            Euclidean distance.
        """
        return np.linalg.norm(self.get_position() - position)

    def get_contact_points(self) -> list:
        """Get all current contact points with other objects.

        Returns:
            List of contact point information from PyBullet.
        """
        return p.getContactPoints(
            bodyA=self.body_id,
            physicsClientId=self.physics_client
        )

    def is_touching_agent(self, agent_body_id: int) -> bool:
        """Check if ball is in contact with specified agent.

        Args:
            agent_body_id: PyBullet body ID of agent to check.

        Returns:
            True if ball is touching the agent.
        """
        contact_points = p.getContactPoints(
            bodyA=self.body_id,
            bodyB=agent_body_id,
            physicsClientId=self.physics_client
        )
        return len(contact_points) > 0

    def get_observation(self) -> np.ndarray:
        """Get ball's observation.

        Returns:
            Observation array: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z].
        """
        position = self.get_position()
        velocity = self.get_velocity()
        return np.concatenate([position, velocity])

    def cleanup(self) -> None:
        """Remove ball body from PyBullet simulation."""
        try:
            p.removeBody(self.body_id, physicsClientId=self.physics_client)
        except p.error:
            pass  # Body already removed
