"""Soccer agent implementation with 3D PyBullet physics.

This module defines the soccer agent entity with physical representation,
movement capabilities, and state tracking.
"""

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
    """Agent physical configuration.

    Attributes:
        radius: Agent body radius in meters.
        height: Agent body height in meters.
        mass: Agent mass in kg.
        max_force: Maximum force applied for movement in Newtons.
        max_angular_force: Maximum torque for rotation in N⋅m.
        lateral_friction: Lateral friction coefficient.
        restitution: Bounciness (0=no bounce, 1=perfect bounce).
    """

    radius: float = 0.5
    height: float = 1.0
    mass: float = 5.0
    max_force: float = 50.0
    max_angular_force: float = 10.0
    lateral_friction: float = 0.8
    restitution: float = 0.3


class SoccerAgent:
    """3D Soccer agent with PyBullet physics.

    Represents a single soccer player with cylindrical body, capable of:
    - Forward/backward movement
    - Rotation (turning)
    - Strafing (lateral movement)
    - Collision detection with ball and other agents

    Attributes:
        team: Agent's team (BLUE or RED).
        config: Physical configuration parameters.
        physics_client: PyBullet physics client ID.
        body_id: PyBullet body ID for this agent.
    """

    def __init__(
        self,
        team: Team,
        physics_client: int,
        start_position: np.ndarray,
        start_orientation: float = 0.0,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize soccer agent.

        Args:
            team: Agent's team.
            physics_client: PyBullet physics client ID.
            start_position: Initial position [x, y, z].
            start_orientation: Initial yaw angle in radians.
            config: Agent configuration, uses defaults if None.
        """
        self.team = team
        self.physics_client = physics_client
        self.config = config or AgentConfig()
        self.start_position = start_position.copy()
        self.start_orientation = start_orientation

        self.body_id = self._create_body()
        self.reset()

    def _create_body(self) -> int:
        """Create agent body in PyBullet.

        Returns:
            PyBullet body ID.
        """
        # Collision shape (cylinder for agent body)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.config.radius,
            height=self.config.height,
            physicsClientId=self.physics_client
        )

        # Visual shape with team color
        color = [0.2, 0.2, 0.8, 1.0] if self.team == Team.BLUE else [0.8, 0.2, 0.2, 1.0]
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.config.radius,
            length=self.config.height,
            rgbaColor=color,
            physicsClientId=self.physics_client
        )

        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=self.config.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.start_position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, self.start_orientation]),
            physicsClientId=self.physics_client
        )

        # Set dynamics properties
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=self.config.lateral_friction,
            restitution=self.config.restitution,
            linearDamping=0.4,
            angularDamping=0.4,
            physicsClientId=self.physics_client
        )

        return body_id

    def reset(self) -> None:
        """Reset agent to starting position and orientation."""
        p.resetBasePositionAndOrientation(
            self.body_id,
            self.start_position,
            p.getQuaternionFromEuler([0, 0, self.start_orientation]),
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
        """Get agent's current position.

        Returns:
            Position array [x, y, z].
        """
        position, _ = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(position)

    def get_velocity(self) -> np.ndarray:
        """Get agent's current linear velocity.

        Returns:
            Velocity array [vx, vy, vz].
        """
        velocity, _ = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return np.array(velocity)

    def get_orientation(self) -> float:
        """Get agent's current yaw orientation.

        Returns:
            Yaw angle in radians (-π to π).
        """
        _, quaternion = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.physics_client
        )
        euler = p.getEulerFromQuaternion(quaternion)
        return euler[2]  # Yaw angle

    def get_angular_velocity(self) -> float:
        """Get agent's current angular velocity around z-axis.

        Returns:
            Angular velocity in radians/second.
        """
        _, angular_velocity = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.physics_client
        )
        return angular_velocity[2]

    def get_forward_vector(self) -> np.ndarray:
        """Get agent's forward direction vector.

        Returns:
            Forward unit vector [x, y, 0].
        """
        yaw = self.get_orientation()
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def get_right_vector(self) -> np.ndarray:
        """Get agent's right direction vector.

        Returns:
            Right unit vector [x, y, 0].
        """
        yaw = self.get_orientation()
        return np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    def apply_action(self, action: np.ndarray) -> None:
        """Apply action to agent.

        Action space: [forward, strafe, turn]
        - forward: [-1, 1] (backward to forward)
        - strafe: [-1, 1] (left to right)
        - turn: [-1, 1] (counter-clockwise to clockwise)

        Args:
            action: Action array of shape (3,).
        """
        forward, strafe, turn = action

        # Clamp actions to valid range
        forward = np.clip(forward, -1.0, 1.0)
        strafe = np.clip(strafe, -1.0, 1.0)
        turn = np.clip(turn, -1.0, 1.0)

        # Calculate force in world frame
        forward_vec = self.get_forward_vector()
        right_vec = self.get_right_vector()

        force = (
            forward * self.config.max_force * forward_vec +
            strafe * self.config.max_force * right_vec
        )

        # Apply linear force
        p.applyExternalForce(
            self.body_id,
            -1,
            forceObj=force,
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )

        # Apply angular torque
        torque = turn * self.config.max_angular_force
        p.applyExternalTorque(
            self.body_id,
            -1,
            torqueObj=[0, 0, torque],
            flags=p.WORLD_FRAME,
            physicsClientId=self.physics_client
        )

    def get_observation(self) -> np.ndarray:
        """Get agent's self-observation.

        Returns:
            Observation array: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, yaw].
        """
        position = self.get_position()
        velocity = self.get_velocity()
        yaw = self.get_orientation()

        return np.concatenate([position, velocity, [yaw]])

    def get_distance_to(self, position: np.ndarray) -> float:
        """Calculate distance to target position.

        Args:
            position: Target position [x, y, z].

        Returns:
            Euclidean distance.
        """
        return np.linalg.norm(self.get_position() - position)

    def cleanup(self) -> None:
        """Remove agent body from PyBullet simulation."""
        try:
            p.removeBody(self.body_id, physicsClientId=self.physics_client)
        except p.error:
            pass  # Body already removed
