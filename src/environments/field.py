"""Soccer field implementation with 3D PyBullet physics.

This module defines the soccer field dimensions, boundaries, goals, and creates
the 3D physical representation in PyBullet.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class FieldDimensions:
    """Immutable field dimensions configuration.

    Attributes:
        length: Field length (along x-axis) in meters.
        width: Field width (along y-axis) in meters.
        wall_height: Height of boundary walls in meters.
        wall_thickness: Thickness of boundary walls in meters.
        goal_width: Width of goal openings in meters.
        goal_depth: Depth of goals (extends outward) in meters.
        goal_height: Height of goal posts in meters.
    """

    length: float = 30.0
    width: float = 20.0
    wall_height: float = 2.0
    wall_thickness: float = 0.5
    goal_width: float = 4.0
    goal_depth: float = 1.0
    goal_height: float = 2.0

    @property
    def half_length(self) -> float:
        """Half field length."""
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        """Half field width."""
        return self.width / 2.0

    @property
    def blue_goal_center(self) -> np.ndarray:
        """Blue team goal center position (negative x)."""
        return np.array([-self.half_length, 0.0, self.goal_height / 2.0])

    @property
    def red_goal_center(self) -> np.ndarray:
        """Red team goal center position (positive x)."""
        return np.array([self.half_length, 0.0, self.goal_height / 2.0])

    def is_in_blue_goal(self, position: np.ndarray) -> bool:
        """Check if position is inside blue team's goal.

        Args:
            position: 3D position [x, y, z].

        Returns:
            True if position is inside blue goal zone.
        """
        return (
            position[0] < -self.half_length
            and abs(position[1]) < self.goal_width / 2.0
            and 0 < position[2] < self.goal_height
        )

    def is_in_red_goal(self, position: np.ndarray) -> bool:
        """Check if position is inside red team's goal.

        Args:
            position: 3D position [x, y, z].

        Returns:
            True if position is inside red goal zone.
        """
        return (
            position[0] > self.half_length
            and abs(position[1]) < self.goal_width / 2.0
            and 0 < position[2] < self.goal_height
        )

    def clamp_position(self, position: np.ndarray) -> np.ndarray:
        """Clamp position to field boundaries.

        Args:
            position: 3D position [x, y, z].

        Returns:
            Clamped position within field bounds.
        """
        return np.array([
            np.clip(position[0], -self.half_length, self.half_length),
            np.clip(position[1], -self.half_width, self.half_width),
            position[2]  # Don't clamp z
        ])


class SoccerField:
    """3D Soccer field with PyBullet physics representation.

    Creates a complete soccer field with:
    - Ground plane
    - Boundary walls (4 sides with gaps for goals)
    - Goal structures (2 goals with posts and nets)
    - Visual markings (center circle, lines)

    Attributes:
        dimensions: Field dimensions configuration.
        physics_client: PyBullet physics client ID.
        body_ids: List of PyBullet body IDs for cleanup.
    """

    def __init__(
        self,
        physics_client: int,
        dimensions: FieldDimensions | None = None,
    ) -> None:
        """Initialize soccer field in PyBullet.

        Args:
            physics_client: PyBullet physics client ID.
            dimensions: Field dimensions, uses defaults if None.
        """
        self.physics_client = physics_client
        self.dimensions = dimensions or FieldDimensions()
        self.body_ids: List[int] = []

        self._create_field()

    def _create_field(self) -> None:
        """Create complete field geometry in PyBullet."""
        self._create_ground()
        self._create_walls()
        self._create_goals()

    def _create_ground(self) -> None:
        """Create ground plane with visual texture."""
        # Ground collision shape
        ground_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[
                self.dimensions.half_length + 5.0,
                self.dimensions.half_width + 5.0,
                0.1
            ],
            physicsClientId=self.physics_client
        )

        # Ground visual shape (green field)
        ground_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[
                self.dimensions.half_length + 5.0,
                self.dimensions.half_width + 5.0,
                0.1
            ],
            rgbaColor=[0.2, 0.6, 0.2, 1.0],  # Green
            physicsClientId=self.physics_client
        )

        ground_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, -0.1],
            physicsClientId=self.physics_client
        )

        self.body_ids.append(ground_id)

    def _create_walls(self) -> None:
        """Create boundary walls with gaps for goals."""
        dim = self.dimensions
        wall_color = [0.5, 0.5, 0.5, 1.0]  # Gray

        # Top wall (positive y)
        self._create_wall(
            position=[0, dim.half_width + dim.wall_thickness / 2, dim.wall_height / 2],
            half_extents=[dim.half_length, dim.wall_thickness / 2, dim.wall_height / 2],
            color=wall_color
        )

        # Bottom wall (negative y)
        self._create_wall(
            position=[0, -dim.half_width - dim.wall_thickness / 2, dim.wall_height / 2],
            half_extents=[dim.half_length, dim.wall_thickness / 2, dim.wall_height / 2],
            color=wall_color
        )

        # Left wall segments (negative x, with goal gap)
        wall_segment_length = (dim.width - dim.goal_width) / 2
        self._create_wall(
            position=[-dim.half_length - dim.wall_thickness / 2, dim.half_width - wall_segment_length / 2, dim.wall_height / 2],
            half_extents=[dim.wall_thickness / 2, wall_segment_length / 2, dim.wall_height / 2],
            color=wall_color
        )
        self._create_wall(
            position=[-dim.half_length - dim.wall_thickness / 2, -dim.half_width + wall_segment_length / 2, dim.wall_height / 2],
            half_extents=[dim.wall_thickness / 2, wall_segment_length / 2, dim.wall_height / 2],
            color=wall_color
        )

        # Right wall segments (positive x, with goal gap)
        self._create_wall(
            position=[dim.half_length + dim.wall_thickness / 2, dim.half_width - wall_segment_length / 2, dim.wall_height / 2],
            half_extents=[dim.wall_thickness / 2, wall_segment_length / 2, dim.wall_height / 2],
            color=wall_color
        )
        self._create_wall(
            position=[dim.half_length + dim.wall_thickness / 2, -dim.half_width + wall_segment_length / 2, dim.wall_height / 2],
            half_extents=[dim.wall_thickness / 2, wall_segment_length / 2, dim.wall_height / 2],
            color=wall_color
        )

    def _create_wall(
        self,
        position: List[float],
        half_extents: List[float],
        color: List[float]
    ) -> None:
        """Create a single wall segment.

        Args:
            position: Wall center position [x, y, z].
            half_extents: Half extents [x, y, z].
            color: RGBA color.
        """
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.physics_client
        )

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.physics_client
        )

        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.physics_client
        )

        self.body_ids.append(wall_id)

    def _create_goals(self) -> None:
        """Create goal structures for both teams."""
        dim = self.dimensions
        post_radius = 0.1

        # Blue goal (negative x) - Color: Blue
        self._create_goal(
            position=[-dim.half_length - dim.goal_depth / 2, 0, 0],
            color=[0.2, 0.2, 0.8, 1.0],
            post_radius=post_radius
        )

        # Red goal (positive x) - Color: Red
        self._create_goal(
            position=[dim.half_length + dim.goal_depth / 2, 0, 0],
            color=[0.8, 0.2, 0.2, 1.0],
            post_radius=post_radius
        )

    def _create_goal(
        self,
        position: List[float],
        color: List[float],
        post_radius: float
    ) -> None:
        """Create goal structure with posts.

        Args:
            position: Goal center position [x, y, z].
            color: RGBA color for goal posts.
            post_radius: Radius of goal posts.
        """
        dim = self.dimensions

        # Create back wall
        back_wall_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[dim.wall_thickness / 2, dim.goal_width / 2, dim.goal_height / 2],
            physicsClientId=self.physics_client
        )

        back_wall_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[dim.wall_thickness / 2, dim.goal_width / 2, dim.goal_height / 2],
            rgbaColor=color,
            physicsClientId=self.physics_client
        )

        back_wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=back_wall_shape,
            baseVisualShapeIndex=back_wall_visual,
            basePosition=position,
            physicsClientId=self.physics_client
        )

        self.body_ids.append(back_wall_id)

        # Create goal posts (left, right, top)
        post_positions = [
            [position[0], position[1] - dim.goal_width / 2, dim.goal_height / 2],  # Left post
            [position[0], position[1] + dim.goal_width / 2, dim.goal_height / 2],  # Right post
            [position[0], position[1], dim.goal_height],  # Top bar
        ]

        for post_pos in post_positions:
            post_shape = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=post_radius,
                height=post_radius * 2,
                physicsClientId=self.physics_client
            )

            post_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=post_radius,
                length=post_radius * 2,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],  # White posts
                physicsClientId=self.physics_client
            )

            post_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=post_shape,
                baseVisualShapeIndex=post_visual,
                basePosition=post_pos,
                physicsClientId=self.physics_client
            )

            self.body_ids.append(post_id)

    def reset(self) -> None:
        """Reset field state (currently no dynamic state)."""
        pass

    def cleanup(self) -> None:
        """Remove all field bodies from PyBullet simulation."""
        for body_id in self.body_ids:
            try:
                p.removeBody(body_id, physicsClientId=self.physics_client)
            except p.error:
                pass  # Body already removed

        self.body_ids.clear()
