"""
Advanced 3D Grease Pencil plotting for astronomical data.

This module provides 3D scatter plots, surface plots, trajectories, and vector fields
using curves as fallbacks for Blender 4.4 compatibility.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None
    mathutils = None


class GreasePencil3DPlotter:
    """Advanced 3D plotting using curves for Grease Pencil-style visualizations."""

    def __init__(self, name: str = "GP3D"):
        self.name = name
        self.created_objects = []

    def create_3d_scatter_plot(
        self,
        data: pl.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: Optional[str] = None,
        size_col: Optional[str] = None,
        title: str = "3D Galaxy Distribution",
        max_points: int = 1000,
    ) -> List[Any]:
        """Create 3D scatter plot like galaxy distribution visualization."""
        if not BLENDER_AVAILABLE:
            return []

        objects = []
        data_subset = data.head(max_points)

        # Get coordinates
        x_data = data_subset[x_col].to_numpy()
        y_data = data_subset[y_col].to_numpy()
        z_data = data_subset[z_col].to_numpy()

        # Normalize coordinates
        coords = np.column_stack([x_data, y_data, z_data])
        coords = self._normalize_coordinates(coords, target_scale=10.0)

        # Get colors and sizes
        colors = self._get_colors(data_subset, color_col)
        sizes = self._get_sizes(data_subset, size_col)

        # Create points
        for i, (coord, color, size) in enumerate(zip(coords, colors, sizes)):
            point_obj = self._create_3d_point(coord, size, color, f"{title}_point_{i}")
            if point_obj:
                objects.append(point_obj)

        # Create coordinate axes
        axes_objects = self._create_coordinate_axes(scale=12.0)
        objects.extend(axes_objects)

        # Add title
        title_obj = self._create_text_object(title, [0, 0, 12], 1.0)
        if title_obj:
            objects.append(title_obj)

        self.created_objects.extend(objects)
        return objects

    def create_3d_trajectory(
        self,
        trajectory_data: np.ndarray,
        title: str = "Orbital Trajectory",
        color: List[float] = [0.2, 0.8, 1.0],
        line_width: float = 0.05,
    ) -> List[Any]:
        """Create 3D trajectory visualization."""
        if not BLENDER_AVAILABLE:
            return []

        objects = []

        # Normalize trajectory
        normalized_traj = self._normalize_coordinates(trajectory_data, target_scale=8.0)

        # Create main trajectory curve
        trajectory_curve = self._create_curve_from_points(
            normalized_traj, f"{title}_trajectory", color, line_width
        )
        if trajectory_curve:
            objects.append(trajectory_curve)

        # Create velocity vectors at intervals
        vector_objects = self._create_velocity_vectors(
            normalized_traj, f"{title}_vectors", [1.0, 0.4, 0.2]
        )
        objects.extend(vector_objects)

        # Add start and end markers
        start_marker = self._create_3d_point(
            normalized_traj[0], 0.2, [0.2, 1.0, 0.2], f"{title}_start"
        )
        end_marker = self._create_3d_point(
            normalized_traj[-1], 0.2, [1.0, 0.2, 0.2], f"{title}_end"
        )

        if start_marker:
            objects.append(start_marker)
        if end_marker:
            objects.append(end_marker)

        self.created_objects.extend(objects)
        return objects

    def create_3d_vector_field(
        self,
        grid_points: np.ndarray,
        vectors: np.ndarray,
        title: str = "Gravitational Field",
        scale: float = 5.0,
        max_vectors: int = 200,
    ) -> List[Any]:
        """Create 3D vector field visualization."""
        if not BLENDER_AVAILABLE:
            return []

        objects = []

        # Subsample if too many vectors
        if len(grid_points) > max_vectors:
            indices = np.random.choice(len(grid_points), max_vectors, replace=False)
            grid_points = grid_points[indices]
            vectors = vectors[indices]

        # Normalize grid and vectors
        grid_norm = self._normalize_coordinates(grid_points, target_scale=scale)
        vector_magnitudes = np.linalg.norm(vectors, axis=1)
        max_magnitude = np.max(vector_magnitudes)

        # Create vector arrows
        for i, (point, vector, magnitude) in enumerate(
            zip(grid_norm, vectors, vector_magnitudes)
        ):
            # Normalize vector length
            if max_magnitude > 0:
                vector_norm = vector / max_magnitude * 0.5

            # Create arrow
            arrow_obj = self._create_vector_arrow(
                point, vector_norm, magnitude / max_magnitude, f"{title}_vector_{i}"
            )
            if arrow_obj:
                objects.append(arrow_obj)

        # Add title
        title_obj = self._create_text_object(title, [0, 0, scale + 2], 1.0)
        if title_obj:
            objects.append(title_obj)

        self.created_objects.extend(objects)
        return objects

    def create_3d_surface_plot(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_values: np.ndarray,
        title: str = "Surface Plot",
        color_map: str = "viridis",
    ) -> List[Any]:
        """Create 3D surface plot using curve wireframe."""
        if not BLENDER_AVAILABLE:
            return []

        objects = []

        # Normalize coordinates
        z_norm = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
        z_norm = z_norm * 3.0  # Scale height

        # Create wireframe lines along x-direction
        for i in range(x_grid.shape[0]):
            if i % 2 == 0:  # Skip every other line for clarity
                points = []
                for j in range(x_grid.shape[1]):
                    x = x_grid[i, j] * 0.1
                    y = y_grid[i, j] * 0.1
                    z = z_norm[i, j]
                    points.append([x, y, z])

                line_curve = self._create_curve_from_points(
                    np.array(points),
                    f"{title}_line_x_{i}",
                    [0.2, 0.6, 1.0],
                    0.02,
                )
                if line_curve:
                    objects.append(line_curve)

        # Create wireframe lines along y-direction
        for j in range(x_grid.shape[1]):
            if j % 2 == 0:  # Skip every other line for clarity
                points = []
                for i in range(x_grid.shape[0]):
                    x = x_grid[i, j] * 0.1
                    y = y_grid[i, j] * 0.1
                    z = z_norm[i, j]
                    points.append([x, y, z])

                line_curve = self._create_curve_from_points(
                    np.array(points),
                    f"{title}_line_y_{j}",
                    [1.0, 0.4, 0.2],
                    0.02,
                )
                if line_curve:
                    objects.append(line_curve)

        self.created_objects.extend(objects)
        return objects

    def _create_3d_point(
        self,
        position: np.ndarray,
        size: float,
        color: List[float],
        name: str,
    ) -> Optional[Any]:
        """Create 3D point as small sphere."""
        if not BLENDER_AVAILABLE:
            return None

        try:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=tuple(position))
            point_obj = bpy.context.active_object
            point_obj.name = name

            # Create material
            mat = self._create_emission_material(f"{name}_mat", color)
            if mat:
                point_obj.data.materials.append(mat)

            return point_obj

        except Exception as e:
            print(f"Failed to create 3D point: {e}")
            return None

    def _create_curve_from_points(
        self,
        points: np.ndarray,
        name: str,
        color: List[float],
        line_width: float = 0.02,
    ) -> Optional[Any]:
        """Create curve from array of points."""
        if not BLENDER_AVAILABLE:
            return None

        try:
            # Create curve
            curve_data = bpy.data.curves.new(name, type="CURVE")
            curve_data.dimensions = "3D"
            curve_data.bevel_depth = line_width

            # Create spline
            spline = curve_data.splines.new("NURBS")
            spline.points.add(len(points) - 1)

            # Set points
            for i, point in enumerate(points):
                spline.points[i].co = (*point, 1.0)

            # Create object
            curve_obj = bpy.data.objects.new(name, curve_data)
            bpy.context.collection.objects.link(curve_obj)

            # Create and apply material
            mat = self._create_emission_material(f"{name}_mat", color)
            if mat:
                curve_obj.data.materials.append(mat)

            return curve_obj

        except Exception as e:
            print(f"Failed to create curve: {e}")
            return None

    def _create_coordinate_axes(self, scale: float = 10.0) -> List[Any]:
        """Create coordinate axes."""
        objects = []
        axes_data = [
            ([0, 0, 0], [scale, 0, 0], [1.0, 0.2, 0.2], "X"),  # Red X
            ([0, 0, 0], [0, scale, 0], [0.2, 1.0, 0.2], "Y"),  # Green Y
            ([0, 0, 0], [0, 0, scale], [0.2, 0.2, 1.0], "Z"),  # Blue Z
        ]

        for start, end, color, label in axes_data:
            # Create axis line
            axis_curve = self._create_curve_from_points(
                np.array([start, end]), f"axis_{label}", color, 0.03
            )
            if axis_curve:
                objects.append(axis_curve)

            # Create axis label
            label_pos = [coord * 1.1 for coord in end]
            label_obj = self._create_text_object(label, label_pos, 0.5)
            if label_obj:
                objects.append(label_obj)

        return objects

    def _create_velocity_vectors(
        self, trajectory: np.ndarray, name_prefix: str, color: List[float]
    ) -> List[Any]:
        """Create velocity vectors along trajectory."""
        objects = []
        step = max(1, len(trajectory) // 10)  # 10 vectors max

        for i in range(0, len(trajectory) - 1, step):
            if i + 1 < len(trajectory):
                start = trajectory[i]
                direction = trajectory[i + 1] - trajectory[i]
                direction = direction / np.linalg.norm(direction) * 0.5

                arrow_obj = self._create_vector_arrow(
                    start, direction, 1.0, f"{name_prefix}_{i}"
                )
                if arrow_obj:
                    objects.append(arrow_obj)

        return objects

    def _create_vector_arrow(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        magnitude: float,
        name: str,
    ) -> Optional[Any]:
        """Create vector arrow."""
        if not BLENDER_AVAILABLE:
            return None

        try:
            end = start + direction * magnitude

            # Create arrow shaft
            shaft_curve = self._create_curve_from_points(
                np.array([start, end]), f"{name}_shaft", [1.0, 0.4, 0.2], 0.02
            )

            # Create arrowhead (simplified as small cone)
            bpy.ops.mesh.primitive_cone_add(
                radius1=0.05, depth=0.1, location=tuple(end)
            )
            arrowhead = bpy.context.active_object
            arrowhead.name = f"{name}_head"

            # Orient arrowhead
            if np.linalg.norm(direction) > 0:
                direction_norm = direction / np.linalg.norm(direction)
                # Simple rotation (could be improved)
                arrowhead.rotation_euler = (
                    0,
                    0,
                    math.atan2(direction_norm[1], direction_norm[0]),
                )

            # Apply material
            mat = self._create_emission_material(f"{name}_mat", [1.0, 0.4, 0.2])
            if mat:
                arrowhead.data.materials.append(mat)

            return shaft_curve  # Return the shaft as main object

        except Exception as e:
            print(f"Failed to create vector arrow: {e}")
            return None

    def _create_text_object(
        self, text: str, position: List[float], size: float = 0.5
    ) -> Optional[Any]:
        """Create text object."""
        if not BLENDER_AVAILABLE:
            return None

        try:
            bpy.ops.object.text_add(location=position)
            text_obj = bpy.context.active_object
            text_obj.data.body = text
            text_obj.data.size = size

            # Create material
            mat = self._create_emission_material(
                f"text_mat_{len(self.created_objects)}", [1.0, 1.0, 1.0]
            )
            if mat:
                text_obj.data.materials.append(mat)

            return text_obj

        except Exception as e:
            print(f"Failed to create text: {e}")
            return None

    def _create_emission_material(
        self, name: str, color: List[float], strength: float = 2.0
    ) -> Optional[Any]:
        """Create emission material."""
        if not BLENDER_AVAILABLE:
            return None

        if name in bpy.data.materials:
            return bpy.data.materials[name]

        try:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            # Output node
            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (300, 0)

            # Emission node
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (*color, 1.0)
            emission.inputs["Strength"].default_value = strength
            emission.location = (0, 0)

            # Link
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            return mat

        except Exception as e:
            print(f"Failed to create material: {e}")
            return None

    def _normalize_coordinates(
        self, coords: np.ndarray, target_scale: float = 10.0
    ) -> np.ndarray:
        """Normalize coordinates to target scale."""
        if coords.size == 0:
            return coords

        min_coord = np.min(coords, axis=0)
        max_coord = np.max(coords, axis=0)
        size = max_coord - min_coord

        # Avoid division by zero
        max_size = np.max(size)
        if max_size > 0:
            coords = (coords - min_coord) / max_size * target_scale
            coords = coords - np.mean(coords, axis=0)

        return coords

    def _get_colors(
        self, data: pl.DataFrame, color_col: Optional[str]
    ) -> List[List[float]]:
        """Get colors for data points."""
        if color_col and color_col in data.columns:
            values = data[color_col].to_numpy()
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            return [[val, 0.5, 1.0 - val] for val in normalized]
        else:
            return [[0.2, 0.8, 1.0]] * len(data)

    def _get_sizes(
        self, data: pl.DataFrame, size_col: Optional[str], base_size: float = 0.1
    ) -> List[float]:
        """Get sizes for data points."""
        if size_col and size_col in data.columns:
            values = data[size_col].to_numpy()
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            return [base_size + norm * base_size * 2 for norm in normalized]
        else:
            return [base_size] * len(data)

    def clear_objects(self) -> None:
        """Clear all created objects."""
        if not BLENDER_AVAILABLE:
            return

        for obj in self.created_objects:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except:
                pass

        self.created_objects.clear()
