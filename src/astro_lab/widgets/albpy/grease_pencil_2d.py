"""
Professional 2D Grease Pencil plotting for astronomical data.

This module provides radar charts, histograms, multi-panel plots, and comparison plots
using Grease Pencil or curve fallbacks for Blender 4.4 compatibility.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportGeneralTypeIssues=false

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from . import bpy

# Set environment variable for NumPy 2.x compatibility with bpy
os.environ["NUMPY_EXPERIMENTAL_ARRAY_API"] = "1"

# Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


class GreasePencil2DPlotter:
    """Professional 2D plotting using Grease Pencil v3 with curve fallbacks."""

    def __init__(self, name: str = "GP2D"):
        self.name = name
        self.created_objects = []

    def create_radar_chart(
        self,
        data: Dict[str, float],
        title: str = "Data Format Comparison",
        colors: Optional[List[List[float]]] = None,
        scale: float = 5.0,
    ) -> List[Any]:
        """Create radar chart like the data format comparison example."""
        if colors is None:
            colors = [
                [0.2, 0.6, 1.0],  # Blue
                [1.0, 0.4, 0.2],  # Orange
                [0.2, 0.8, 0.4],  # Green
            ]

        objects = []
        labels = list(data.keys())
        values = list(data.values())
        n_vars = len(labels)

        if n_vars < 3:
            print("Need at least 3 variables for radar chart")
            return []

        # Calculate angles for each axis
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)

        # Create background grid
        grid_objects = self._create_radar_grid(angles, labels, scale)
        objects.extend(grid_objects)

        # Create data polygon
        data_coords = []
        for i, (angle, value) in enumerate(zip(angles, values)):
            x = value * scale * np.cos(angle)
            y = value * scale * np.sin(angle)
            data_coords.append([x, y, 0])

        # Close the polygon
        data_coords.append(data_coords[0])

        # Create the data line
        data_curve = self._create_curve_line(
            data_coords, f"{title}_data", colors[0], line_width=0.02
        )
        if data_curve:
            objects.append(data_curve)

        # Add title text
        title_obj = self._create_text_object(title, [0, scale + 1, 0], 0.5)
        if title_obj:
            objects.append(title_obj)

        self.created_objects.extend(objects)
        return objects

    def create_multi_panel_plot(
        self,
        datasets: List[pl.DataFrame],
        panel_titles: List[str],
        plot_types: List[str],
        layout: Tuple[int, int] = (2, 2),
        panel_size: float = 3.0,
    ) -> List[Any]:
        """Create multi-panel plot like NSA galaxy analysis."""
        objects = []
        rows, cols = layout
        panel_spacing = panel_size + 0.5

        for i, (data, title, plot_type) in enumerate(
            zip(datasets, panel_titles, plot_types)
        ):
            if i >= rows * cols:
                break

            # Calculate panel position
            row = i // cols
            col = i % cols
            x_offset = (col - cols / 2 + 0.5) * panel_spacing
            y_offset = (rows / 2 - row - 0.5) * panel_spacing

            # Create individual panel
            panel_objects = self._create_panel(
                data, title, plot_type, [x_offset, y_offset, 0], panel_size
            )
            objects.extend(panel_objects)

        self.created_objects.extend(objects)
        return objects

    def create_comparison_plot(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        labels: List[str],
        title: str = "Mass Distribution Comparison",
        colors: Optional[List[List[float]]] = None,
    ) -> List[Any]:
        """Create comparison histogram plot."""
        if colors is None:
            colors = [[0.2, 0.6, 1.0], [1.0, 0.4, 0.2]]

        objects = []

        # Create histograms
        hist1_objects = self._create_histogram(
            data1, f"{title}_hist1", colors[0], offset_x=-2.0
        )
        hist2_objects = self._create_histogram(
            data2, f"{title}_hist2", colors[1], offset_x=2.0
        )

        objects.extend(hist1_objects)
        objects.extend(hist2_objects)

        # Add title and labels
        title_obj = self._create_text_object(title, [0, 4, 0], 0.6)
        if title_obj:
            objects.append(title_obj)

        self.created_objects.extend(objects)
        return objects

    def _create_radar_grid(
        self, angles: np.ndarray, labels: List[str], scale: float
    ) -> List[Any]:
        """Create radar chart background grid."""
        objects = []

        # Create concentric circles
        for radius in [0.2, 0.4, 0.6, 0.8, 1.0]:
            circle_points = []
            for angle in np.linspace(0, 2 * np.pi, 64):
                x = radius * scale * np.cos(angle)
                y = radius * scale * np.sin(angle)
                circle_points.append([x, y, 0])

            circle_curve = self._create_curve_line(
                circle_points,
                f"grid_circle_{radius}",
                [0.5, 0.5, 0.5],
                line_width=0.005,
            )
            if circle_curve:
                objects.append(circle_curve)

        # Create radial lines
        for i, angle in enumerate(angles):
            line_points = [
                [0, 0, 0],
                [scale * np.cos(angle), scale * np.sin(angle), 0],
            ]
            line_curve = self._create_curve_line(
                line_points, f"grid_line_{i}", [0.5, 0.5, 0.5], line_width=0.005
            )
            if line_curve:
                objects.append(line_curve)

            # Add labels
            label_pos = [
                scale * 1.1 * np.cos(angle),
                scale * 1.1 * np.sin(angle),
                0,
            ]
            label_obj = self._create_text_object(labels[i], label_pos, 0.3)
            if label_obj:
                objects.append(label_obj)

        return objects

    def _create_panel(
        self,
        data: pl.DataFrame,
        title: str,
        plot_type: str,
        position: List[float],
        size: float,
    ) -> List[Any]:
        """Create individual panel for multi-panel plot."""
        objects = []

        # Create panel border
        border_points = [
            [position[0] - size / 2, position[1] - size / 2, 0],
            [position[0] + size / 2, position[1] - size / 2, 0],
            [position[0] + size / 2, position[1] + size / 2, 0],
            [position[0] - size / 2, position[1] + size / 2, 0],
            [position[0] - size / 2, position[1] - size / 2, 0],
        ]

        border_curve = self._create_curve_line(
            border_points, f"{title}_border", [0.7, 0.7, 0.7], line_width=0.01
        )
        if border_curve:
            objects.append(border_curve)

        # Add title
        title_pos = [position[0], position[1] + size / 2 + 0.2, 0]
        title_obj = self._create_text_object(title, title_pos, 0.3)
        if title_obj:
            objects.append(title_obj)

        # Create plot based on type
        if plot_type == "scatter":
            # Extract x, y columns (assuming first two numeric columns)
            numeric_cols = data.select(
                pl.col("^.*$").filter(pl.col("^.*$").is_numeric())
            ).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                x_data = data[x_col].to_numpy()
                y_data = data[y_col].to_numpy()

                # Create scatter plot
                for x, y in zip(x_data, y_data):
                    point_pos = [
                        position[0]
                        + (x - x_data.min()) / (x_data.max() - x_data.min()) * size
                        - size / 2,
                        position[1]
                        + (y - y_data.min()) / (y_data.max() - y_data.min()) * size
                        - size / 2,
                        0,
                    ]
                    point_obj = self._create_point(point_pos, 0.01)
                    if point_obj:
                        objects.append(point_obj)

        return objects

    def _create_histogram(
        self,
        data: np.ndarray,
        name: str,
        color: List[float],
        bins: int = 20,
        offset_x: float = 0.0,
    ) -> List[Any]:
        """Create histogram visualization."""
        objects = []

        # Calculate histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize heights
        max_height = 2.0
        normalized_heights = hist / hist.max() * max_height if hist.max() > 0 else hist

        # Create bars
        bar_width = (bin_edges[1] - bin_edges[0]) * 0.8
        for i, (center, height) in enumerate(zip(bin_centers, normalized_heights)):
            if height > 0:
                bar_points = [
                    [offset_x + center - bar_width / 2, 0, 0],
                    [offset_x + center + bar_width / 2, 0, 0],
                    [offset_x + center + bar_width / 2, height, 0],
                    [offset_x + center - bar_width / 2, height, 0],
                    [offset_x + center - bar_width / 2, 0, 0],
                ]

                bar_curve = self._create_curve_line(
                    bar_points, f"{name}_bar_{i}", color, line_width=0.02
                )
                if bar_curve:
                    objects.append(bar_curve)

        return objects

    def _create_curve_line(
        self,
        points: List[List[float]],
        name: str,
        color: List[float],
        line_width: float = 0.01,
    ) -> Optional[Any]:
        """Create a curve line object."""
        try:
            # Create curve data
            curve_data = bpy.data.curves.new(name, type="CURVE")
            curve_data.dimensions = "3D"
            curve_data.resolution_u = 2

            # Create spline
            spline = curve_data.splines.new("POLY")
            spline.points.add(len(points) - 1)

            for i, point in enumerate(points):
                spline.points[i].co = (*point, 1)

            # Create object
            curve_obj = bpy.data.objects.new(name, curve_data)
            bpy.context.scene.collection.objects.link(curve_obj)

            # Create material
            material = self._create_emission_material(f"{name}_mat", color)
            if material:
                curve_obj.data.materials.append(material)

            # Set line width
            curve_data.bevel_depth = line_width

            return curve_obj

        except Exception as e:
            print(f"Failed to create curve line: {e}")
            return None

    def _create_point(self, position: List[float], size: float = 0.02) -> Optional[Any]:
        """Create a point object."""
        try:
            # Create mesh data
            mesh_data = bpy.data.meshes.new("Point")
            mesh_obj = bpy.data.objects.new("Point", mesh_data)

            # Create simple sphere
            bpy.context.collection.objects.link(mesh_obj)
            bpy.context.view_layer.objects.active = mesh_obj

            # Add sphere primitive
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=size, location=position, segments=8, ring_count=6
            )
            sphere = bpy.context.active_object

            # Create material
            material = self._create_emission_material("PointMaterial", [1.0, 1.0, 1.0])
            if material:
                sphere.data.materials.append(material)

            return sphere

        except Exception as e:
            print(f"Failed to create point: {e}")
            return None

    def _create_text_object(
        self, text: str, position: List[float], size: float = 0.5
    ) -> Optional[Any]:
        """Create a text object."""
        try:
            # Create text data
            text_data = bpy.data.curves.new("Text", type="FONT")
            text_data.body = text
            text_data.size = size

            # Create text object
            text_obj = bpy.data.objects.new("Text", text_data)
            text_obj.location = position
            bpy.context.scene.collection.objects.link(text_obj)

            # Create material
            material = self._create_emission_material("TextMaterial", [1.0, 1.0, 1.0])
            if material:
                text_obj.data.materials.append(material)

            return text_obj

        except Exception as e:
            print(f"Failed to create text object: {e}")
            return None

    def _create_emission_material(
        self, name: str, color: List[float], strength: float = 2.0
    ) -> Optional[Any]:
        """Create an emission material."""
        try:
            material = bpy.data.materials.new(name)
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # Clear default nodes
            nodes.clear()

            # Create emission shader
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (*color, 1.0)
            emission.inputs["Strength"].default_value = strength

            # Create output node
            output = nodes.new("ShaderNodeOutputMaterial")

            # Link nodes
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            return material

        except Exception as e:
            print(f"Failed to create emission material: {e}")
            return None

    def clear_objects(self) -> None:
        """Clear all created objects."""
        for obj in self.created_objects:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception as e:
                print(f"Failed to remove object {obj.name}: {e}")

        self.created_objects.clear()
