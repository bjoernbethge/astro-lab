import logging
from typing import Any, Dict

import numpy as np
import pyvista as pv

from .astronomical_plotter import AstronomicalPyVistaPlotter as AstronomicalPlotter
from .coordinate_systems import (
    AstronomicalCoordinateConverter as CoordinateSystemVisualizer,
)
from .solar_system import (
    SolarSystemVisualizer,
    create_earth_moon_system,
    create_planetary_comparison,
    create_solar_system_scene,
)
from .stellar_visualization import StellarPyVistaVisualizer as StellarVisualization
from .tensor_bridge import AstronomicalPyVistaZeroCopyBridge

"""
PyVista Widgets for AstroLab - Scientific 3D Visualization
==========================================================

High-quality scientific 3D visualization using PyVista/VTK backend.
"""

logger = logging.getLogger(__name__)


def create_visualization(data: Any, plot_type: str = "scatter", **kwargs) -> pv.Plotter:
    """Create PyVista visualization with specified plot type.

    Args:
        data: Input data (TensorDict, numpy array, etc.)
        plot_type: Type of plot ('scatter', 'stellar', 'cosmic_web', 'mesh')
        **kwargs: Additional PyVista parameters

    Returns:
        PyVista plotter object
    """
    plotter = pv.Plotter(**kwargs.get("plotter_kwargs", {}))

    # Handle different data types
    if hasattr(data, "coordinates"):
        points = data.coordinates.cpu().numpy()
    elif hasattr(data, "pos"):
        points = data.pos.cpu().numpy()
    elif isinstance(data, np.ndarray):
        points = data
    else:
        points = np.array(data)

    # Create visualization based on plot type
    if plot_type == "scatter" or plot_type == "stellar":
        point_cloud = pv.PolyData(points)
        plotter.add_points(
            point_cloud,
            color=kwargs.get("color", "gold"),
            point_size=kwargs.get("point_size", 5),
            render_points_as_spheres=True,
        )

    elif plot_type == "cosmic_web":
        # Add points
        point_cloud = pv.PolyData(points)
        plotter.add_points(
            point_cloud,
            color=kwargs.get("node_color", "white"),
            point_size=kwargs.get("node_size", 3),
        )

        # Add edges if available
        if hasattr(data, "edge_index"):
            edges = data.edge_index.cpu().numpy().T
            for edge in edges:
                line = pv.Line(points[edge[0]], points[edge[1]])
                plotter.add_mesh(
                    line, color=kwargs.get("edge_color", "blue"), line_width=1
                )

    elif plot_type == "mesh":
        if hasattr(data, "faces"):
            mesh = pv.PolyData(points, data.faces.cpu().numpy())
        else:
            # Create mesh from points using Delaunay
            cloud = pv.PolyData(points)
            mesh = cloud.delaunay_3d()
        plotter.add_mesh(mesh, color=kwargs.get("color", "lightblue"))

    elif plot_type == "galaxy":
        # Specialized galaxy visualization
        point_cloud = pv.PolyData(points)
        plotter.add_points(
            point_cloud,
            scalars=kwargs.get("scalars"),
            cmap=kwargs.get("cmap", "viridis"),
            point_size=kwargs.get("point_size", 10),
            render_points_as_spheres=True,
        )

    # Add coordinate axes if requested
    if kwargs.get("show_axes", True):
        plotter.show_axes()

    # Set background
    plotter.background_color = kwargs.get("background_color", "black")

    return plotter


def create_multi_survey_visualization(tensordicts: Dict[str, Any], **kwargs):
    """
    Create multi-survey PyVista visualization.

    Args:
        tensordicts: Dictionary of {survey_name: tensordict}
        **kwargs: Visualization parameters

    Returns:
        PyVista plotter with multiple datasets
    """
    plotter = pv.Plotter(**kwargs)

    colors = ["gold", "blue", "red", "green", "purple", "orange"]

    for i, (survey, tensordict) in enumerate(tensordicts.items()):
        color = colors[i % len(colors)]

        # Create individual visualization
        viz = create_pyvista_visualization(tensordict, color=color, show=False)

        if hasattr(viz, "mesh"):
            plotter.add_mesh(viz.mesh, name=survey, color=color)
        elif hasattr(viz, "points"):
            plotter.add_points(viz.points, name=survey, color=color)

    return plotter


def create_astronomical_scene(coordinate_system: str = "galactocentric", **kwargs):
    """
    Create astronomical scene with coordinate system.

    Args:
        coordinate_system: Coordinate system to visualize
        **kwargs: Scene parameters

    Returns:
        PyVista plotter with coordinate system
    """
    plotter = pv.Plotter(**kwargs)

    # Add coordinate system visualization
    if coordinate_system == "galactocentric":
        # Create simple coordinate axes
        origin = np.array([0, 0, 0])
        x_axis = np.array([1000, 0, 0])
        y_axis = np.array([0, 1000, 0])
        z_axis = np.array([0, 0, 1000])

        plotter.add_lines([origin, x_axis], color="red", width=3, name="X-axis")
        plotter.add_lines([origin, y_axis], color="green", width=3, name="Y-axis")
        plotter.add_lines([origin, z_axis], color="blue", width=3, name="Z-axis")

    return plotter


def create_stellar_neighborhood(spatial_tensor: Any, **kwargs):
    """
    Create stellar neighborhood visualization.

    Args:
        spatial_tensor: SpatialTensorDict with stellar coordinates
        **kwargs: Visualization parameters

    Returns:
        Stellar visualization
    """
    # Create simple stellar neighborhood visualization
    if hasattr(spatial_tensor, "coordinates"):
        coords = spatial_tensor["coordinates"]
    else:
        coords = spatial_tensor

    # Create PyVista point cloud
    points = pv.PolyData(coords.cpu().numpy())

    # Add to plotter
    plotter = pv.Plotter(**kwargs)
    plotter.add_points(points, point_size=5, color="yellow", name="stars")

    return plotter


# Cross-backend functionality
def export_to_open3d(pyvista_data: Any, **kwargs):
    """
    Export PyVista data to Open3D format.

    Args:
        pyvista_data: PyVista mesh or point cloud
        **kwargs: Export options

    Returns:
        Open3D geometry object
    """
    from ..cross_backend_bridge import cross_bridge

    result = cross_bridge.pyvista_to_open3d(pyvista_data, **kwargs)
    return result.data


def export_to_blender(pyvista_data: Any, **kwargs):
    """
    Export PyVista data to Blender format.

    Args:
        pyvista_data: PyVista mesh or point cloud
        **kwargs: Export options

    Returns:
        Blender object
    """
    from ..cross_backend_bridge import cross_bridge

    result = cross_bridge.pyvista_to_blender(pyvista_data, **kwargs)
    return result.data


def apply_post_processing(pyvista_data: Any, effects: list, **kwargs):
    """
    Apply post-processing effects to PyVista data.

    Args:
        pyvista_data: PyVista data object
        effects: List of effects to apply
        **kwargs: Effect parameters

    Returns:
        Post-processed PyVista data
    """
    from ..cross_backend_bridge import BackendData, cross_bridge

    backend_data = BackendData(
        data=pyvista_data,
        metadata={},
        backend="pyvista",
        data_type="mesh" if hasattr(pyvista_data, "faces") else "point_cloud",
    )

    result = cross_bridge.apply_post_processing(backend_data, effects, **kwargs)
    return result.data


__all__ = [
    # Main API function
    "create_visualization",
    # Other functions
    "create_multi_survey_visualization",
    "create_astronomical_scene",
    "create_stellar_neighborhood",
    # Solar system functions
    "create_solar_system_scene",
    "create_earth_moon_system",
    "create_planetary_comparison",
    # Classes
    "AstronomicalPyVistaZeroCopyBridge",
    "StellarVisualization",
    "AstronomicalPlotter",
    "CoordinateSystemVisualizer",
    "SolarSystemVisualizer",
    # Cross-backend functions
    "export_to_open3d",
    "export_to_blender",
    "apply_post_processing",
]

# Keep legacy function for backward compatibility
create_pyvista_visualization = create_visualization
