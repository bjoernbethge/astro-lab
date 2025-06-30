"""
PyVista Widgets for AstroLab - Scientific 3D Visualization
==========================================================

High-quality scientific 3D visualization using PyVista/VTK backend.
"""

import logging
from typing import Any, Dict

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

logger = logging.getLogger(__name__)


def create_pyvista_visualization(tensordict: Any, **kwargs):
    """
    Create PyVista visualization from TensorDict.

    Args:
        tensordict: AstroLab TensorDict
        **kwargs: PyVista-specific parameters

    Returns:
        PyVista plotter or data object
    """
    bridge = AstronomicalPyVistaZeroCopyBridge()

    # Route based on TensorDict type
    from astro_lab.tensors import (
        AnalysisTensorDict,
        PhotometricTensorDict,
        SpatialTensorDict,
    )

    if isinstance(tensordict, SpatialTensorDict):
        return bridge.spatial_to_pyvista(tensordict, **kwargs)
    elif isinstance(tensordict, PhotometricTensorDict):
        return bridge.photometric_to_pyvista(tensordict, **kwargs)
    elif isinstance(tensordict, AnalysisTensorDict):
        return bridge.analysis_to_pyvista(tensordict, **kwargs)
    else:
        # Generic coordinate conversion
        coords = (
            tensordict["coordinates"] if "coordinates" in tensordict else tensordict
        )
        return bridge.coordinates_to_pyvista(coords, **kwargs)


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
    visualizer = CoordinateSystemVisualizer()
    return visualizer.create_scene(coordinate_system, **kwargs)


def create_stellar_neighborhood(spatial_tensor: Any, **kwargs):
    """
    Create stellar neighborhood visualization.

    Args:
        spatial_tensor: SpatialTensorDict with stellar coordinates
        **kwargs: Visualization parameters

    Returns:
        Stellar visualization
    """
    stellar_viz = StellarVisualization()
    return stellar_viz.create_neighborhood(spatial_tensor, **kwargs)


__all__ = [
    # Main functions
    "create_pyvista_visualization",
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
]
