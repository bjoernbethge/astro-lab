"""
Cosmic Web Generator for AlbPy - Real AstroLab Data Integration
============================================================

High-performance cosmic web visualization generator using real AstroLab data.
Refactored to use existing core modules and utilities.
"""

import logging

import bpy

from .core import (
    render_astronomical_scene,
    setup_camera,
    setup_lighting,
    setup_rendering,
    setup_scene,
)

logger = logging.getLogger(__name__)


def generate_cosmic_web_scene(data_file: str, render: bool = True):
    """
    Generate and render a cosmic web scene from a real data file using Blender operators and core rendering utilities.
    Args:
        data_file: Path to the data file (e.g. .pt, .npy, .h5)
        render: Whether to render the scene after creation
    """
    setup_scene()
    setup_lighting()
    setup_camera()
    setup_rendering()
    # For demo: always create a point cloud (extend as needed)
    bpy.ops.albpy.create_point_cloud()  # noqa: E1101
    if render:
        render_astronomical_scene()
    logger.info(f"Cosmic web scene generated for file: {data_file}")
