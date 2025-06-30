"""
AstroLab Blender Objects API
===========================

High-level API for creating astronomical objects (galaxies, stars, nebulae, HR diagrams) in Blender.
Uses advanced geometry nodes and core features for DRY, maintainable code.
"""

from .. import bpy  # type: ignore
from .advanced.geometry_nodes import ProceduralAstronomy


def create_galaxy(
    galaxy_type: str = "spiral",
    num_stars: int = 50000,
    radius: float = 20.0,
    center=None,
) -> bpy.types.Object:
    """
    Create a procedural galaxy object using geometry nodes.
    Args:
        galaxy_type: 'spiral', 'elliptical', or 'irregular'
        num_stars: Number of stars
        radius: Galaxy radius
        center: Blender Vector or tuple (default: (0,0,0))
    Returns:
        Blender object
    """
    if center is None:
        from mathutils import Vector

        center = Vector((0, 0, 0))
    return ProceduralAstronomy.create_galaxy_structure(
        center=center, galaxy_type=galaxy_type, num_stars=num_stars, radius=radius
    )


def create_hr_diagram(
    stellar_data,
    scale_factor: float = 1.0,
) -> bpy.types.Object:
    """
    Create a 3D Hertzsprung-Russell diagram from stellar data.
    Args:
        stellar_data: List of dicts with 'temperature', 'luminosity', 'mass'
        scale_factor: Scaling for axes
    Returns:
        Blender object
    """
    return ProceduralAstronomy.create_hr_diagram_3d(
        stellar_data, scale_factor=scale_factor
    )


def create_nebula(
    nebula_type: str = "emission",
    size: float = 15.0,
    density: float = 0.2,
    center=None,
) -> bpy.types.Object:
    """
    Create a nebula object (emission, planetary, etc.)
    Args:
        nebula_type: 'emission', 'planetary', etc.
        size: Nebula size
        density: Nebula density
        center: Blender Vector or tuple (default: (0,0,0))
    Returns:
        Blender object
    """
    # For now, use galaxy structure as placeholder; extend with nebula logic as needed
    if center is None:
        from mathutils import Vector

        center = Vector((0, 0, 0))
    return ProceduralAstronomy.create_galaxy_structure(
        center=center, galaxy_type="irregular", num_stars=10000, radius=size
    )


__all__ = [
    "create_galaxy",
    "create_hr_diagram",
    "create_nebula",
]
