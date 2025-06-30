"""
AstroLab Blender Bridge
======================

Unified API for creating astronomical scenes, objects, and materials in Blender.
Composes from objects, materials, and scene modules for DRY, maintainable code.
"""

from . import materials, objects, scene


def create_astronomical_scene(**kwargs):
    """Initialize an astronomically correct scene."""
    return scene.initialize_scene(**kwargs)


def create_galaxy(**kwargs):
    """Create a galaxy object."""
    return objects.create_galaxy(**kwargs)


def create_hr_diagram(**kwargs):
    """Create a Hertzsprung-Russell diagram object."""
    return objects.create_hr_diagram(**kwargs)


def create_nebula(**kwargs):
    """Create a nebula object."""
    return objects.create_nebula(**kwargs)


def create_material(material_type: str = "iridescent", **kwargs):
    """Create a material by type (iridescent, glass, metallic, etc.)."""
    return materials.create_material(material_type=material_type, **kwargs)


__all__ = [
    "create_astronomical_scene",
    "create_galaxy",
    "create_hr_diagram",
    "create_nebula",
    "create_material",
]
