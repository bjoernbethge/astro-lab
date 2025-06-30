"""
AstroLab Blender Scene API
=========================

High-level API for initializing and managing astronomical scenes in Blender.
Uses advanced/core.py for DRY, maintainable code.
"""

from .advanced.core import initialize_astronomically_correct_scene


def initialize_scene(
    quality_preset: str = "high",
    coordinate_system: str = "icrs",
    distance_unit: str = "pc",
):
    """
    Initialize an astronomically correct scene with specified settings.
    Args:
        quality_preset: 'low', 'medium', 'high', 'ultra'
        coordinate_system: Astronomical coordinate system
        distance_unit: Distance unit
    Returns:
        Initialized visualization suite
    """
    return initialize_astronomically_correct_scene(
        quality_preset=quality_preset,
        coordinate_system=coordinate_system,
        distance_unit=distance_unit,
    )


__all__ = ["initialize_scene"]
