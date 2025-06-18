"""
Unified Blender utilities for astronomical data visualization.

This module provides a comprehensive, DRY system for creating advanced 3D
visualizations using Blender as the rendering engine.
"""

import warnings

# Blender availability check
try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    warnings.warn("Blender not available. Install Blender or run inside Blender.")

# Re-export everything when available
if BLENDER_AVAILABLE:
    try:
        from .core import *
        from .grease_pencil_2d import GreasePencil2DPlotter
        from .grease_pencil_3d import GreasePencil3DPlotter
    except ImportError as e:
        print(f"Warning: Some Blender modules failed to import: {e}")
        BLENDER_AVAILABLE = False

__all__ = [
    "BLENDER_AVAILABLE",
]

if BLENDER_AVAILABLE:
    __all__.extend(
        [
            "AstroPlotter",
            "FuturisticAstroPlotter",
            "GeometryNodesVisualizer",
            "GreasePencilPlotter",
            "GreasePencil2DPlotter",
            "GreasePencil3DPlotter",
            "reset_scene",
            "normalize_scene",
            "setup_scene",
            "create_material",
            "create_light",
            "setup_lighting_preset",
            "create_camera",
            "animate_camera",
            "create_camera_path",
            "create_astro_object",
            "setup_astronomical_scene",
            "render_scene",
            "setup_render_settings",
        ]
    )
