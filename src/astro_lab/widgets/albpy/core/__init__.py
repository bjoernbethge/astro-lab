"""
Core AlbPy Components
====================

Core functionality for scene setup, rendering, camera, and lighting.
"""

from .camera import setup_camera
from .data import list_available_surveys
from .lighting import setup_lighting
from .rendering import render_astronomical_scene, setup_rendering
from .scene import setup_scene

__all__ = [
    "setup_camera",
    "setup_lighting",
    "setup_rendering",
    "render_astronomical_scene",
    "setup_scene",
    "list_available_surveys",
]
