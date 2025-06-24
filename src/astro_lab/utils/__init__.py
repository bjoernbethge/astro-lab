"""
AstroLab Utilities - Core Utility Functions
==========================================

Provides core utility functions for configuration, visualization,
and data processing in the AstroLab framework.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# DO NOT import blender automatically - only when explicitly needed
# DO NOT import viz functions automatically - they load Blender modules
# Import them manually when needed: from astro_lab.utils.viz import ...
# Core dependencies - should always be available
import torch_geometric
import yaml

# Import core utility modules directly
from . import config, viz

logger = logging.getLogger(__name__)

# Import graph utilities
from .viz.graph import (
    calculate_graph_metrics,
    create_spatial_graph,
    spatial_distance_matrix,
)

# Base exports - always available (minimal to avoid Blender loading)
__all__ = [
    "config",
    "viz",
    "get_utils_info",
    "calculate_volume",
    "calculate_mean_density",
    "create_spatial_graph",
    "calculate_graph_metrics",
    "spatial_distance_matrix",
]


def get_utils_info() -> Dict[str, Any]:
    """Get information about available utilities."""
    # Check blender availability WITHOUT importing our blender module
    blender_available = False
    try:
        import bpy  # Direct check without loading our blender module

        blender_available = True
    except ImportError:
        pass

    return {
        "config_available": True,
        "viz_available": True,
        "blender_available": blender_available,
        "torch_geometric_available": True,
        "graph_available": True,
    }


def calculate_volume(coords: "np.ndarray | torch.Tensor") -> float:
    """Calculates the volume of the 3D cuboid enclosing all points."""
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    return float((x_max - x_min) * (y_max - y_min) * (z_max - z_min))


def calculate_mean_density(coords: "np.ndarray | torch.Tensor") -> float:
    """Calculates the mean density of objects in the volume."""
    n = coords.shape[0]
    vol = calculate_volume(coords)
    return n / vol if vol > 0 else float("nan")
