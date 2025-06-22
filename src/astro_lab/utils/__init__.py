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
import yaml

# Import core utility modules directly
from . import config, viz

# DO NOT import blender automatically - only when explicitly needed
# DO NOT import viz functions automatically - they load Blender modules
# Import them manually when needed: from astro_lab.utils.viz import ...

# Check for optional dependencies
try:
    import torch_geometric

    TORCH_GEOMETRIC_AVAILABLE = True
    GRAPH_AVAILABLE = True  # For backward compatibility
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    GRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Base exports - always available (minimal to avoid Blender loading)
__all__ = [
    "config",
    "viz",
    # "blender",  # REMOVED - only import when explicitly needed
    # "list_available_data",  # REMOVED - loads viz modules with Blender
    "TORCH_GEOMETRIC_AVAILABLE",
    "GRAPH_AVAILABLE",  # For backward compatibility
    "get_utils_info",
    "calculate_volume",
    "calculate_mean_density",
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
        "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE,
        "graph_available": GRAPH_AVAILABLE,
    }


def calculate_volume(coords: "np.ndarray | torch.Tensor") -> float:
    """Berechnet das Volumen des 3D-Quaders, der alle Punkte umfasst."""
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    return float((x_max - x_min) * (y_max - y_min) * (z_max - z_min))


def calculate_mean_density(coords: "np.ndarray | torch.Tensor") -> float:
    """Berechnet die mittlere Dichte der Objekte im Volumen."""
    n = coords.shape[0]
    vol = calculate_volume(coords)
    return n / vol if vol > 0 else float("nan")
