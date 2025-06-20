"""
AstroLab Utilities - Core Utility Functions
==========================================

Provides core utility functions for configuration, visualization,
and data processing in the AstroLab framework.
"""

import logging
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

# Import all utility modules directly
from . import config
from . import viz
from . import blender

# Import specific functions
from .viz import (
    PyVistaZeroCopyBridge,
    BlenderZeroCopyBridge,
    NumpyZeroCopyBridge,
    ZeroCopyBridge,
    transfer_to_framework,
    optimize_tensor_layout,
    get_tensor_memory_info,
    zero_copy_context,
    pinned_memory_context,
    TensorProtocol,
    TNG50Visualizer,
    load_tng50_gas,
    load_tng50_stars,
    quick_pyvista_plot,
    quick_blender_import,
    list_available_data,
    create_spatial_graph,
    calculate_graph_metrics,
    spatial_distance_matrix,
)

# Check for optional dependencies
try:
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
    GRAPH_AVAILABLE = True  # For backward compatibility
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    GRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Base exports - always available
__all__ = [
    "config",
    "viz", 
    "blender",
    "list_available_data",
    "TORCH_GEOMETRIC_AVAILABLE",
    "GRAPH_AVAILABLE"  # For backward compatibility
]

def get_utils_info() -> Dict[str, Any]:
    """Get information about available utilities."""
    return {
        "config_available": True,
        "viz_available": True,
        "blender_available": blender.bpy is not None,
        "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE,
        "graph_available": GRAPH_AVAILABLE,
    }

def calculate_volume(coords: 'np.ndarray | torch.Tensor') -> float:
    """Berechnet das Volumen des 3D-Quaders, der alle Punkte umfasst."""
    if hasattr(coords, 'detach'):
        coords = coords.detach().cpu().numpy()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    return float((x_max - x_min) * (y_max - y_min) * (z_max - z_min))


def calculate_mean_density(coords: 'np.ndarray | torch.Tensor') -> float:
    """Berechnet die mittlere Dichte der Objekte im Volumen."""
    n = coords.shape[0]
    vol = calculate_volume(coords)
    return n / vol if vol > 0 else float('nan')
