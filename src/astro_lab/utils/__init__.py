"""
Graph and visualization utilities for astronomical data.
"""

import warnings
import logging
from typing import Dict, Any

# Suppress all numpy warnings before any imports
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*numpy.core.multiarray.*")
warnings.filterwarnings("ignore", message=".*A module that was compiled using NumPy 1.x.*")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# Setup logging for utils
logger = logging.getLogger(__name__)

# Graph utilities
GRAPH_AVAILABLE = False
try:
    from .graph import (
        TORCH_GEOMETRIC_AVAILABLE,
        calculate_graph_metrics,
        create_spatial_graph,
        spatial_distance_matrix,
    )
    GRAPH_AVAILABLE = True
    logger.info("✅ Graph utilities loaded")
except ImportError as e:
    logger.info(f"ℹ️ Graph utilities not available: {e}")

# Blender utilities with comprehensive error handling
BLENDER_AVAILABLE = False
BLENDER_ERROR = None

# Use centralized Blender lazy loading
from .blender_lazy import is_blender_available, get_blender_error

# TNG50 visualization utilities
TNG50_VIZ_AVAILABLE = False
try:
    from .tng50_viz import (
        TNG50Visualizer,
        load_tng50_gas,
        load_tng50_stars,
        quick_pyvista_plot,
        quick_blender_import,
        list_available_data,
    )
    TNG50_VIZ_AVAILABLE = True
    logger.info("✅ TNG50 visualization utilities loaded")
except ImportError as e:
    logger.info(f"ℹ️ TNG50 visualization utilities not available: {e}")

# Base exports - always available
__all__ = [
    "GRAPH_AVAILABLE", 
    "TNG50_VIZ_AVAILABLE",
    "is_blender_available",
    "get_blender_error",
]

# Conditional exports based on successful imports
if GRAPH_AVAILABLE:
    __all__.extend([
        "create_spatial_graph",
        "calculate_graph_metrics", 
        "spatial_distance_matrix",
        "TORCH_GEOMETRIC_AVAILABLE",
    ])

# Blender utilities loaded on demand

if TNG50_VIZ_AVAILABLE:
    __all__.extend([
        "TNG50Visualizer",
        "load_tng50_gas",
        "load_tng50_stars",
        "quick_pyvista_plot",
        "quick_blender_import",
        "list_available_data",
    ])

def get_utils_info() -> Dict[str, Any]:
    """Get information about available utilities."""
    blender_details = {"available": is_blender_available()}
    if not is_blender_available():
        blender_details["error"] = get_blender_error()
    
    torch_geometric_available = False
    if GRAPH_AVAILABLE:
        try:
            torch_geometric_available = TORCH_GEOMETRIC_AVAILABLE
        except Exception:
            torch_geometric_available = False
    
    return {
        "graph_available": GRAPH_AVAILABLE,
        "blender_available": is_blender_available(),
        "blender_details": blender_details,
        "torch_geometric_available": torch_geometric_available,
        "tng50_viz_available": TNG50_VIZ_AVAILABLE,
    }
