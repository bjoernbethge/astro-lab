"""
AstroLab Utilities
=================

Comprehensive utilities for astronomical data processing, visualization, and configuration management.
"""

import warnings
import logging
from typing import Dict, Any
import numpy as np
import torch

# Suppress all numpy warnings before any imports
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*numpy.core.multiarray.*")
warnings.filterwarnings("ignore", message=".*A module that was compiled using NumPy 1.x.*")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# Setup logging for utils
logger = logging.getLogger(__name__)

# Configuration management
try:
    from .config import (
        ConfigLoader,
        load_experiment_config,
        load_survey_config,
        setup_experiment_from_config,
        distribute_config_parameters,
        get_trainer_params,
        get_lightning_params,
        get_optuna_params,
        get_mlflow_params,
        get_data_params,
        validate_parameter_conflicts,
        print_parameter_distribution,
    )
    CONFIG_AVAILABLE = True
    logger.info("✅ Configuration management loaded")
except ImportError as e:
    CONFIG_AVAILABLE = False
    logger.info(f"ℹ️ Configuration management not available: {e}")

# Visualization utilities
VIZ_AVAILABLE = False
try:
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
        TORCH_GEOMETRIC_AVAILABLE,
    )
    VIZ_AVAILABLE = True
    logger.info("✅ Visualization utilities loaded")
except ImportError as e:
    VIZ_AVAILABLE = False
    logger.info(f"ℹ️ Visualization utilities not available: {e}")
except Exception as e:
    VIZ_AVAILABLE = False
    logger.warning(f"⚠️ Visualization utilities failed to load: {e}")

# Blender utilities with comprehensive error handling
BLENDER_AVAILABLE = False
BLENDER_ERROR = None

try:
    from .blender.lazy import is_blender_available, get_blender_error
    BLENDER_AVAILABLE = is_blender_available()
    BLENDER_ERROR = get_blender_error()
    logger.info("✅ Blender utilities loaded")
except ImportError as e:
    logger.info(f"ℹ️ Blender utilities not available: {e}")
except Exception as e:
    logger.warning(f"⚠️ Blender utilities failed to load: {e}")

# Base exports - always available
__all__ = [
    "CONFIG_AVAILABLE",
    "VIZ_AVAILABLE", 
    "BLENDER_AVAILABLE",
    "is_blender_available",
    "get_blender_error",
]

# Configuration exports
if CONFIG_AVAILABLE:
    __all__.extend([
        "ConfigLoader",
        "load_experiment_config",
        "load_survey_config", 
        "setup_experiment_from_config",
        "distribute_config_parameters",
        "get_trainer_params",
        "get_lightning_params",
        "get_optuna_params",
        "get_mlflow_params",
        "get_data_params",
        "validate_parameter_conflicts",
        "print_parameter_distribution",
    ])

# Visualization exports
if VIZ_AVAILABLE:
    __all__.extend([
        "PyVistaZeroCopyBridge",
        "BlenderZeroCopyBridge",
        "NumpyZeroCopyBridge", 
        "ZeroCopyBridge",
        "transfer_to_framework",
        "optimize_tensor_layout",
        "get_tensor_memory_info",
        "zero_copy_context",
        "pinned_memory_context",
        "TensorProtocol",
        "TNG50Visualizer",
        "load_tng50_gas",
        "load_tng50_stars",
        "quick_pyvista_plot",
        "quick_blender_import",
        "list_available_data",
        "create_spatial_graph",
        "calculate_graph_metrics",
        "spatial_distance_matrix",
        "TORCH_GEOMETRIC_AVAILABLE",
    ])

def get_utils_info() -> Dict[str, Any]:
    """Get information about available utilities."""
    blender_details = {"available": is_blender_available()}
    if not is_blender_available():
        blender_details["error"] = get_blender_error()
    
    torch_geometric_available = False
    if VIZ_AVAILABLE:
        try:
            torch_geometric_available = TORCH_GEOMETRIC_AVAILABLE
        except Exception:
            torch_geometric_available = False
    
    return {
        "config_available": CONFIG_AVAILABLE,
        "viz_available": VIZ_AVAILABLE,
        "blender_available": is_blender_available(),
        "blender_details": blender_details,
        "torch_geometric_available": torch_geometric_available,
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
