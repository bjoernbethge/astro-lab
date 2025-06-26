"""
AstroLab Utilities - Core Utility Functions
==========================================

Provides core utility functions for configuration and numerical helper functions.
"""

import logging
from typing import Any, Dict

import numpy as np
import torch

# Import core utility modules directly
from . import config

logger = logging.getLogger(__name__)

# Configuration utilities
from .config import ConfigLoader, get_survey_config, load_experiment_config


# Core utility functions
def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def calculate_volume(coords: "np.ndarray | torch.Tensor") -> float:
    """Calculates the volume of the 3D cuboid enclosing all points."""
    if isinstance(coords, torch.Tensor):
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


def get_utils_info() -> Dict[str, Any]:
    """Get information about available utilities."""
    return {
        "config_available": True,
        "visualization_in_widgets": True,
    }


# Base exports - clean and focused
__all__ = [
    # Configuration
    "ConfigLoader",
    "load_experiment_config",
    "get_survey_config",
    # Core utilities
    "get_utils_info",
    "calculate_volume",
    "calculate_mean_density",
    "get_device",
    "set_random_seed",
    "setup_logging",
]
