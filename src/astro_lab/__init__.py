"""
Comprehensive Astronomical Data Analysis Framework

A modern Python framework for astronomical data analysis, machine learning, and visualization that combines specialized astronomy libraries with cutting-edge ML tools.
"""

import os
import warnings
from pathlib import Path

# Suppress NumPy 2.x compatibility warnings while keeping NumPy 2.x
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
    warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")

# Set astroML data directory to project root/data
_project_root = Path(__file__).parent.parent.parent
_data_dir = _project_root / "data"
_data_dir.mkdir(exist_ok=True)

# Set environment variable for astroML
os.environ["ASTROML_DATA"] = str(_data_dir)

# Import main modules - moved to top to fix E402
from astro_lab import data

# Import specific tensor classes instead of star imports
from astro_lab.tensors import (
    AstroTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
    SpectralTensorDict,
    SurveyTensorDict,
)

# Import specific utils instead of star imports
from astro_lab.utils import get_device, set_random_seed, setup_logging

__version__ = "0.1.0"
__all__ = [
    "data",
    "AstroTensorDict",
    "SpatialTensorDict",
    "PhotometricTensorDict",
    "SpectralTensorDict",
    "LightcurveTensorDict",
    "SurveyTensorDict",
    "setup_logging",
    "get_device",
    "set_random_seed",
]
