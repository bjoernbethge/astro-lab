"""
Comprehensive Astronomical Data Analysis Framework

A modern Python framework for astronomical data analysis, machine learning,
and visualization that combines specialized astronomy libraries with
cutting-edge ML tools.
"""

import os
import warnings
from pathlib import Path

# Import data module with relative import
from . import data

# Import config module with relative import
from .config import (
    SURVEY_CONFIGS,
    get_survey_config,
)

# Import specific tensor classes with relative imports
from .tensors import (
    AstroTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
)

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

__version__ = "0.1.0"
__all__ = [
    "data",
    "AstroTensorDict",
    "SpatialTensorDict",
    "PhotometricTensorDict",
    "LightcurveTensorDict",
    "get_survey_config",
    "SURVEY_CONFIGS",
]
