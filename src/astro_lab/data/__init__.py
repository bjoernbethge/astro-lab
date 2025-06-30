"""
AstroLab Data Module
====================

Unified interface for astronomical data loading and processing.
"""

# Import the unified datamodule interface
from .datamodules import (
    create_datamodule,
    AstroLightningDataset,
    AstroLightningNodeData,
)

# Import datasets
from .datasets import (
    SurveyGraphDataset,
    AstroPointCloudDataset,
    validate_survey,
    get_supported_surveys,
)

# Import preprocessors
from .preprocessors import get_preprocessor

# Import other utilities
from .cross_match import AstroCrossMatch
from .converters import create_spatial_tensor_from_survey, get_converter

__all__ = [
    # Main factory
    "create_datamodule",
    # Lightning wrappers
    "AstroLightningDataset",
    "AstroLightningNodeData",
    # Datasets
    "SurveyGraphDataset",
    "AstroPointCloudDataset",
    "validate_survey",
    "get_supported_surveys",
    # Utilities
    "get_preprocessor",
    "AstroCrossMatch",
    "create_spatial_tensor_from_survey",
    "get_converter",
]
