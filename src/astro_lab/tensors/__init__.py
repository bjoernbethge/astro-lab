"""AstroLab TensorDict Module - Specialized TensorDicts for astronomical data.

This module provides TensorDict subclasses optimized for different types
of astronomical data, built on top of the official TensorDict library.
"""

# Base class
from .analysis import AnalysisTensorDict
from .base import AstroTensorDict

# Domain-specific classes
from .cosmology import CosmologyTensorDict
from .image import ImageTensorDict
from .lightcurve import LightcurveTensorDict

# Mixins for common functionality
from .mixins import (
    CoordinateConversionMixin,
    FeatureExtractionMixin,
    NormalizationMixin,
    ValidationMixin,
)
from .photometric import PhotometricTensorDict

# Core specialized classes
from .spatial import SpatialTensorDict
from .survey import SurveyTensorDict

__all__ = [
    "AstroTensorDict",
    "SpatialTensorDict",
    "PhotometricTensorDict",
    "ImageTensorDict",
    "LightcurveTensorDict",
    "CosmologyTensorDict",
    "SurveyTensorDict",
    "NormalizationMixin",
    "FeatureExtractionMixin",
    "CoordinateConversionMixin",
    "ValidationMixin",
    "AnalysisTensorDict",
]
