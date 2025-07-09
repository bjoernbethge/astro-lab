"""
AstroLab Data Transforms
=======================

Collection of transforms for astronomical data processing.
Uses PyTorch Geometric and Astropy APIs directly.
"""

# Import our astronomical-specific transforms
from .astronomical import (
    AstronomicalFeatures,
    ExtinctionCorrection,
    GalacticCoordinateTransform,
    MultiScaleSampling,
    ProperMotionCorrection,
)
from .cosmic_web import (
    CosmicWebClassification,
    DensityFieldEstimation,
    FilamentDetection,
    HaloIdentification,
    VoidDetection,
)
from .heterogeneous import (
    CrossMatchObjects,
    MultiSurveyMerger,
)

__all__ = [
    # Astronomical transforms
    "AstronomicalFeatures",
    "GalacticCoordinateTransform",
    "ProperMotionCorrection",
    "ExtinctionCorrection",
    "MultiScaleSampling",
    # Cosmic Web transforms
    "CosmicWebClassification",
    "FilamentDetection",
    "DensityFieldEstimation",
    "VoidDetection",
    "HaloIdentification",
    # Heterogeneous transforms
    "MultiSurveyMerger",
    "CrossMatchObjects",
]
