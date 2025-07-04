"""
AstroLab Data Transforms
=======================

Collection of transforms for astronomical data processing.
Uses PyTorch Geometric and Astropy APIs directly.
"""

# Re-export PyTorch Geometric transforms directly
from torch_geometric.transforms import (
    Center,
    Compose,
    Delaunay,
    KNNGraph,
    LinearTransformation,
    NormalizeFeatures,
    RadiusGraph,
    RandomJitter,
    RandomRotate,
    RandomTranslate,
    ToDevice,
)

# Import our astronomical-specific transforms
from .astronomical import (
    AstronomicalFeatures,
    ExtinctionCorrection,
    GalacticCoordinateTransform,
    ProperMotionCorrection,
)
from .cosmic_web import (
    CosmicWebClassification,
    DensityFieldEstimation,
    FilamentDetection,
)
from .heterogeneous import (
    CrossMatchObjects,
    MultiSurveyMerger,
)

__all__ = [
    # PyTorch Geometric transforms (re-exported)
    "KNNGraph",
    "RadiusGraph",
    "Delaunay",
    "RandomJitter",
    "RandomRotate",
    "RandomTranslate",
    "NormalizeFeatures",
    "Compose",
    "ToDevice",
    "Center",
    "LinearTransformation",
    # Astronomical transforms
    "AstronomicalFeatures",
    "GalacticCoordinateTransform",
    "ProperMotionCorrection",
    "ExtinctionCorrection",
    # Cosmic Web transforms
    "CosmicWebClassification",
    "FilamentDetection",
    "DensityFieldEstimation",
    # Heterogeneous transforms
    "MultiSurveyMerger",
    "CrossMatchObjects",
]
