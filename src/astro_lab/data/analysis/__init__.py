"""
Astronomical Data Analysis Module
=================================

Comprehensive analysis tools for astronomical data with TensorDict integration.
"""

# Import autoencoders for analysis purposes
from astro_lab.models import BaseAutoencoder, PointCloudAutoencoder

from .clustering import SpatialClustering, analyze_with_autoencoder
from .cosmic_web import ScalableCosmicWebAnalyzer, analyze_cosmic_web_50m
from .structures import CosmicWebAnalyzer, FilamentDetector, StructureAnalyzer

__all__ = [
    # Clustering and spatial analysis
    "SpatialClustering",
    "analyze_with_autoencoder",
    # Cosmic web analysis
    "ScalableCosmicWebAnalyzer",
    "CosmicWebAnalyzer",
    "analyze_cosmic_web_50m",
    # Structure detection
    "FilamentDetector",
    "StructureAnalyzer",
    # Autoencoders for analysis
    "BaseAutoencoder",
    "PointCloudAutoencoder",
]
