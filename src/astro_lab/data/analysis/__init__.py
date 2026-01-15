"""
Astronomical Data Analysis Module
=================================

Comprehensive analysis tools for astronomical data with TensorDict integration.
"""

from .clustering import SpatialClustering
from .cosmic_web import ScalableCosmicWebAnalyzer, analyze_cosmic_web_50m
from .structures import CosmicWebAnalyzer, FilamentDetector, StructureAnalyzer

__all__ = [
    # Clustering and spatial analysis
    "SpatialClustering",
    # Cosmic web analysis
    "ScalableCosmicWebAnalyzer",
    "CosmicWebAnalyzer",
    "analyze_cosmic_web_50m",
    # Structure detection
    "FilamentDetector",
    "StructureAnalyzer",
]
