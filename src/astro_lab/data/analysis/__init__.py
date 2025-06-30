"""
Astronomical Data Analysis Module
=================================

Clean, focused analysis tools for astronomical data with Graph Neural Networks.
"""

from .cosmic_web import CosmicWebAnalyzer, analyze_cosmic_web
from .clustering import SpatialClustering
from .structures import FilamentDetector, StructureAnalyzer

__all__ = [
    "CosmicWebAnalyzer",
    "analyze_cosmic_web",
    "SpatialClustering",
    "FilamentDetector",
    "StructureAnalyzer",
]
