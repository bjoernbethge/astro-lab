"""
Centralized Graph Building for AstroLab
======================================

This module provides centralized graph construction from SurveyTensorDict data.
All graph building logic is consolidated here for consistency and maintainability.
"""

from .base import BaseGraphBuilder, GraphConfig
from .builders import (
    AstronomicalGraphBuilder,
    KNNGraphBuilder,
    RadiusGraphBuilder,
    create_astronomical_graph,
    create_knn_graph,
    create_radius_graph,
)

__all__ = [
    # Builders
    "AstronomicalGraphBuilder",
    "KNNGraphBuilder",
    "RadiusGraphBuilder",
    "BaseGraphBuilder",
    # Config
    "GraphConfig",
    # Convenience functions
    "create_astronomical_graph",
    "create_knn_graph",
    "create_radius_graph",
]
