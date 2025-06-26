"""
Graph Building Module
====================

State-of-the-art graph construction for astronomical data.
"""

from .base import BaseGraphBuilder, GraphConfig
from .builders import (
    AdaptiveGraphBuilder,
    AstronomicalGraphBuilder,
    HeterogeneousGraphBuilder,
    KNNGraphBuilder,
    MultiScaleGraphBuilder,
    RadiusGraphBuilder,
    create_adaptive_graph,
    create_astronomical_graph,
    create_heterogeneous_graph,
    create_knn_graph,
    create_multiscale_graph,
    create_radius_graph,
)
from .pointcloud import (
    AdaptivePointCloudGraphBuilder,
    PointCloudGraphBuilder,
    create_adaptive_pointcloud_graph,
    create_pointcloud_graph,
)
from .advanced import (
    DynamicGraphBuilder,
    GeometricPriorGraphBuilder,
    GraphOfGraphsBuilder,
    TemporalGraphBuilder,
    create_dynamic_graph,
    create_geometric_prior_graph,
    create_hierarchical_graph,
    create_temporal_graph,
)

__all__ = [
    # Base classes
    "BaseGraphBuilder",
    "GraphConfig",
    # Standard Builders
    "KNNGraphBuilder",
    "RadiusGraphBuilder",
    "AstronomicalGraphBuilder",
    "MultiScaleGraphBuilder",
    "AdaptiveGraphBuilder",
    "HeterogeneousGraphBuilder",
    # Point Cloud Builders
    "PointCloudGraphBuilder",
    "AdaptivePointCloudGraphBuilder",
    # Advanced Builders
    "DynamicGraphBuilder",
    "GraphOfGraphsBuilder",
    "TemporalGraphBuilder",
    "GeometricPriorGraphBuilder",
    # Convenience functions
    "create_knn_graph",
    "create_radius_graph",
    "create_astronomical_graph",
    "create_multiscale_graph",
    "create_adaptive_graph",
    "create_heterogeneous_graph",
    "create_pointcloud_graph",
    "create_adaptive_pointcloud_graph",
    "create_dynamic_graph",
    "create_hierarchical_graph",
    "create_temporal_graph",
    "create_geometric_prior_graph",
]
