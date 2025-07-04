"""Graph samplers for astronomical data.

Provides unified sampling strategies for graph construction from
astronomical data using PyTorch Geometric infrastructure.
"""

from .base import (
    AstroLabSampler,
    AstronomicalSamplerMixin,
    ClusterSamplerMixin,
    SpatialSamplerMixin,
)
from .cluster import ClusterSampler, DBSCANClusterSampler, HierarchicalClusterSampler
from .neighbor import (
    AdaptiveRadiusSampler,
    KNNSampler,
    NeighborSubgraphSampler,
    RadiusSampler,
)
from .saint import (
    AdaptiveGraphSAINTSampler,
    GraphSAINTEdgeSamplerWrapper,
    GraphSAINTNodeSamplerWrapper,
    GraphSAINTRandomWalkSamplerWrapper,
)

# Sampler registry
SAMPLER_REGISTRY = {
    # Neighbor-based samplers
    "knn": KNNSampler,
    "radius": RadiusSampler,
    "neighbor_subgraph": NeighborSubgraphSampler,
    "adaptive_radius": AdaptiveRadiusSampler,
    # Cluster-based samplers
    "cluster": ClusterSampler,
    "dbscan": DBSCANClusterSampler,
    "hierarchical": HierarchicalClusterSampler,
    # GraphSAINT samplers
    "saint_node": GraphSAINTNodeSamplerWrapper,
    "saint_edge": GraphSAINTEdgeSamplerWrapper,
    "saint_rw": GraphSAINTRandomWalkSamplerWrapper,
    "saint_adaptive": AdaptiveGraphSAINTSampler,
}


def get_sampler(sampler_type: str, config=None):
    """Get sampler for specified type.

    Args:
        sampler_type: Type of sampler
        config: Optional configuration dict

    Returns:
        Initialized sampler instance

    Raises:
        ValueError: If sampler type not supported
    """
    if sampler_type not in SAMPLER_REGISTRY:
        available = list(SAMPLER_REGISTRY.keys())
        raise ValueError(
            f"Sampler type '{sampler_type}' not supported. Available: {available}"
        )

    sampler_class = SAMPLER_REGISTRY[sampler_type]

    # Handle config properly
    if config is None:
        return sampler_class()
    else:
        return sampler_class(**config)


def list_available_samplers():
    """List all available sampler types.

    Returns:
        List of sampler type names
    """
    return list(SAMPLER_REGISTRY.keys())


__all__ = [
    # Base classes and mixins
    "AstroLabSampler",
    "SpatialSamplerMixin",
    "ClusterSamplerMixin",
    "AstronomicalSamplerMixin",
    # Neighbor-based samplers
    "KNNSampler",
    "RadiusSampler",
    "NeighborSubgraphSampler",
    "AdaptiveRadiusSampler",
    # Cluster-based samplers
    "ClusterSampler",
    "DBSCANClusterSampler",
    "HierarchicalClusterSampler",
    # GraphSAINT samplers
    "GraphSAINTNodeSamplerWrapper",
    "GraphSAINTEdgeSamplerWrapper",
    "GraphSAINTRandomWalkSamplerWrapper",
    "AdaptiveGraphSAINTSampler",
    # Factory functions
    "get_sampler",
    "list_available_samplers",
    "SAMPLER_REGISTRY",
]
