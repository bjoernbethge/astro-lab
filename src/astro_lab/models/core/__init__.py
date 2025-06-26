"""
Core Models for AstroLab
=======================

The consolidated 4-model architecture for astronomical machine learning:
- AstroNodeGNN: Node-level tasks (classification, regression, segmentation)
- AstroGraphGNN: Graph-level tasks (survey classification, cluster analysis)
- AstroTemporalGNN: Temporal tasks (lightcurves, time series)
- AstroPointNet: Point cloud tasks (classification, segmentation, registration)

All models use Lightning Mixins for consistent training functionality.
"""

from .astro_graph_gnn import AstroGraphGNN, create_astro_graph_gnn
from .astro_node_gnn import AstroNodeGNN, create_astro_node_gnn
from .astro_pointnet import AstroPointNet, create_astro_pointnet
from .astro_temporal_gnn import AstroTemporalGNN, create_astro_temporal_gnn
from .factory import (
    FACTORY_REGISTRY,
    MODEL_REGISTRY,
    create_graph_gnn,
    create_model,
    create_model_for_task,
    create_model_from_config,
    create_model_from_dict,
    create_model_from_preset,
    create_node_gnn,
    create_pointnet,
    create_temporal_gnn,
    list_lightning_models,
    list_presets,
)

__all__ = [
    # Core models
    "AstroNodeGNN",
    "AstroGraphGNN",
    "AstroTemporalGNN",
    "AstroPointNet",
    # Factory functions
    "create_model",
    "create_model_from_config",
    "create_model_from_preset",
    "create_model_for_task",
    "create_model_from_dict",
    # Convenience functions
    "create_node_gnn",
    "create_graph_gnn",
    "create_temporal_gnn",
    "create_pointnet",
    # Individual factory functions
    "create_astro_node_gnn",
    "create_astro_graph_gnn",
    "create_astro_temporal_gnn",
    "create_astro_pointnet",
    # List functions
    "list_lightning_models",
    "list_presets",
    # Registries
    "MODEL_REGISTRY",
    "FACTORY_REGISTRY",
]
