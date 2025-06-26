"""
AstroLab Models
==============

Neural network models for astronomical data processing.

This package provides a consolidated 4-model architecture with Lightning Mixins:
- AstroNodeGNN: Node-level tasks (classification, regression, segmentation)
- AstroGraphGNN: Graph-level tasks (survey classification, cluster analysis)
- AstroTemporalGNN: Temporal tasks (lightcurves, time series)
- AstroPointNet: Point cloud tasks (classification, segmentation, registration)

All models use Lightning Mixins for consistent training, optimization, and metrics.
"""

from . import components, config, core, encoders, utils
from .components import (
    AstroLightningMixin,
    ClassificationHead,
    RegressionHead,
    create_mlp,
    create_output_head,
    get_activation,
)
from .config import (
    TASK_TO_MODEL,
    AstroGraphGNNConfig,
    AstroNodeGNNConfig,
    AstroPointNetConfig,
    AstroTemporalGNNConfig,
    ModelConfig,
    create_config_from_dict,
    get_available_presets,
    get_model_type_for_task,
    get_preset,
)
from .core import (
    FACTORY_REGISTRY,
    MODEL_REGISTRY,
    AstroGraphGNN,
    AstroNodeGNN,
    AstroPointNet,
    AstroTemporalGNN,
    create_astro_graph_gnn,
    create_astro_node_gnn,
    create_astro_pointnet,
    create_astro_temporal_gnn,
    create_graph_gnn,
    create_model,
    create_model_for_task,
    create_model_from_config,
    create_model_from_dict,
    create_model_from_preset,
    create_node_gnn,
    create_pointnet,
    create_temporal_gnn,
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
    # Configuration
    "ModelConfig",
    "AstroNodeGNNConfig",
    "AstroGraphGNNConfig",
    "AstroTemporalGNNConfig",
    "AstroPointNetConfig",
    "get_preset",
    "get_available_presets",
    "create_config_from_dict",
    "get_model_type_for_task",
    "TASK_TO_MODEL",
    # Components
    "AstroLightningMixin",
    "ClassificationHead",
    "RegressionHead",
    "create_mlp",
    "create_output_head",
    "get_activation",
    # Registries
    "MODEL_REGISTRY",
    "FACTORY_REGISTRY",
    # Submodules
    "components",
    "config",
    "core",
    "encoders",
    "utils",
]
