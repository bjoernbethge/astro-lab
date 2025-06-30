"""
Core Models for AstroLab
=========================

Core neural network models with TensorDict integration and astronomical optimization.
"""

# Import all core models
from .base_model import AstroBaseModel
from .astro_graph_gnn import AstroGraphGNN
from .astro_node_gnn import AstroNodeGNN
from .astro_pointnet import AstroPointNet
from .astro_point_cloud_gnn import AstroPointCloudGNN
from .astro_temporal_gnn import AstroTemporalGNN
from .astro_cosmic_web_gnn import AstroCosmicWebGNN
from .factory import (
    create_model,
    create_model_for_task,
    get_available_models,
    get_available_tasks,
    get_model_type_for_task,
)

__all__ = [
    # Base model
    "AstroBaseModel",

    # Core models
    "AstroGraphGNN",
    "AstroNodeGNN",
    "AstroPointNet",
    "AstroPointCloudGNN",
    "AstroTemporalGNN",
    "AstroCosmicWebGNN",

    # Factory functions
    "create_model",
    "create_model_for_task",
    "get_available_models",
    "get_available_tasks",
    "get_model_type_for_task",
]
