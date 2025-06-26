"""
Model Factory for AstroLab Models
================================

Factory functions to create the consolidated 4-model architecture:
- AstroNodeGNN: Node-level tasks
- AstroGraphGNN: Graph-level tasks
- AstroTemporalGNN: Temporal tasks
- AstroPointNet: Point cloud tasks
"""

import logging
from typing import Any, Dict, Optional, Union

from ..config import (
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
    list_presets,
)
from .astro_graph_gnn import AstroGraphGNN, create_astro_graph_gnn
from .astro_node_gnn import AstroNodeGNN, create_astro_node_gnn
from .astro_pointnet import AstroPointNet, create_astro_pointnet
from .astro_temporal_gnn import AstroTemporalGNN, create_astro_temporal_gnn

logger = logging.getLogger(__name__)


def create_model(
    model_type: str,
    num_features: int,
    num_classes: int,
    task: Optional[str] = None,
    **kwargs,
) -> Union[AstroNodeGNN, AstroGraphGNN, AstroTemporalGNN, AstroPointNet]:
    """
    Create a model based on model type with robust error handling and automatic task assignment.

    Args:
        model_type: Type of model ('node', 'graph', 'temporal', 'point')
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type (optional, will be auto-assigned if not provided)
        **kwargs: Additional model parameters

    Returns:
        Configured model instance

    Raises:
        ValueError: If model_type is unknown or parameters are invalid
    """
    # Validate model_type
    valid_model_types = ["node", "graph", "temporal", "point"]
    if model_type not in valid_model_types:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Valid types: {', '.join(valid_model_types)}"
        )

    # Auto-assign task if not provided
    if task is None:
        task_mapping = {
            "node": "node_classification",
            "graph": "graph_classification",
            "temporal": "time_series_classification",
            "point": "point_classification",
        }
        task = task_mapping[model_type]
        logger.info(f"Auto-assigned task '{task}' for model_type '{model_type}'")

    # Validate parameters
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    # Create model with appropriate task validation
    try:
        if model_type == "node":
            return create_astro_node_gnn(
                num_features=num_features,
                num_classes=num_classes,
                task=task,
                **kwargs,
            )
        elif model_type == "graph":
            return create_astro_graph_gnn(
                num_features=num_features,
                num_classes=num_classes,
                task=task,
                **kwargs,
            )
        elif model_type == "temporal":
            return create_astro_temporal_gnn(
                num_features=num_features,
                num_classes=num_classes,
                task=task,
                **kwargs,
            )
        elif model_type == "point":
            return create_astro_pointnet(
                num_features=num_features,
                num_classes=num_classes,
                task=task,
                **kwargs,
            )
    except Exception as e:
        # Provide more helpful error messages
        if "Unknown task" in str(e):
            valid_tasks = {
                "node": ["node_classification", "node_regression", "node_segmentation"],
                "graph": ["graph_classification", "graph_regression"],
                "temporal": [
                    "time_series_classification",
                    "forecasting",
                    "anomaly_detection",
                ],
                "point": [
                    "point_classification",
                    "point_segmentation",
                    "point_registration",
                ],
            }
            raise ValueError(
                f"Invalid task '{task}' for model_type '{model_type}'. "
                f"Valid tasks: {', '.join(valid_tasks[model_type])}"
            ) from e
        else:
            raise ValueError(f"Failed to create {model_type} model: {e}") from e

    # This should never be reached, but satisfies the type checker
    raise RuntimeError(
        f"Unexpected error: model_type '{model_type}' was validated but not handled"
    )


def create_model_from_config(
    config: ModelConfig,
) -> Union[AstroNodeGNN, AstroGraphGNN, AstroTemporalGNN, AstroPointNet]:
    """
    Create a model from a configuration object.

    Args:
        config: Model configuration object

    Returns:
        Configured model instance
    """
    if isinstance(config, AstroNodeGNNConfig):
        return create_astro_node_gnn(
            num_features=config.model_params.get("num_features", 64),
            num_classes=config.num_classes or 2,
            task=config.task,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            **config.model_params,
        )
    elif isinstance(config, AstroGraphGNNConfig):
        return create_astro_graph_gnn(
            num_features=config.model_params.get("num_features", 64),
            num_classes=config.num_classes or 2,
            task=config.task,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            pooling=config.pooling_type,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            **config.model_params,
        )
    elif isinstance(config, AstroTemporalGNNConfig):
        return create_astro_temporal_gnn(
            num_features=config.model_params.get("num_features", 64),
            num_classes=config.num_classes or 2,
            task=config.task,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            conv_type=config.model_params.get("conv_type", "gcn"),
            temporal_model=config.temporal_conv_type,
            sequence_length=config.sequence_length,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            use_batch_norm=config.model_params.get("use_batch_norm", True),
            **config.model_params,
        )
    elif isinstance(config, AstroPointNetConfig):
        return create_astro_pointnet(
            num_features=config.model_params.get("num_features", 3),
            num_classes=config.num_classes or 2,
            task=config.task,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            use_batch_norm=config.model_params.get("use_batch_norm", True),
            **config.model_params,
        )
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def create_model_from_preset(
    preset_name: str,
    num_features: Optional[int] = None,
    num_classes: Optional[int] = None,
    **kwargs,
) -> Union[AstroNodeGNN, AstroGraphGNN, AstroTemporalGNN, AstroPointNet]:
    """
    Create a model from a preset configuration.

    Args:
        preset_name: Name of the preset configuration
        num_features: Override number of features (optional)
        num_classes: Override number of classes (optional)
        **kwargs: Additional parameters to override

    Returns:
        Configured model instance
    """
    config = get_preset(preset_name)

    # Override parameters if provided
    if num_features is not None:
        config.model_params["num_features"] = num_features
    if num_classes is not None:
        config.num_classes = num_classes

    # Override any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.model_params[key] = value

    return create_model_from_config(config)


def create_model_for_task(
    task: str,
    num_features: int,
    num_classes: int,
    **kwargs,
) -> Union[AstroNodeGNN, AstroGraphGNN, AstroTemporalGNN, AstroPointNet]:
    """
    Create a model automatically based on task type.

    Args:
        task: Task type (e.g., 'node_classification', 'graph_regression')
        num_features: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional model parameters

    Returns:
        Configured model instance
    """
    model_type = get_model_type_for_task(task)
    return create_model(
        model_type=model_type,
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )


def create_model_from_dict(
    config_dict: Dict[str, Any],
) -> Union[AstroNodeGNN, AstroGraphGNN, AstroTemporalGNN, AstroPointNet]:
    """
    Create a model from a configuration dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Configured model instance
    """
    config = create_config_from_dict(config_dict)
    return create_model_from_config(config)


# Convenience functions for each model type
def create_node_gnn(
    num_features: int,
    num_classes: int,
    task: str = "node_classification",
    **kwargs,
) -> AstroNodeGNN:
    """Create an AstroNodeGNN model."""
    return create_astro_node_gnn(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )


def create_graph_gnn(
    num_features: int,
    num_classes: int,
    task: str = "graph_classification",
    **kwargs,
) -> AstroGraphGNN:
    """Create an AstroGraphGNN model."""
    return create_astro_graph_gnn(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )


def create_temporal_gnn(
    num_features: int,
    num_classes: int,
    task: str = "time_series_classification",
    **kwargs,
) -> AstroTemporalGNN:
    """Create an AstroTemporalGNN model."""
    return create_astro_temporal_gnn(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )


def create_pointnet(
    num_features: int,
    num_classes: int,
    task: str = "point_classification",
    **kwargs,
) -> AstroPointNet:
    """Create an AstroPointNet model."""
    return create_astro_pointnet(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )


# Model registry for easy access
MODEL_REGISTRY = {
    "node": AstroNodeGNN,
    "graph": AstroGraphGNN,
    "temporal": AstroTemporalGNN,
    "point": AstroPointNet,
}

FACTORY_REGISTRY = {
    "node": create_astro_node_gnn,
    "graph": create_astro_graph_gnn,
    "temporal": create_astro_temporal_gnn,
    "point": create_astro_pointnet,
}


def list_lightning_models() -> Dict[str, str]:
    """
    List all available Lightning models with descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        "astro_node_gnn": "AstroNodeGNN - Node-level tasks (classification, regression, segmentation)",
        "astro_graph_gnn": "AstroGraphGNN - Graph-level tasks (survey classification, cluster analysis)",
        "astro_temporal_gnn": "AstroTemporalGNN - Temporal tasks (lightcurves, time series)",
        "astro_pointnet": "AstroPointNet - Point cloud tasks (classification, segmentation, registration)",
        "node_classifier_small": "Small node classifier (32 dim, 2 layers, GCN)",
        "node_classifier_medium": "Medium node classifier (64 dim, 3 layers, GAT)",
        "node_classifier_large": "Large node classifier (128 dim, 4 layers, GIN)",
        "graph_classifier_small": "Small graph classifier (32 dim, 2 layers, GCN)",
        "graph_classifier_medium": "Medium graph classifier (64 dim, 3 layers, GAT)",
        "graph_classifier_large": "Large graph classifier (128 dim, 4 layers, GIN)",
        "temporal_classifier_small": "Small temporal classifier (32 dim, 2 layers, GRU)",
        "temporal_classifier_medium": "Medium temporal classifier (64 dim, 3 layers, LSTM)",
        "forecaster_medium": "Medium forecaster (64 dim, 3 layers, Transformer)",
        "point_classifier_small": "Small point classifier (32 dim, 2 layers, 512 points)",
        "point_classifier_medium": "Medium point classifier (64 dim, 3 layers, 1024 points)",
        "point_segmenter_medium": "Medium point segmenter (64 dim, 3 layers, 1024 points)",
    }
