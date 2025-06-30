"""
Model Factory for creating AstroLab models
==========================================

Factory functions for creating appropriate models based on task and data characteristics.
"""

from typing import Any, Dict, Optional, Union

from astro_lab.config import get_model_config, get_model_type_for_task

from .astro_graph_gnn import AstroGraphGNN, create_astro_graph_gnn
from .astro_node_gnn import AstroNodeGNN
from .astro_temporal_gnn import AstroTemporalGNN
from .astro_unified_point_cloud import AstroUnifiedPointCloud, create_unified_point_cloud_model
from .astro_cosmic_web_gnn import AstroCosmicWebGNN


# Model registry with aliases and constructors
MODEL_REGISTRY = {
    # Graph-level models
    "astro_graph_gnn": AstroGraphGNN,
    "graph_gnn": AstroGraphGNN,
    "graph": AstroGraphGNN,
    
    # Node-level models
    "astro_node_gnn": AstroNodeGNN,
    "node_gnn": AstroNodeGNN,
    "node": AstroNodeGNN,
    
    # Point cloud models (unified)
    "point_cloud": AstroUnifiedPointCloud,
    "point": AstroUnifiedPointCloud,  # Short alias
    "astro_point_cloud": AstroUnifiedPointCloud,
    "astro_pointnet": AstroUnifiedPointCloud,  # Legacy alias
    "astro_point_cloud_gnn": AstroUnifiedPointCloud,  # Legacy alias
    "unified_point_cloud": AstroUnifiedPointCloud,
    
    # Temporal models
    "astro_temporal_gnn": AstroTemporalGNN,
    "temporal_gnn": AstroTemporalGNN,
    "temporal": AstroTemporalGNN,
    
    # Specialized models
    "cosmic_web": AstroCosmicWebGNN,
    "astro_cosmic_web_gnn": AstroCosmicWebGNN,
}


def create_model(
    model_type: str,
    num_features: int,
    num_classes: int,
    task: Optional[str] = None,
    survey: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Create an AstroLab model based on type and configuration.
    
    Args:
        model_type: Type of model to create
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type (optional, for validation)
        survey: Survey name for survey-specific defaults
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model instance
        
    Examples:
        >>> # Create a graph-level GNN
        >>> model = create_model("graph", num_features=7, num_classes=3)
        
        >>> # Create a point cloud model
        >>> model = create_model("point_cloud", num_features=10, num_classes=4,
        ...                     architecture="hybrid", scale="large")
        
        >>> # Create a temporal GNN
        >>> model = create_model("temporal", num_features=5, num_classes=2,
        ...                     window_size=10)
    """
    
    # Normalize model type
    model_type = model_type.lower()
    
    # Get configuration from central config system
    config = get_model_config(model_type)
    config.update(kwargs)
    
    # Extract model parameters
    model_params = {
        "num_features": num_features,
        "num_classes": num_classes,
        "task": task or config.get("task", "classification"),
        "hidden_dim": config.get("hidden_dim", 128),
        "num_layers": config.get("num_layers", 3),
        "dropout": config.get("dropout", 0.1),
        "learning_rate": config.get("learning_rate", 0.001),
        "weight_decay": config.get("weight_decay", 0.01),
    }
    
    # Add model-specific parameters from config
    if model_type in ["graph", "node", "cosmic_web", "astro_graph_gnn", "astro_node_gnn"]:
        model_params["conv_type"] = config.get("conv_type", "gcn")
        if config.get("conv_type") == "gat":
            model_params["heads"] = config.get("heads", 4)
        if config.get("edge_dim"):
            model_params["edge_dim"] = config["edge_dim"]
    
    if model_type in ["graph", "astro_graph_gnn"]:
        model_params["pooling"] = config.get("pooling", "mean")
        # Add point cloud layer support
        if config.get("graph_layer_type") == "point_cloud":
            model_params["graph_layer_type"] = "point_cloud"
            model_params["point_cloud_config"] = config.get("point_cloud_config", {})
        # Add output head configuration
        if config.get("output_head"):
            model_params["output_head"] = config["output_head"]
            if config.get("num_harmonics"):
                model_params["num_harmonics"] = config["num_harmonics"]
    
    if model_type in ["temporal", "astro_temporal_gnn"]:
        model_params["rnn_type"] = config.get("rnn_type", "lstm")
        model_params["sequence_length"] = config.get("sequence_length", 10)
        model_params["temporal_layers"] = config.get("temporal_layers", 2)
    
    if model_type in ["cosmic_web", "astro_cosmic_web_gnn"]:
        model_params["multi_scale"] = config.get("multi_scale", True)
    
    # Special handling for point cloud models
    if model_type in ["point_cloud", "point", "astro_point_cloud", "astro_pointnet", 
                      "astro_point_cloud_gnn", "unified_point_cloud"]:
        # Use specialized factory for point cloud models
        scale = kwargs.get("scale", "medium")
        num_objects = kwargs.get("num_objects", 1_000_000)
        
        # Determine scale from num_objects if not specified
        if scale == "medium" and "num_objects" in kwargs:
            if num_objects < 100_000:
                scale = "small"
            elif num_objects < 1_000_000:
                scale = "medium"
            elif num_objects < 10_000_000:
                scale = "large"
            else:
                scale = "xlarge"
        
        # Add all config parameters
        model_params.update(config)
                
        return create_unified_point_cloud_model(
            num_features=num_features,
            num_classes=num_classes,
            task=task or "graph_classification",
            scale=scale,
            **model_params
        )
    
    # Check if model type exists
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(set(MODEL_REGISTRY.keys())))
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}"
        )
    
    # Get model class
    model_class = MODEL_REGISTRY[model_type]
    
    # Handle special factory functions
    if model_type in ["graph", "graph_gnn", "astro_graph_gnn"]:
        # Use graph GNN factory for variants
        variant = kwargs.pop("variant", "standard")
        if variant != "standard":
            return create_astro_graph_gnn(
                model_type=variant,
                num_features=num_features,
                num_classes=num_classes,
                task=task,
                **model_params
            )
    
    # Create model instance
    return model_class(**model_params)


def create_model_for_task(
    task: str,
    num_features: int,
    num_classes: int,
    survey: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Create a model for a specific task using central configuration.

    Args:
        task: Task type
        num_features: Number of input features
        num_classes: Number of output classes
        survey: Survey name for survey-specific defaults
        **kwargs: Additional model arguments

    Returns:
        Model instance
    """
    model_type = get_model_type_for_task(task)
    return create_model(
        model_type=model_type,
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        survey=survey,
        **kwargs,
    )


def get_model_config_dict(model_type: str, **overrides) -> Dict[str, Any]:
    """
    Get model configuration from central defaults.

    Args:
        model_type: Type of model
        **overrides: Configuration overrides

    Returns:
        Model configuration dictionary
    """
    config = get_model_config(model_type)
    config.update(overrides)
    return config


def get_model_type_for_task_name(task: str) -> str:
    """Get the model type for a given task."""
    return get_model_type_for_task(task)


def get_available_models() -> list:
    """Get list of available model types."""
    return list(MODEL_REGISTRY.keys())


def get_available_tasks() -> list:
    """Get list of available tasks."""
    # This is now delegated to config if needed
    from astro_lab.config import get_model_presets

    return list(get_model_presets().keys())


def auto_select_model(
    data_characteristics: Dict[str, Any],
    task: str,
) -> tuple[str, Dict[str, Any]]:
    """
    Automatically select the best model based on data characteristics.
    
    Args:
        data_characteristics: Dictionary with data properties
        task: Task name
        
    Returns:
        Tuple of (model_type, config)
    """
    
    # Extract characteristics
    num_nodes = data_characteristics.get("num_nodes", 0)
    num_edges = data_characteristics.get("num_edges", 0)
    num_graphs = data_characteristics.get("num_graphs", 1)
    has_positions = data_characteristics.get("has_positions", False)
    has_temporal = data_characteristics.get("has_temporal", False)
    is_dynamic = data_characteristics.get("is_dynamic", False)
    
    # Determine model type
    if has_temporal or is_dynamic:
        model_type = "temporal"
    elif has_positions and num_edges == 0:
        # Point cloud data without explicit edges
        model_type = "point_cloud"
    elif "graph" in task or num_graphs > 1:
        model_type = "graph"
    else:
        model_type = "node"
    
    # Get base config
    config = get_model_config(model_type)
    
    # Adjust for scale
    if model_type == "point_cloud":
        if num_nodes < 100_000:
            config["scale"] = "small"
        elif num_nodes < 1_000_000:
            config["scale"] = "medium"
        elif num_nodes < 10_000_000:
            config["scale"] = "large"
        else:
            config["scale"] = "xlarge"
            config["use_hierarchical"] = True
    
    # Adjust for sparsity
    if num_nodes > 0:
        avg_degree = (2 * num_edges) / num_nodes if num_edges > 0 else 0
        if avg_degree < 10:
            # Sparse graph - use attention mechanisms
            config["conv_type"] = "gat"
            config["heads"] = 8
        elif avg_degree > 100:
            # Dense graph - use simpler convolutions
            config["conv_type"] = "gcn"
    
    return model_type, config


# Convenience functions for specific model types
def create_graph_model(**kwargs) -> AstroGraphGNN:
    """Create a graph-level GNN model."""
    return create_model("graph", **kwargs)


def create_node_model(**kwargs) -> AstroNodeGNN:
    """Create a node-level GNN model."""
    return create_model("node", **kwargs)


def create_point_cloud_model(**kwargs) -> AstroUnifiedPointCloud:
    """Create a point cloud model."""
    return create_model("point_cloud", **kwargs)


def create_temporal_model(**kwargs) -> AstroTemporalGNN:
    """Create a temporal GNN model."""
    return create_model("temporal", **kwargs)
