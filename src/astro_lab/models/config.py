"""
Model Configuration and Presets
==============================

Configuration management for the consolidated 4-model architecture.
Provides presets for each model type and task-specific configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Base configuration for all AstroLab models."""

    # Model architecture
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"

    # Training parameters
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 1e-5

    # Task-specific
    task: str = "node_classification"
    num_classes: Optional[int] = None
    output_dim: Optional[int] = None

    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AstroNodeGNNConfig(ModelConfig):
    """Configuration for AstroNodeGNN (Node-Level Tasks)."""

    # GNN-specific
    conv_type: str = "gcn"  # gcn, gat, sage, gin
    use_batch_norm: bool = True
    use_residual: bool = True

    # Node classification specific
    node_features: Optional[List[str]] = None

    def __post_init__(self):
        if self.task not in [
            "node_classification",
            "node_regression",
            "node_segmentation",
        ]:
            raise ValueError(
                f"AstroNodeGNN only supports node-level tasks, got: {self.task}"
            )


@dataclass
class AstroGraphGNNConfig(ModelConfig):
    """Configuration for AstroGraphGNN (Graph-Level Tasks)."""

    # GNN-specific
    conv_type: str = "gcn"  # gcn, gat, sage, gin
    pooling_type: str = "mean"  # mean, max, sum, attention
    use_batch_norm: bool = True
    use_residual: bool = True

    # Graph classification specific
    graph_features: Optional[List[str]] = None

    def __post_init__(self):
        if self.task not in [
            "graph_classification",
            "graph_regression",
            "anomaly_detection",
        ]:
            raise ValueError(
                f"AstroGraphGNN only supports graph-level tasks, got: {self.task}"
            )


@dataclass
class AstroTemporalGNNConfig(ModelConfig):
    """Configuration for AstroTemporalGNN (Temporal Tasks)."""

    # Temporal-specific
    sequence_length: int = 10
    temporal_conv_type: str = "gru"  # gru, lstm, transformer
    use_attention: bool = True

    # Time series specific
    time_features: Optional[List[str]] = None

    def __post_init__(self):
        if self.task not in ["time_series_classification", "forecasting"]:
            raise ValueError(
                f"AstroTemporalGNN only supports temporal tasks, got: {self.task}"
            )


@dataclass
class AstroPointNetConfig(ModelConfig):
    """Configuration for AstroPointNet (Point Cloud Tasks)."""

    # PointNet-specific
    num_points: int = 1024
    use_tnet: bool = True
    use_feature_transform: bool = True

    # Point cloud specific
    point_features: Optional[List[str]] = None

    def __post_init__(self):
        if self.task not in [
            "point_classification",
            "point_registration",
            "point_segmentation",
        ]:
            raise ValueError(
                f"AstroPointNet only supports point cloud tasks, got: {self.task}"
            )


# Preset configurations for common use cases
PRESETS = {
    # AstroNodeGNN presets
    "node_classifier_small": AstroNodeGNNConfig(
        task="node_classification",
        hidden_dim=32,
        num_layers=2,
        conv_type="gcn",
        learning_rate=0.001,
    ),
    "node_classifier_medium": AstroNodeGNNConfig(
        task="node_classification",
        hidden_dim=64,
        num_layers=3,
        conv_type="gat",
        learning_rate=0.001,
    ),
    "node_classifier_large": AstroNodeGNNConfig(
        task="node_classification",
        hidden_dim=128,
        num_layers=4,
        conv_type="gin",
        learning_rate=0.0005,
    ),
    # AstroGraphGNN presets
    "graph_classifier_small": AstroGraphGNNConfig(
        task="graph_classification",
        hidden_dim=32,
        num_layers=2,
        conv_type="gcn",
        pooling_type="mean",
        learning_rate=0.001,
    ),
    "graph_classifier_medium": AstroGraphGNNConfig(
        task="graph_classification",
        hidden_dim=64,
        num_layers=3,
        conv_type="gat",
        pooling_type="attention",
        learning_rate=0.001,
    ),
    "graph_classifier_large": AstroGraphGNNConfig(
        task="graph_classification",
        hidden_dim=128,
        num_layers=4,
        conv_type="gin",
        pooling_type="attention",
        learning_rate=0.0005,
    ),
    # AstroTemporalGNN presets
    "temporal_classifier_small": AstroTemporalGNNConfig(
        task="time_series_classification",
        hidden_dim=32,
        num_layers=2,
        temporal_conv_type="gru",
        sequence_length=10,
        learning_rate=0.001,
        model_params={"num_features": 64, "conv_type": "gcn", "use_batch_norm": False},  # Disable batch norm for stability
    ),
    "temporal_classifier_medium": AstroTemporalGNNConfig(
        task="time_series_classification",
        hidden_dim=64,
        num_layers=3,
        temporal_conv_type="lstm",
        sequence_length=20,
        use_attention=True,
        learning_rate=0.001,
    ),
    "forecaster_medium": AstroTemporalGNNConfig(
        task="forecasting",
        hidden_dim=64,
        num_layers=3,
        temporal_conv_type="transformer",
        sequence_length=30,
        use_attention=True,
        learning_rate=0.001,
    ),
    # AstroPointNet presets
    "point_classifier_small": AstroPointNetConfig(
        task="point_classification",
        hidden_dim=32,
        num_layers=2,
        num_points=512,
        use_tnet=True,
        learning_rate=0.001,
    ),
    "point_classifier_medium": AstroPointNetConfig(
        task="point_classification",
        hidden_dim=64,
        num_layers=3,
        num_points=1024,
        use_tnet=True,
        use_feature_transform=True,
        learning_rate=0.001,
    ),
    "point_segmenter_medium": AstroPointNetConfig(
        task="point_segmentation",
        hidden_dim=64,
        num_layers=3,
        num_points=1024,
        use_tnet=True,
        use_feature_transform=True,
        learning_rate=0.001,
    ),
}


def get_preset(preset_name: str) -> ModelConfig:
    """Get a preset configuration by name."""
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    return PRESETS[preset_name]


def get_available_presets() -> List[str]:
    """Get list of available preset names."""
    return list(PRESETS.keys())


def list_presets() -> Dict[str, str]:
    """
    List all available model presets with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {
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


def create_config_from_dict(config_dict: Dict[str, Any]) -> ModelConfig:
    """Create a model configuration from a dictionary."""
    model_type = config_dict.get("model_type", "node")

    if model_type == "node":
        return AstroNodeGNNConfig(**config_dict)
    elif model_type == "graph":
        return AstroGraphGNNConfig(**config_dict)
    elif model_type == "temporal":
        return AstroTemporalGNNConfig(**config_dict)
    elif model_type == "point":
        return AstroPointNetConfig(**config_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Task type mappings
TASK_TO_MODEL = {
    # Node-level tasks
    "node_classification": "node",
    "node_regression": "node",
    "node_segmentation": "node",
    # Graph-level tasks
    "graph_classification": "graph",
    "graph_regression": "graph",
    "anomaly_detection": "graph",
    # Temporal tasks
    "time_series_classification": "temporal",
    "forecasting": "temporal",
    # Point cloud tasks
    "point_classification": "point",
    "point_registration": "point",
    "point_segmentation": "point",
}


def get_model_type_for_task(task: str) -> str:
    """Get the appropriate model type for a given task."""
    if task not in TASK_TO_MODEL:
        available = ", ".join(TASK_TO_MODEL.keys())
        raise ValueError(f"Unknown task '{task}'. Available: {available}")
    return TASK_TO_MODEL[task]
