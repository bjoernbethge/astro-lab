"""
Model Presets and Model Configurations
=====================================

Centralized model presets and configuration dataclasses for AstroLab.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Base configuration for all AstroLab models."""

    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 1e-5
    task: str = "node_classification"
    num_classes: Optional[int] = None
    output_dim: Optional[int] = None
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AstroNodeGNNConfig(ModelConfig):
    conv_type: str = "gcn"
    use_batch_norm: bool = True
    use_residual: bool = True
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
    conv_type: str = "gcn"
    pooling_type: str = "mean"
    use_batch_norm: bool = True
    use_residual: bool = True
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
    sequence_length: int = 10
    temporal_conv_type: str = "gru"
    use_attention: bool = True
    time_features: Optional[List[str]] = None

    def __post_init__(self):
        if self.task not in ["time_series_classification", "forecasting"]:
            raise ValueError(
                f"AstroTemporalGNN only supports temporal tasks, got: {self.task}"
            )


@dataclass
class AstroPointNetConfig(ModelConfig):
    num_points: int = 1024
    use_tnet: bool = True
    use_feature_transform: bool = True
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
    "temporal_classifier_small": AstroTemporalGNNConfig(
        task="time_series_classification",
        hidden_dim=32,
        num_layers=2,
        temporal_conv_type="gru",
        sequence_length=10,
        learning_rate=0.001,
        model_params={"num_features": 64, "conv_type": "gcn", "use_batch_norm": False},
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

TASK_TO_MODEL = {
    "node_classification": "node",
    "node_regression": "node",
    "node_segmentation": "node",
    "graph_classification": "graph",
    "graph_regression": "graph",
    "anomaly_detection": "graph",
    "time_series_classification": "temporal",
    "forecasting": "temporal",
    "point_classification": "point",
    "point_registration": "point",
    "point_segmentation": "point",
}


def get_preset(preset_name: str) -> ModelConfig:
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    return PRESETS[preset_name]


def get_available_presets() -> List[str]:
    return list(PRESETS.keys())


def list_presets() -> Dict[str, str]:
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


def get_model_type_for_task(task: str) -> str:
    if task not in TASK_TO_MODEL:
        available = ", ".join(TASK_TO_MODEL.keys())
        raise ValueError(f"Unknown task '{task}'. Available: {available}")
    return TASK_TO_MODEL[task]


def get_model_config(
    model_name: str = None, preset: str = None, **kwargs
) -> Dict[str, Any]:
    if preset:
        config = get_preset(preset)
        return {
            "model_type": get_model_type_for_task(config.task),
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "task": config.task,
            **config.model_params,
            **kwargs,
        }
    else:
        return {
            "model_type": "node",
            "hidden_dim": 64,
            "num_layers": 3,
            "dropout": 0.1,
            **kwargs,
        }


def get_model_presets() -> Dict[str, Dict[str, Any]]:
    presets_dict = {}
    for name, config in PRESETS.items():
        presets_dict[name] = {
            "model_type": get_model_type_for_task(config.task),
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "task": config.task,
            **config.model_params,
        }
    return presets_dict
