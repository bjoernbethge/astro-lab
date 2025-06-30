"""
Default Configuration Values for AstroLab
========================================

Central configuration defaults for all AstroLab components.
"""

from typing import Any, Dict, Optional, Union

import torch

# Training defaults
TRAINING_DEFAULTS = {
    # Model architecture
    "model_type": "node",
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.1,
    "conv_type": "gcn",  # For GNN models
    "heads": 4,  # For GAT models
    "pooling": "mean",  # For graph-level tasks
    # Training hyperparameters
    "max_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "gradient_clip_val": 1.0,
    "early_stopping_patience": 10,
    # Optimizer settings
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_steps": 1000,
    # Hardware
    "accelerator": "auto",
    "devices": 1,
    "precision": "16-mixed",
    # Data loading
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    # MLflow
    "use_mlflow": True,
    "experiment_name": "astrolab_experiments",
    # Task defaults
    "task": "node_classification",
    "num_classes": 2,
}

# Model-specific defaults
MODEL_DEFAULTS = {
    "graph": {
        "conv_type": "gcn",
        "pooling": "mean",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "task": "graph_classification",
    },
    "node": {
        "conv_type": "gcn",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "task": "node_classification",
    },
    "temporal": {
        "rnn_type": "lstm",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "sequence_length": 10,
        "task": "time_series_classification",
    },
    "point": {
        "num_points": 1024,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "task": "point_classification",
    },
    "cosmic_web": {
        "conv_type": "gat",
        "heads": 4,
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.1,
        "multi_scale": True,
        "task": "structure_detection",
    },
}

# HPO search spaces - REASONABLE for 2025
HPO_SEARCH_SPACES = {
    # Architecture parameters - Conservative ranges
    "hidden_dim": [64, 128, 256],  # Removed 512 - too large for most GPUs
    "num_layers": {"low": 2, "high": 4},  # Reduced from 6 to 4
    "dropout": {"low": 0.0, "high": 0.3},  # Reduced from 0.5 to 0.3
    # Learning parameters - Reasonable ranges
    "learning_rate": {"low": 1e-4, "high": 5e-3, "log": True},  # Reduced upper bound
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
    "batch_size": [8, 16, 32, 64],  # Removed 128, 256 - too large
    # GNN-specific parameters - Conservative
    "conv_type": ["gcn", "gat", "sage"],  # Removed complex ones
    "heads": [1, 2, 4],  # Removed 8, 16 - too many heads
    "pooling": ["mean", "max"],  # Removed complex pooling
    # Point cloud parameters - Reasonable
    "k_neighbors": {"low": 8, "high": 20},  # Reduced from 32 to 20
    "point_cloud_layer_type": ["pointnet", "dynamic_edge"],  # Removed transformer
    # Training parameters - Conservative
    "gradient_clip_val": {"low": 0.5, "high": 1.5},  # Reduced range
    "warmup_steps": [0, 500, 1000],  # Removed 2000
}


def get_adaptive_hpo_search_space(
    gpu_memory_gb: Optional[float] = None, dataset_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get adaptive HPO search space based on hardware and dataset.

    Args:
        gpu_memory_gb: GPU memory in GB
        dataset_size: Number of samples in dataset

    Returns:
        Adaptive search space dictionary
    """
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Base search space
    adaptive_space = HPO_SEARCH_SPACES.copy()

    # Adapt based on GPU memory
    if gpu_memory_gb:
        if gpu_memory_gb < 8:
            # Low memory GPUs (RTX 3060, etc.)
            adaptive_space["hidden_dim"] = [32, 64, 128]
            adaptive_space["batch_size"] = [4, 8, 16]
            adaptive_space["num_layers"] = {"low": 2, "high": 3}
        elif gpu_memory_gb < 16:
            # Mid-range GPUs (RTX 4070, etc.)
            adaptive_space["hidden_dim"] = [64, 128, 256]
            adaptive_space["batch_size"] = [8, 16, 32]
            adaptive_space["num_layers"] = {"low": 2, "high": 4}
        else:
            # High-end GPUs (RTX 4090, A100, etc.)
            adaptive_space["hidden_dim"] = [128, 256, 512]
            adaptive_space["batch_size"] = [16, 32, 64, 128]
            adaptive_space["num_layers"] = {"low": 2, "high": 5}

    # Adapt based on dataset size
    if dataset_size:
        if dataset_size < 10000:
            # Small dataset - simpler models
            adaptive_space["hidden_dim"] = [32, 64, 128]
            adaptive_space["dropout"] = {"low": 0.0, "high": 0.2}
        elif dataset_size > 1000000:
            # Large dataset - can use more complex models
            adaptive_space["hidden_dim"] = [128, 256, 512]
            adaptive_space["batch_size"] = [32, 64, 128, 256]

    return adaptive_space


# Data module defaults
DATA_DEFAULTS = {
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "k_neighbors": 8,
    "max_samples": None,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    # Large-scale sampling
    "sampling_strategy": "none",  # "none", "neighbor", "cluster", "saint"
    "neighbor_sizes": [25, 10],
    "num_clusters": 1500,
    "saint_sample_coverage": 50,
    "saint_walk_length": 2,
    # Dynamic batching
    "enable_dynamic_batching": False,
    "min_batch_size": 1,
    "max_batch_size": 512,
    # Partitioning
    "partition_method": None,  # "metis", "random"
    "num_partitions": 4,
}

# Survey-specific training optimizations
SURVEY_TRAINING_DEFAULTS = {
    "gaia": {
        "batch_size": 512,
        "k_neighbors": 10,
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
    },
    "sdss": {
        "batch_size": 256,
        "k_neighbors": 12,
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
    },
    "nsa": {
        "batch_size": 128,
        "k_neighbors": 15,
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
    },
    "tng50": {
        "batch_size": 64,
        "k_neighbors": 20,
        "precision": "32-true",  # Simulation needs full precision
        "gradient_clip_val": 0.5,
    },
    "exoplanet": {
        "batch_size": 128,
        "k_neighbors": 8,
        "precision": "16-mixed",
        "gradient_clip_val": 1.0,
    },
}

# Task-specific defaults
TASK_DEFAULTS = {
    "node_classification": {
        "model_type": "node",
        "loss": "cross_entropy",
        "metrics": ["accuracy", "f1"],
    },
    "graph_classification": {
        "model_type": "graph",
        "loss": "cross_entropy",
        "metrics": ["accuracy", "f1"],
    },
    "node_regression": {
        "model_type": "node",
        "loss": "mse",
        "metrics": ["mse", "mae", "r2"],
    },
    "graph_regression": {
        "model_type": "graph",
        "loss": "mse",
        "metrics": ["mse", "mae", "r2"],
    },
    "time_series_classification": {
        "model_type": "temporal",
        "loss": "cross_entropy",
        "metrics": ["accuracy", "f1"],
    },
    "cosmic_web_classification": {
        "model_type": "cosmic_web",
        "loss": "cross_entropy",
        "metrics": ["accuracy", "f1", "iou"],
    },
}


def get_training_config(
    model_type: str = None, survey: str = None, task: str = None, **overrides
) -> dict:
    """
    Get training configuration with appropriate defaults.

    Args:
        model_type: Type of model
        survey: Survey name for data-specific settings
        task: Task type
        **overrides: Additional config overrides

    Returns:
        Complete training configuration
    """
    # Start with base defaults
    config = TRAINING_DEFAULTS.copy()

    # Apply model-specific defaults
    if model_type and model_type in MODEL_DEFAULTS:
        config.update(MODEL_DEFAULTS[model_type])

    # Apply survey-specific defaults
    if survey and survey in SURVEY_TRAINING_DEFAULTS:
        config.update(SURVEY_TRAINING_DEFAULTS[survey])

    # Apply task-specific defaults
    if task and task in TASK_DEFAULTS:
        task_config = TASK_DEFAULTS[task]
        config.update(task_config)
        # Ensure model type matches task
        if "model_type" in task_config:
            config["model_type"] = task_config["model_type"]

    # Apply user overrides
    config.update(overrides)

    return config


def get_hpo_search_space(param_name: str):
    """Get HPO search space for a parameter."""
    return HPO_SEARCH_SPACES.get(param_name, None)


def get_data_config(**overrides) -> dict:
    """Get data module configuration."""
    config = DATA_DEFAULTS.copy()
    config.update(overrides)
    return config
