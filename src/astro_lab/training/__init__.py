"""
AstroLab Training Module (Lightning Edition)
==========================================

Simplified training utilities for Lightning-based astronomical ML.
The main training functionality is now handled by Lightning wrappers
in astro_lab.models.lightning.
"""

from .astro_trainer import AstroTrainer, train_model
from .config import TrainingConfig
from .mlflow_logger import LightningMLflowLogger, MLflowModelCheckpoint

__all__ = [
    # Main trainer
    "AstroTrainer",
    "train_model",
    # MLflow integration for Lightning
    "LightningMLflowLogger",
    "MLflowModelCheckpoint",
    # Basic configuration
    "TrainingConfig",
]
