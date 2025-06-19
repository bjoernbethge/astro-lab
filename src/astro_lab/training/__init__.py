"""
AstroLab Training Module

PyTorch Lightning + MLflow + Optuna integration for astronomical models.

Features:
- LightningModule wrapper for AstroLab models
- MLflow experiment tracking
- Optuna hyperparameter optimization
- Multi-GPU training support
- Automated checkpointing and logging
"""

from .lightning_module import AstroLightningModule
from .mlflow_logger import AstroMLflowLogger
from .optuna_trainer import OptunaTrainer
from .trainer import AstroTrainer

__all__ = [
    "AstroLightningModule",
    "AstroMLflowLogger",
    "OptunaTrainer",
    "AstroTrainer",
]
