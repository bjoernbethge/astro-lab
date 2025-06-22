"""
Training Module - Neural Network Training Infrastructure
======================================================

Provides training infrastructure for neural network models including
Lightning modules, MLflow logging, and Optuna optimization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

# Core dependencies - should always be available
import lightning
import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn

from .lightning_module import AstroLightningModule
from .mlflow_logger import AstroMLflowLogger

# Import training components
from .trainer import AstroTrainer

# OptunaTrainer removed - functionality integrated into AstroTrainer
OptunaTrainer = None

__all__ = [
    "AstroTrainer",
    "AstroLightningModule",
    "AstroMLflowLogger",
]
