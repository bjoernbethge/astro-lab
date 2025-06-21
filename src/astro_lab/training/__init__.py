"""
Training Module - Neural Network Training Infrastructure
======================================================

Provides training infrastructure for neural network models including
Lightning modules, MLflow logging, and Optuna optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Tuple

# Import training components
from .trainer import AstroTrainer
from .lightning_module import AstroLightningModule
from .mlflow_logger import AstroMLflowLogger

# Check for optional dependencies
try:
    import lightning
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
    from .optuna_trainer import OptunaTrainer
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaTrainer = None

__all__ = [
    "AstroTrainer",
    "AstroLightningModule", 
    "AstroMLflowLogger",
    "OptunaTrainer",
    "LIGHTNING_AVAILABLE",
    "MLFLOW_AVAILABLE",
    "OPTUNA_AVAILABLE"
]
