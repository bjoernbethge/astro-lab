"""
AstroLab Training Module with Enhanced Tensor Integration
========================================================

PyTorch Lightning-based training with native SurveyTensor support.
"""

from .lightning_module import AstroLightningModule
from .trainer import AstroTrainer

# MLFlow logger available conditionally
try:
    from .mlflow_logger import AstroMLflowLogger, setup_mlflow_experiment

    MLFLOW_AVAILABLE = True
except ImportError:
    AstroMLflowLogger = None
    setup_mlflow_experiment = None
    MLFLOW_AVAILABLE = False

# Optuna trainer available conditionally
try:
    from .optuna_trainer import OptunaTrainer

    OPTUNA_AVAILABLE = True
except ImportError:
    OptunaTrainer = None
    OPTUNA_AVAILABLE = False

# Data integration
try:
    from astro_lab.data import (
        AstroDataModule,
        create_astro_dataloader,
        create_astro_datamodule,
    )

    DATA_MODULE_AVAILABLE = True
except ImportError:
    DATA_MODULE_AVAILABLE = False
    create_astro_datamodule = None
    create_astro_dataloader = None
    AstroDataModule = None

__all__ = [
    "AstroTrainer",
    "AstroLightningModule",
    "OptunaTrainer",
    "AstroMLflowLogger",
    "setup_mlflow_experiment",
    # Data integration
    "create_astro_datamodule",
    "create_astro_dataloader",
    "AstroDataModule",
]
