"""
AstroLab Training Module with Enhanced Tensor Integration
========================================================

PyTorch Lightning-based training with native SurveyTensor support.
"""

from .lightning_module import AstroLightningModule

# MLFlow logger available conditionally
try:
    from .mlflow_logger import MLFlowLogger

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFlowLogger = None
    MLFLOW_AVAILABLE = False
from .optuna_trainer import OptunaTrainer
from .trainer import AstroTrainer

# 🌟 Enhanced imports with tensor support
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
    "MLFlowLogger",
    # Data integration
    "create_astro_datamodule",
    "create_astro_dataloader",
    "AstroDataModule",
]
