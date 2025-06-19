"""
Training utilities for AstroLab
===============================

PyTorch Lightning training components with state-of-the-art optimizations.
"""

from .lightning_module import AstroLightningModule
from .trainer import AstroTrainer
from .optuna_trainer import OptunaTrainer

# Import DataModule factory
try:
    from astro_lab.data.datamodules import create_astro_datamodule
except ImportError:
    # Fallback if datamodules not available
    create_astro_datamodule = None

__all__ = [
    "AstroLightningModule",
    "AstroTrainer", 
    "OptunaTrainer",
    "create_astro_datamodule",
]
