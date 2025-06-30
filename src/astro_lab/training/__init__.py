"""
AstroLab Training Module
========================

High-level training interface for astronomical ML models with integrated
hardware optimizations and monitoring.
"""

from astro_lab.training.astro_trainer import AstroTrainer
from astro_lab.training.trainer_core import TrainingCore
from astro_lab.training.trainer_utils import (
    detect_hardware,
    log_training_info,
    setup_cuda_graphs,
    setup_device,
    setup_mixed_precision,
    setup_torch_compile,
    validate_model_inputs,
)

__all__ = [
    # Main trainer
    "AstroTrainer",
    # Core training
    "TrainingCore",
    # Utilities
    "detect_hardware",
    "setup_device",
    "setup_mixed_precision",
    "setup_torch_compile",
    "setup_cuda_graphs",
    "validate_model_inputs",
    "log_training_info",
]


# Version info
__version__ = "3.0.0"
