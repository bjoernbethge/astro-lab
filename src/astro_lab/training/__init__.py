"""AstroLab Training Module - PyTorch Lightning based training utilities."""

from .train import train_model
from .trainer import AstroTrainer

__all__ = ["AstroTrainer", "train_model"]
