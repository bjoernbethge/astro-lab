"""
Lightning Mixins for AstroLab Models
===================================

Mixin classes to provide Lightning functionality to AstroLab models
without code duplication. Each mixin provides a specific aspect of
Lightning functionality.

Mixins:
- TrainingMixin: Provides training, validation, and test steps
- OptimizerMixin: Provides optimizer and scheduler configuration
- LossMixin: Provides task-specific loss functions
- MetricsMixin: Provides task-specific metrics
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor


class TrainingMixin:
    """Mixin providing training, validation, and test steps."""

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Generic training step for all AstroLab models."""
        # Robust: batch is a PyG Data or Batch object
        data = batch
        target = getattr(batch, "y", None)
        pred = self(data)
        loss = self.criterion(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """Generic validation step for all AstroLab models."""
        data = batch
        target = getattr(batch, "y", None)
        pred = self(data)
        loss = self.criterion(pred, target)
        self.log("val_loss", loss, prog_bar=True)
        if self.task.endswith("classification") or self.task.endswith("segmentation"):
            acc = (pred.argmax(dim=1) == target).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        elif self.task.endswith("regression") or self.task == "forecasting":
            mae = torch.abs(pred - target).mean()
            self.log("val_mae", mae, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """Generic test step for all AstroLab models."""
        data = batch
        target = getattr(batch, "y", None)
        pred = self(data)
        loss = self.criterion(pred, target)
        self.log("test_loss", loss)
        if self.task.endswith("classification") or self.task.endswith("segmentation"):
            acc = (pred.argmax(dim=1) == target).float().mean()
            self.log("test_acc", acc)
        elif self.task.endswith("regression") or self.task == "forecasting":
            mae = torch.abs(pred - target).mean()
            self.log("test_mae", mae)


class OptimizerMixin:
    """Mixin providing optimizer and scheduler configuration."""

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers for all AstroLab models."""
        from torch.optim import Adam, AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

        if self.optimizer_name == "adam":
            optimizer = Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
        elif self.scheduler_name == "onecycle":
            # Use a reasonable default for steps_per_epoch
            steps_per_epoch = 100  # Default value
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            )
        else:
            scheduler = None

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            return {"optimizer": optimizer}


class LossMixin:
    """Mixin providing task-specific loss functions."""

    def _setup_criterion(self):
        """Setup task-specific loss function."""
        if self.task in [
            "node_classification",
            "graph_classification",
            "time_series_classification",
            "point_classification",
            "node_segmentation",
            "point_segmentation",
        ]:
            self.criterion = nn.CrossEntropyLoss()
        elif self.task in [
            "node_regression",
            "graph_regression",
            "forecasting",
            "point_registration",
        ]:
            self.criterion = nn.MSELoss()
        elif self.task == "anomaly_detection":
            self.criterion = nn.BCELoss()
        else:
            # Default to MSE for unknown tasks
            self.criterion = nn.MSELoss()


class MetricsMixin:
    """Mixin providing task-specific metrics."""

    def compute_accuracy(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute accuracy for classification tasks."""
        return (pred.argmax(dim=1) == target).float().mean()

    def compute_mae(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Mean Absolute Error for regression tasks."""
        return torch.abs(pred - target).mean()

    def compute_rmse(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Root Mean Square Error for regression tasks."""
        return torch.sqrt(torch.mean((pred - target) ** 2))


class DeviceMixin:
    """Mixin providing device management."""

    def get_device(self, device: Optional[str] = None) -> torch.device:
        """Get torch device with automatic CUDA detection."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def move_to_device(self, *tensors, device: Optional[str] = None):
        """Move tensors to device."""
        target_device = self.get_device(device)
        return [t.to(target_device) if t is not None else None for t in tensors]


# Convenience mixin that combines all functionality
class AstroLightningMixin(
    TrainingMixin, OptimizerMixin, LossMixin, MetricsMixin, DeviceMixin
):
    """Combined mixin providing all Lightning functionality for AstroLab models."""

    def __init__(
        self,
        task: str,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """Initialize the mixin with common parameters."""
        self.task = task
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay
