"""
PyTorch Lightning Module for AstroLab Models

Unified Lightning wrapper for all AstroLab model types with automatic
loss function selection and metric tracking.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from ..models import ALCDEFTemporalGNN, AstroPhotGNN, AstroSurveyGNN


class AstroLightningModule(LightningModule):
    """Lightning wrapper for AstroLab models with automatic optimization."""

    def __init__(
        self,
        model: Union[AstroSurveyGNN, ALCDEFTemporalGNN, AstroPhotGNN],
        task_type: str = "classification",  # classification, regression, multi_task
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",  # cosine, onecycle
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def _multi_task_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Multi-task loss with automatic weighting."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for key in predictions:
            if key in targets:
                if "classification" in key:
                    loss = F.cross_entropy(predictions[key], targets[key])
                else:
                    loss = F.mse_loss(predictions[key], targets[key])
                total_loss = total_loss + loss
        return total_loss

    def forward(
        self, batch: Dict[str, Any]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass."""
        if hasattr(self.model, "extract_survey_features"):
            # AstroSurveyGNN
            return self.model(
                x=batch["x"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_embeddings=False,
            )
        elif hasattr(self.model, "lightcurve_encoder"):
            # ALCDEFTemporalGNN
            return self.model(
                lightcurve=batch["lightcurve"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_embeddings=False,
            )
        elif hasattr(self.model, "component_heads"):
            # AstroPhotGNN
            return self.model(
                x=batch["x"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_components=True,
            )
        else:
            # Generic fallback
            return self.model(batch["x"], batch["edge_index"])

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        predictions = self(batch)
        targets = batch["y"]

        # Handle multi-task vs single-task
        if isinstance(predictions, dict) and isinstance(targets, dict):
            loss = self._multi_task_loss(predictions, targets)
        else:
            # Single-task loss
            if self.task_type == "classification":
                loss = F.cross_entropy(predictions, targets)  # type: ignore
                # Classification metrics
                acc = (predictions.argmax(dim=1) == targets).float().mean()  # type: ignore
                self.log("train_acc", acc, on_step=True, on_epoch=True)
            elif self.task_type == "regression":
                loss = F.mse_loss(predictions, targets)  # type: ignore
                # Regression metrics
                mae = F.l1_loss(predictions, targets)  # type: ignore
                self.log("train_mae", mae, on_step=True, on_epoch=True)
            else:
                loss = F.mse_loss(predictions, targets)  # type: ignore

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        predictions = self(batch)
        targets = batch["y"]

        # Handle multi-task vs single-task
        if isinstance(predictions, dict) and isinstance(targets, dict):
            loss = self._multi_task_loss(predictions, targets)
        else:
            # Single-task loss
            if self.task_type == "classification":
                loss = F.cross_entropy(predictions, targets)  # type: ignore
                # Classification metrics
                acc = (predictions.argmax(dim=1) == targets).float().mean()  # type: ignore
                self.log("val_acc", acc, on_step=False, on_epoch=True)
            elif self.task_type == "regression":
                loss = F.mse_loss(predictions, targets)  # type: ignore
                # Regression metrics
                mae = F.l1_loss(predictions, targets)  # type: ignore
                self.log("val_mae", mae, on_step=False, on_epoch=True)
            else:
                loss = F.mse_loss(predictions, targets)  # type: ignore

        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step."""
        predictions = self(batch)
        targets = batch["y"]

        # Handle multi-task vs single-task
        if isinstance(predictions, dict) and isinstance(targets, dict):
            loss = self._multi_task_loss(predictions, targets)
        else:
            # Single-task loss
            if self.task_type == "classification":
                loss = F.cross_entropy(predictions, targets)  # type: ignore
                # Classification metrics
                acc = (predictions.argmax(dim=1) == targets).float().mean()  # type: ignore
                self.log("test_acc", acc, on_step=False, on_epoch=True)
            elif self.task_type == "regression":
                loss = F.mse_loss(predictions, targets)  # type: ignore
                # Regression metrics
                mae = F.l1_loss(predictions, targets)  # type: ignore
                self.log("test_mae", mae, on_step=False, on_epoch=True)
            else:
                loss = F.mse_loss(predictions, targets)  # type: ignore

        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Union[AdamW, Dict[str, Any]]:
        """Configure optimizer and scheduler."""
        # Optimizer
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Scheduler
        if self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        elif self.scheduler == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_epochs / self.max_epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prediction step."""
        return self(batch)


__all__ = ["AstroLightningModule"]
