"""
Base Lightning Mixin for AstroLab Models
=======================================

Provides common Lightning functionality that can be mixed into any AstroLab model.
Optimized for PyTorch Lightning 2.x and modern GPUs.
"""

import logging
from typing import Any, Dict, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MulticlassF1Score

logger = logging.getLogger(__name__)


class AstroLabLightningMixin(L.LightningModule):
    """
    Base Lightning mixin for all AstroLab models.

    This mixin provides:
    - Automatic training/validation/test loops
    - Task-specific loss functions and metrics
    - Optimizer and scheduler configuration
    - Model summary logging
    - Compatible with all AstroLab model architectures
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        task: str = "classification",
        num_classes: Optional[int] = None,
        loss_function: Optional[str] = None,
        min_lr: float = 1e-6,
        **kwargs,
    ):
        """
        Initialize Lightning mixin.

        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            optimizer: Optimizer name ('adamw', 'adam', 'sgd', 'rmsprop')
            scheduler: Scheduler name ('cosine', 'step', 'onecycle', 'none', 'exponential')
            warmup_epochs: Number of warmup epochs
            task: Task type ('classification', 'regression', 'period_detection', 'shape_modeling')
            num_classes: Number of classes for classification
            loss_function: Override loss function ('mse', 'cross_entropy', 'l1', 'huber', 'focal')
            min_lr: Minimum learning rate for schedulers
            **kwargs: Additional arguments passed to parent
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.warmup_epochs = warmup_epochs
        self.task = task
        self.num_classes = num_classes
        self.min_lr = min_lr

        # Setup task-specific loss function
        self.loss_fn = self._setup_loss_function(loss_function)

        # Setup metrics
        self._setup_metrics()

    def _setup_loss_function(self, loss_function: Optional[str] = None):
        """Setup task-appropriate loss function."""
        if loss_function:
            # Explicit loss function override
            if loss_function == "mse":
                return nn.MSELoss()
            elif loss_function == "cross_entropy":
                return nn.CrossEntropyLoss()
            elif loss_function == "l1":
                return nn.L1Loss()
            elif loss_function == "huber":
                return nn.HuberLoss()
            elif loss_function == "focal":
                # Focal Loss für unbalancierte Datensätze
                from torchvision.ops import sigmoid_focal_loss

                return lambda pred, target: sigmoid_focal_loss(
                    pred,
                    F.one_hot(target, num_classes=self.num_classes).float(),
                    reduction="mean",
                )
            else:
                logger.warning(
                    f"Unknown loss function {loss_function}, using task default"
                )

        # Task-based default
        if self.task == "classification":
            return nn.CrossEntropyLoss()
        elif self.task == "regression":
            return nn.MSELoss()
        elif self.task == "period_detection":
            return nn.L1Loss()  # Better for period estimation
        elif self.task == "shape_modeling":
            return nn.MSELoss()  # For harmonic coefficients
        else:
            logger.warning(f"Unknown task {self.task}, using MSE loss")
            return nn.MSELoss()

    def _setup_metrics(self):
        """Setup training, validation, and test metrics."""
        # Set default num_classes if not provided
        if self.num_classes is None:
            self.num_classes = 2  # Default to binary classification

        if self.task == "classification":
            # Classification metrics
            self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

            # F1 Score für bessere Evaluation
            self.train_f1 = MulticlassF1Score(num_classes=self.num_classes)
            self.val_f1 = MulticlassF1Score(num_classes=self.num_classes)
            self.test_f1 = MulticlassF1Score(num_classes=self.num_classes)

        elif self.task == "regression":
            # Regression metrics
            self.train_mse = MeanSquaredError()
            self.train_mae = MeanAbsoluteError()
            self.val_mse = MeanSquaredError()
            self.val_mae = MeanAbsoluteError()
            self.test_mse = MeanSquaredError()
            self.test_mae = MeanAbsoluteError()

    def forward(self, batch):
        """Forward pass - handles different batch formats."""
        # Extract features from batch
        if hasattr(batch, "x"):
            return self.model(batch) if hasattr(self, "model") else self(batch)
        elif isinstance(batch, dict) and "x" in batch:
            return (
                self.model(batch["x"]) if hasattr(self, "model") else self(batch["x"])
            )
        else:
            # Assume batch is the input directly
            return (
                self.model(batch) if hasattr(self, "model") else super().forward(batch)
            )

    def _extract_predictions_and_targets(self, batch, outputs):
        """Extract predictions and targets from batch and model outputs."""
        # Handle different output formats from AstroLab models
        if isinstance(outputs, dict):
            predictions = outputs.get(
                "predictions", outputs.get("output", outputs.get("logits", outputs))
            )
        else:
            predictions = outputs

        # Extract targets from batch
        targets = None
        if hasattr(batch, "y") and batch.y is not None:
            targets = batch.y
        elif isinstance(batch, dict) and "y" in batch:
            targets = batch["y"]
        elif isinstance(batch, dict) and "targets" in batch:
            targets = batch["targets"]
        elif isinstance(batch, (list, tuple)) and len(batch) > 1:
            targets = batch[1]

        # Robust mask handling for node-level predictions
        if hasattr(batch, "train_mask") and targets is not None:
            # Get the appropriate mask
            mask = None
            if self.training and hasattr(batch, "train_mask"):
                mask = batch.train_mask
            elif (
                hasattr(self, "trainer")
                and hasattr(self.trainer, "validating")
                and self.trainer.validating
                and hasattr(batch, "val_mask")
            ):
                mask = batch.val_mask
            elif hasattr(batch, "test_mask"):
                mask = batch.test_mask
            else:
                mask = batch.train_mask  # fallback

            # Apply mask if it exists and has True values
            if mask is not None and mask.any():
                # Ensure mask and targets have same size
                if mask.size(0) == targets.size(0):
                    targets = targets[mask]

                    # Handle predictions - check if they need masking too
                    if predictions is not None:
                        if predictions.size(0) == mask.size(0):
                            # Node-level predictions, apply same mask
                            predictions = predictions[mask]
                        elif predictions.size(0) == mask.sum():
                            # Predictions already masked, keep as-is
                            pass
                        else:
                            # Size mismatch - try to fix it
                            if predictions.size(0) == 1 and mask.sum() > 1:
                                # Single prediction, expand to match targets
                                predictions = (
                                    predictions.expand(mask.sum().item(), -1)
                                    if predictions.dim() > 1
                                    else predictions.expand(mask.sum().item())
                                )
                            elif predictions.size(0) > mask.sum():
                                # Too many predictions, take first N
                                predictions = predictions[: mask.sum()]

        # Final shape fixes
        if predictions is not None and targets is not None:
            # For classification, ensure targets are long type
            if self.task == "classification" and targets.dtype != torch.long:
                targets = targets.long()

            # Handle dimension mismatches
            if predictions.size(0) != targets.size(0):
                min_size = min(predictions.size(0), targets.size(0))
                predictions = predictions[:min_size]
                targets = targets[:min_size]

            # Squeeze extra dimensions if needed
            if predictions.dim() > 2:
                predictions = predictions.squeeze()
            if targets.dim() > 1 and self.task == "classification":
                targets = targets.squeeze()

        return predictions, targets

    def training_step(self, batch, batch_idx):
        """Training step - works with all AstroLab model outputs."""
        outputs = self(batch)
        predictions, targets = self._extract_predictions_and_targets(batch, outputs)

        # Calculate loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.task == "classification" and hasattr(self, "train_acc"):
            self.train_acc(predictions, targets)
            self.log(
                "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
            )
            if hasattr(self, "train_f1"):
                self.train_f1(predictions, targets)
                self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        elif self.task == "regression":
            if hasattr(self, "train_mse"):
                self.train_mse(predictions, targets)
                self.log("train_mse", self.train_mse, on_step=False, on_epoch=True)
            if hasattr(self, "train_mae"):
                self.train_mae(predictions, targets)
                self.log(
                    "train_mae",
                    self.train_mae,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(batch)
        predictions, targets = self._extract_predictions_and_targets(batch, outputs)

        # Calculate loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.task == "classification" and hasattr(self, "val_acc"):
            self.val_acc(predictions, targets)
            self.log(
                "val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True
            )
            if hasattr(self, "val_f1"):
                self.val_f1(predictions, targets)
                self.log("val_f1", self.val_f1, on_epoch=True, sync_dist=True)

        elif self.task == "regression":
            if hasattr(self, "val_mse"):
                self.val_mse(predictions, targets)
                self.log("val_mse", self.val_mse, on_epoch=True, sync_dist=True)
            if hasattr(self, "val_mae"):
                self.val_mae(predictions, targets)
                self.log(
                    "val_mae",
                    self.val_mae,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self(batch)
        predictions, targets = self._extract_predictions_and_targets(batch, outputs)

        # Calculate loss
        loss = self.loss_fn(predictions, targets)

        # Log metrics
        self.log("test_loss", loss, sync_dist=True)

        if self.task == "classification" and hasattr(self, "test_acc"):
            self.test_acc(predictions, targets)
            self.log("test_acc", self.test_acc, sync_dist=True)
            if hasattr(self, "test_f1"):
                self.test_f1(predictions, targets)
                self.log("test_f1", self.test_f1, sync_dist=True)

        elif self.task == "regression":
            if hasattr(self, "test_mse"):
                self.test_mse(predictions, targets)
                self.log("test_mse", self.test_mse, sync_dist=True)
            if hasattr(self, "test_mae"):
                self.test_mae(predictions, targets)
                self.log("test_mae", self.test_mae, sync_dist=True)

        return {"test_loss": loss, "predictions": predictions, "targets": targets}

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Parameter groups für unterschiedliche Learning Rates
        param_groups = self._get_parameter_groups()

        # Create optimizer
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                param_groups, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        elif self.optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            logger.warning(f"Unknown optimizer {self.optimizer_name}, using AdamW")
            optimizer = torch.optim.AdamW(
                param_groups, lr=self.learning_rate, weight_decay=self.weight_decay
            )

        # No scheduler case
        if self.scheduler_name == "none":
            return optimizer

        # Create scheduler configuration
        scheduler_config = self._create_scheduler(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def _get_parameter_groups(self):
        """Create parameter groups for different learning rates."""
        # Default: alle Parameter mit gleicher Learning Rate
        return self.parameters()

    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler configuration."""
        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if hasattr(self, "trainer") else 100,
                eta_min=self.min_lr,
            )
            return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        elif self.scheduler_name == "step":
            scheduler = StepLR(
                optimizer,
                step_size=max(10, self.trainer.max_epochs // 3)
                if hasattr(self, "trainer")
                else 30,
                gamma=0.1,
            )
            return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        elif self.scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        elif self.scheduler_name == "onecycle":
            # OneCycleLR needs total steps
            total_steps = (
                self.trainer.estimated_stepping_batches
                if hasattr(self, "trainer")
                else 1000
            )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate * 10,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy="cos",
                cycle_momentum=True if self.optimizer_name != "adam" else False,
                base_momentum=0.85 if self.optimizer_name != "adam" else 0,
                max_momentum=0.95 if self.optimizer_name != "adam" else 0,
            )
            return {"scheduler": scheduler, "interval": "step", "frequency": 1}

        else:
            logger.warning(f"Unknown scheduler {self.scheduler_name}, using cosine")
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if hasattr(self, "trainer") else 100,
                eta_min=self.min_lr,
            )
            return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    def on_train_start(self):
        """Called when training starts."""
        # Log model summary
        if hasattr(self, "model"):
            # For wrapped models, log the inner model
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        else:
            # For direct inheritance
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

        self.log_dict(
            {
                "model/total_params": float(total_params),
                "model/trainable_params": float(trainable_params),
                "model/size_mb": float(total_params * 4 / 1024 / 1024),
            }
        )

        logger.info(
            f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)"
        )

    def on_train_epoch_end(self):
        """Log learning rate at end of epoch."""
        # Log current learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("learning_rate", current_lr, prog_bar=True)
