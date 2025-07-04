"""Base model class for AstroLab using PyTorch Lightning."""

from typing import Any, Dict, Optional, Union

import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torchmetrics import Accuracy, F1Score, MeanSquaredError, MetricCollection

from .mixins import ExplainabilityMixin


class AstroBaseModel(L.LightningModule, ExplainabilityMixin):
    """Base class for all AstroLab models with Lightning integration."""

    def __init__(
        self,
        task: str = "node_classification",
        num_features: int = 128,
        num_classes: int = 10,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        class_weights: Optional[torch.Tensor] = None,
        handle_imbalance: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Task configuration
        self.task = task
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Class imbalance handling
        self.handle_imbalance = handle_imbalance
        self.class_weights = class_weights

        # Setup metrics using torchmetrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup task-specific metrics using torchmetrics."""
        if "classification" in self.task:
            if self.num_classes == 1:
                # Binary classification
                metrics = {
                    "acc": Accuracy(task="binary"),
                    "f1": F1Score(task="binary"),
                }
            else:
                # Multiclass classification
                metrics = {
                    "acc": Accuracy(task="multiclass", num_classes=self.num_classes),
                    "f1": F1Score(
                        task="multiclass", num_classes=self.num_classes, average="macro"
                    ),
                }
            self.train_metrics = MetricCollection(metrics, prefix="train_")
            self.val_metrics = MetricCollection(metrics.copy(), prefix="val_")
            self.test_metrics = MetricCollection(metrics.copy(), prefix="test_")
        elif "regression" in self.task:
            metrics = {
                "mse": MeanSquaredError(),
                "rmse": MeanSquaredError(squared=False),
            }
            self.train_metrics = MetricCollection(metrics, prefix="train_")
            self.val_metrics = MetricCollection(metrics.copy(), prefix="val_")
            self.test_metrics = MetricCollection(metrics.copy(), prefix="test_")
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def forward(self, batch: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss with class imbalance handling."""
        if "classification" in self.task:
            # Ensure target is long type for classification
            target = target.long()
            
            if self.handle_imbalance and self.class_weights is not None:
                # Use class weights for imbalanced data
                return F.cross_entropy(pred, target, weight=self.class_weights)
            else:
                # Standard cross entropy
                return F.cross_entropy(pred, target)
        elif "regression" in self.task:
            return F.mse_loss(pred, target.float())
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def training_step(self, batch: Union[Data, Batch], batch_idx: int) -> torch.Tensor:
        """Training step."""
        pred = self(batch)

        # Get target based on task
        if self.task in ["node_classification", "node_regression"]:
            target = batch.y
            # Use train_mask if available
            if hasattr(batch, "train_mask") and batch.train_mask is not None:
                pred = pred[batch.train_mask]
                target = target[batch.train_mask]
        elif self.task in ["graph_classification", "graph_regression"]:
            # For graph classification, use graph_y if available, otherwise y
            if hasattr(batch, "graph_y") and batch.graph_y is not None:
                target = batch.graph_y
            else:
                target = batch.y
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Skip if no valid samples
        if target.numel() == 0:
            return torch.tensor(0.0, requires_grad=True)

        loss = self._compute_loss(pred, target)

        # Update metrics - ensure target is long for classification metrics
        if "classification" in self.task:
            self.train_metrics.update(pred, target.long())
        else:
            self.train_metrics.update(pred, target)

        # Log loss with batch_size to avoid inference warning
        batch_size = getattr(batch, "num_graphs", getattr(batch, "num_nodes", 1))
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(
        self, batch: Union[Data, Batch], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        pred = self(batch)

        # Get target based on task
        if self.task in ["node_classification", "node_regression"]:
            target = batch.y
            # Use val_mask if available
            if hasattr(batch, "val_mask") and batch.val_mask is not None:
                pred = pred[batch.val_mask]
                target = target[batch.val_mask]
        elif self.task in ["graph_classification", "graph_regression"]:
            # For graph classification, use graph_y if available, otherwise y
            if hasattr(batch, "graph_y") and batch.graph_y is not None:
                target = batch.graph_y
            else:
                target = batch.y
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Skip if no valid samples
        if target.numel() == 0:
            return torch.tensor(0.0)

        loss = self._compute_loss(pred, target)

        # Update metrics - ensure target is long for classification metrics
        if "classification" in self.task:
            self.val_metrics.update(pred, target.long())
        else:
            self.val_metrics.update(pred, target)

        # Log loss with batch_size to avoid inference warning
        batch_size = getattr(batch, "num_graphs", getattr(batch, "num_nodes", 1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def test_step(self, batch: Union[Data, Batch], batch_idx: int) -> torch.Tensor:
        """Test step."""
        pred = self(batch)

        # Get target based on task
        if self.task in ["node_classification", "node_regression"]:
            target = batch.y
            # Use test_mask if available
            if hasattr(batch, "test_mask") and batch.test_mask is not None:
                pred = pred[batch.test_mask]
                target = target[batch.test_mask]
        elif self.task in ["graph_classification", "graph_regression"]:
            # For graph classification, use graph_y if available, otherwise y
            if hasattr(batch, "graph_y") and batch.graph_y is not None:
                target = batch.graph_y
            else:
                target = batch.y
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Skip if no valid samples
        if target.numel() == 0:
            return torch.tensor(0.0)

        loss = self._compute_loss(pred, target)

        # Update metrics - ensure target is long for classification metrics
        if "classification" in self.task:
            self.test_metrics.update(pred, target.long())
        else:
            self.test_metrics.update(pred, target)

        # Log loss with batch_size to avoid inference warning
        batch_size = getattr(batch, "num_graphs", getattr(batch, "num_nodes", 1))
        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        """Log train metrics at epoch end."""
        self.log_dict(self.train_metrics.compute(), on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        self.log_dict(self.val_metrics.compute(), on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Scheduler
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=1e-6
            )
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
        elif self.scheduler == "warmup_cosine":
            # Warmup + Cosine annealing
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=self.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
            )

            # Combined scheduler
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = None

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss" if self.scheduler == "plateau" else None,
                },
            }
        else:
            return {"optimizer": optimizer}

    def compute_class_weights(self, train_dataset) -> torch.Tensor:
        """Compute class weights from training data to handle imbalance."""
        if not self.handle_imbalance:
            return None

        # Get all labels from training data
        all_labels = []
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            if self.task in ["graph_classification", "graph_regression"]:
                # For graph classification, use graph_y if available
                if hasattr(data, "graph_y") and data.graph_y is not None:
                    all_labels.append(data.graph_y.item() if data.graph_y.dim() == 0 else data.graph_y.tolist())
                elif hasattr(data, "y") and data.y is not None:
                    all_labels.append(data.y.item() if data.y.dim() == 0 else data.y.tolist())
            else:
                # For node classification
                if hasattr(data, "y") and data.y is not None:
                    if hasattr(data, "train_mask") and data.train_mask is not None:
                        # Only use training nodes
                        all_labels.extend(data.y[data.train_mask].tolist())
                    else:
                        all_labels.extend(data.y.tolist())

        if not all_labels:
            return None

        # Flatten nested lists if necessary
        if isinstance(all_labels[0], list):
            all_labels = [item for sublist in all_labels for item in sublist]

        # Convert to tensor
        labels = torch.tensor(all_labels, dtype=torch.long)

        # Compute class counts
        class_counts = torch.bincount(labels, minlength=self.num_classes)

        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1)

        # Compute inverse weights (more samples = lower weight)
        class_weights = 1.0 / class_counts.float()

        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_weights)

        return class_weights
