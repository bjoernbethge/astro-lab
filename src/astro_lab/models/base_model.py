"""
Minimal AstroBaseModel for AstroLab
==================================

Clean, modern base class for all AstroLab models.
No Mixins, no legacy code, no special-case logic.
"""

from typing import Any, Dict, Union

import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torchmetrics import Accuracy, F1Score, MeanSquaredError, Metric, MetricCollection

from astro_lab.models.mixins.analysis import ModelAnalysisMixin
from astro_lab.models.mixins.explainability import ExplainabilityMixin
from astro_lab.models.mixins.tensordict import TensorDictMixin


class AstroBaseModel(
    L.LightningModule, ModelAnalysisMixin, ExplainabilityMixin, TensorDictMixin
):
    """
    Minimal base class for all AstroLab models.
    Only core Lightning and metrics logic.
    Now includes model analysis, explainability, and TensorDict methods via Mixins.
    """

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
        max_epochs: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.num_features = num_features
        self.num_classes = max(2, num_classes)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self._setup_metrics()

    def _setup_metrics(self):
        metrics: dict[str, Metric]
        if "classification" in self.task:
            if self.num_classes == 2:
                metrics = {"acc": Accuracy(task="binary"), "f1": F1Score(task="binary")}
            else:
                metrics = {
                    "acc": Accuracy(task="multiclass", num_classes=self.num_classes),
                    "f1": F1Score(
                        task="multiclass", num_classes=self.num_classes, average="macro"
                    ),
                }
        elif "regression" in self.task:
            metrics = {
                "mse": MeanSquaredError(),
                "rmse": MeanSquaredError(squared=False),
            }
        else:
            raise ValueError(f"Unknown task: {self.task}")
        self.train_metrics = MetricCollection(metrics, prefix="train_")
        self.val_metrics = MetricCollection(metrics.copy(), prefix="val_")
        self.test_metrics = MetricCollection(metrics.copy(), prefix="test_")

    def _extract_target(self, batch):
        # Robust extraction of target from batch
        if hasattr(batch, "y"):
            target = batch.y
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            return target
        raise AttributeError(
            "Batch does not have a 'y' attribute for target extraction."
        )

    def forward(self, batch: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if "classification" in self.task:
            return F.cross_entropy(pred, target.long())
        elif "regression" in self.task:
            return F.mse_loss(pred, target.float())
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def training_step(self, batch: Union[Data, Batch], batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        target = self._extract_target(batch)
        loss = self._compute_loss(pred, target)
        if "classification" in self.task:
            self.train_metrics.update(pred, target.long())
        else:
            self.train_metrics.update(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Union[Data, Batch], batch_idx: int
    ) -> torch.Tensor:
        pred = self(batch)
        target = self._extract_target(batch)
        loss = self._compute_loss(pred, target)
        if "classification" in self.task:
            self.val_metrics.update(pred, target.long())
        else:
            self.val_metrics.update(pred, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Union[Data, Batch], batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        target = self._extract_target(batch)
        loss = self._compute_loss(pred, target)
        if "classification" in self.task:
            self.test_metrics.update(pred, target.long())
        else:
            self.test_metrics.update(pred, target)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
