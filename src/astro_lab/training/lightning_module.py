"""
PyTorch Lightning Module for AstroLab Models

Optimized Lightning wrapper with modern training techniques and real model integration.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

# Import real model classes
from astro_lab.models.astro import AstroSurveyGNN
from astro_lab.models.astrophot_models import AstroPhotGNN
from astro_lab.models.tgnn import ALCDEFTemporalGNN
from astro_lab.models.utils import create_gaia_classifier, create_sdss_galaxy_classifier

# Tensor integration
try:
    from astro_lab.tensors import SurveyTensor

    TENSOR_INTEGRATION_AVAILABLE = True
except ImportError:
    TENSOR_INTEGRATION_AVAILABLE = False
    SurveyTensor = None


class AstroLightningModule(LightningModule):
    """Optimized Lightning wrapper for real AstroLab models."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_config: Optional[Dict[str, Any]] = None,
        task_type: str = "classification",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: Optional[str] = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        gradient_clip_val: Optional[float] = 1.0,
        projection_dim: int = 128,
        temperature: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Store or create model
        if model is not None:
            self.model = model
        elif model_config is not None:
            self.model = self._create_model_from_config(model_config)
        else:
            # Default model will be created in first forward pass
            self.model = None

        self.task_type = task_type
        self.projection_dim = projection_dim
        self.temperature = temperature

        # Create projection head for contrastive learning
        if task_type == "unsupervised":
            self.projection_head = None  # Will be created dynamically

        # Performance tracking
        self._step_times = []
        self._memory_usage = []

    def _create_model_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Create model from configuration using real model factories."""
        model_type = config.get("type", "gaia_classifier")
        model_params = config.get("params", {})

        if model_type == "gaia_classifier":
            return create_gaia_classifier(**model_params)
        elif model_type == "sdss_classifier":
            return create_sdss_galaxy_classifier(**model_params)
        elif model_type == "astro_survey_gnn":
            return AstroSurveyGNN(**model_params)
        elif model_type == "astrophot_gnn":
            return AstroPhotGNN(**model_params)
        elif model_type == "alcdef_temporal":
            return ALCDEFTemporalGNN(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _auto_create_model(self, input_features: int) -> nn.Module:
        """Auto-create model based on input feature dimensions."""
        print(f"🔧 Auto-creating model for {input_features} input features")

        # Create default Gaia classifier
        model = create_gaia_classifier(
            hidden_dim=self.hparams.get("hidden_dim", 128),
            num_classes=self.hparams.get("num_classes", 8),
        )
        return model

    def _auto_create_projection_head(self, hidden_dim: int) -> nn.Module:
        """Auto-create projection head for contrastive learning."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.projection_dim),
        )

    def forward(self, batch):
        """Forward pass with automatic model creation."""
        # Extract data from batch
        if hasattr(batch, "x"):
            x, edge_index = batch.x, batch.edge_index
            batch_tensor = getattr(batch, "batch", None)
        elif isinstance(batch, dict):
            x, edge_index = batch["x"], batch["edge_index"]
            batch_tensor = batch.get("batch", None)
        else:
            x, edge_index = batch[0].x, batch[0].edge_index
            batch_tensor = getattr(batch[0], "batch", None)

        # Auto-create model if needed
        if self.model is None:
            input_features = x.shape[1]
            self.model = self._auto_create_model(input_features)

        # Auto-create projection head if needed
        if self.task_type == "unsupervised" and self.projection_head is None:
            if hasattr(self.model, "hidden_dim") and isinstance(
                self.model.hidden_dim, int
            ):
                hidden_dim = self.model.hidden_dim
            else:
                hidden_dim = 128  # Default
            self.projection_head = self._auto_create_projection_head(hidden_dim)

        # Forward pass through model
        if isinstance(self.model, (AstroSurveyGNN, AstroPhotGNN, ALCDEFTemporalGNN)):
            # Use native model interface
            output = self.model(x, edge_index, batch_tensor, return_embeddings=True)
            if isinstance(output, dict):
                embeddings = output.get("embeddings", output.get("output", x))
            else:
                embeddings = output
        else:
            # Generic model interface
            embeddings = self.model(x, edge_index)

        return embeddings

    def _get_embeddings(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get embeddings from the model in a unified way."""
        if self.model is None:
            raise ValueError("Model not initialized")

        if isinstance(self.model, AstroSurveyGNN):
            # AstroSurveyGNN with tensor support
            output = self.model(
                x=batch["x"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_embeddings=True,
            )
        elif isinstance(self.model, ALCDEFTemporalGNN):
            # ALCDEFTemporalGNN with lightcurve support
            output = self.model(
                lightcurve=batch["lightcurve"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_embeddings=True,
            )
        elif isinstance(self.model, AstroPhotGNN):
            # AstroPhotGNN with component support
            output = self.model(
                x=batch["x"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_components=False,
            )
        else:
            # Generic fallback
            output = self.model(batch["x"], batch["edge_index"])

        # Extract embeddings from output
        if isinstance(output, dict):
            return output.get(
                "embeddings", output.get("output", list(output.values())[0])
            )
        return output

    def _contrastive_loss(
        self, embeddings: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Compute contrastive loss for unsupervised learning."""
        if self.projection_head is None:
            raise ValueError("Projection head not initialized")

        # Project embeddings
        projections = self.projection_head(embeddings)
        projections = F.normalize(projections, dim=1)

        # Create positive pairs (augmented versions of same sample)
        noise = torch.randn_like(projections) * 0.1
        aug_projections = F.normalize(projections + noise, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(projections, aug_projections.t()) / self.temperature

        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=projections.device)

        # Contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def _compute_step(
        self, batch: Dict[str, Any], stage: str
    ) -> Dict[str, torch.Tensor]:
        """Unified computation for all steps."""
        results = {}

        if self.task_type == "unsupervised":
            # Unsupervised learning with contrastive loss
            embeddings = self._get_embeddings(batch)
            batch_size = embeddings.size(0)
            loss = self._contrastive_loss(embeddings, batch_size)
            results["loss"] = loss

        else:
            # Supervised learning
            predictions = self(batch)
            targets = batch["y"]

            # Compute loss based on task type
            if self.task_type == "classification":
                loss = F.cross_entropy(predictions, targets)
                acc = (predictions.argmax(dim=1) == targets).float().mean()
                results["loss"] = loss
                results["acc"] = acc
            elif self.task_type == "regression":
                loss = F.mse_loss(predictions, targets)
                mae = F.l1_loss(predictions, targets)
                results["loss"] = loss
                results["mae"] = mae
            else:
                loss = F.mse_loss(predictions, targets)
                results["loss"] = loss

        return results

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        results = self._compute_step(batch, "train")

        # Log metrics
        for key, value in results.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=True)

        return results["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        results = self._compute_step(batch, "val")

        # Log metrics
        for key, value in results.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=True)

        return results["loss"]

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step."""
        results = self._compute_step(batch, "test")

        # Log metrics
        for key, value in results.items():
            self.log(f"test_{key}", value, on_step=False, on_epoch=True)

        return results["loss"]

    def configure_optimizers(self) -> Union[AdamW, Dict[str, Any]]:
        """Configure optimizer with modern best practices."""
        # Get hyperparameters
        learning_rate = self.hparams.get("learning_rate", 1e-3)
        weight_decay = self.hparams.get("weight_decay", 1e-4)
        scheduler = self.hparams.get("scheduler", "cosine")
        max_epochs = self.hparams.get("max_epochs", 100)
        warmup_epochs = self.hparams.get("warmup_epochs", 5)

        # Optimizer with parameter groups
        param_groups = [
            {
                "params": [p for n, p in self.named_parameters() if "bias" not in n],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if "bias" in n],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        # Scheduler configuration
        if scheduler == "cosine":
            scheduler_obj = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=learning_rate * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler_obj,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        elif scheduler == "onecycle":
            # Estimate total steps
            total_steps = self.hparams.get("total_steps", max_epochs * 100)  # Estimate
            scheduler_obj = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=warmup_epochs / max_epochs,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=10000.0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler_obj,
                    "interval": "step",
                },
            }
        elif scheduler == "plateau":
            scheduler_obj = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler_obj,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        else:
            return optimizer

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        return self(batch)


__all__ = ["AstroLightningModule"]
