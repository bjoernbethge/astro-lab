"""
PyTorch Lightning Module for AstroLab Models

Optimized Lightning wrapper with modern training techniques and reduced code duplication.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
import torch.nn as nn

from ..models import ALCDEFTemporalGNN, AstroPhotGNN, AstroSurveyGNN


class AstroLightningModule(LightningModule):
    """Optimized Lightning wrapper with modern ML techniques."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_config: Optional[Dict[str, Any]] = None,
        task_type: str = "unsupervised",
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
        self.save_hyperparameters(ignore=['model'])
        
        # Store or create model
        if model is not None:
            self.model = model
        elif model_config is not None:
            self.model = self._create_model_from_config(model_config)
        else:
            # Create default model with automatic feature detection
            self.model = None  # Will be created in first forward pass
            
        self.task_type = task_type
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Create projection head for contrastive learning
        if task_type == "unsupervised":
            self.projection_head = None  # Will be created dynamically
            
        # Metrics
        self._setup_metrics()
        
        # Performance tracking
        self._step_times = []
        self._memory_usage = []

    def _create_model_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Create model from configuration."""
        from astro_lab.models.astro import AstroSurveyGNN
        
        model_type = config.get("type", "gaia_classifier")
        model_params = config.get("params", {})
        
        if model_type == "gaia_classifier":
            # Auto-detect input features during first forward pass
            return AstroSurveyGNN(
                hidden_dim=model_params.get("hidden_dim", 128),
                output_dim=model_params.get("num_classes", 8),
                num_layers=model_params.get("num_layers", 3),
                dropout=model_params.get("dropout", 0.1),
                conv_type=model_params.get("conv_type", "gat"),
                **model_params
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _auto_create_model(self, input_features: int) -> nn.Module:
        """Auto-create model based on input feature dimensions."""
        from astro_lab.models.astro import AstroSurveyGNN
        
        print(f"🔧 Auto-creating model for {input_features} input features")
        
        # Create model with proper input projection
        model = AstroSurveyGNN(
            hidden_dim=self.hparams.get("hidden_dim", 128),
            output_dim=self.hparams.get("num_classes", 8),
            num_layers=3,
            dropout=0.1,
            conv_type="gat",
        )
        
        # Add input projection layer
        model.input_proj = nn.Linear(input_features, model.hidden_dim)
        
        return model

    def _auto_create_projection_head(self, hidden_dim: int) -> nn.Module:
        """Auto-create projection head for contrastive learning."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.projection_dim),
            nn.L2Norm(dim=1)  # Normalize for contrastive learning
        )

    def forward(self, batch):
        """Forward pass with automatic model creation."""
        # Extract data
        if hasattr(batch, 'x'):
            x, edge_index = batch.x, batch.edge_index
            batch_tensor = getattr(batch, 'batch', None)
        else:
            x, edge_index = batch[0].x, batch[0].edge_index
            batch_tensor = getattr(batch[0], 'batch', None)
        
        # Auto-create model if needed
        if self.model is None:
            input_features = x.shape[1]
            self.model = self._auto_create_model(input_features)
            
        # Auto-create projection head if needed
        if self.task_type == "unsupervised" and self.projection_head is None:
            self.projection_head = self._auto_create_projection_head(self.model.hidden_dim)
        
        # Forward pass through model
        if hasattr(self.model, 'input_proj'):
            h = self.model.input_proj(x)
            # Manual forward through GNN layers
            for i, (conv, norm) in enumerate(zip(self.model.convs, self.model.norms)):
                h_prev = h
                h = conv(h, edge_index)
                h = norm(h)
                h = F.relu(h)
                h = F.dropout(h, p=self.model.dropout, training=self.training)
                if i > 0:  # Skip connection from second layer onward
                    h = h + h_prev
            embeddings = h
        else:
            embeddings = self.model(x, edge_index, batch_tensor)
            
        return embeddings

    def _setup_task_components(self) -> None:
        """Setup task-specific components like loss functions and metrics."""
        if self.task_type == "unsupervised":
            # For unsupervised learning, we'll use contrastive learning
            self.contrastive_temperature = 0.1
            self.embedding_dim = getattr(self.model, 'hidden_dim', 128)
            
            # Projection head for contrastive learning
            self.projection_head = torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            )

    def _get_embeddings(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get embeddings from the model in a unified way."""
        if hasattr(self.model, "extract_survey_features"):
            # AstroSurveyGNN
            output = self.model(
                x=batch["x"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_embeddings=True,
            )
        elif hasattr(self.model, "lightcurve_encoder"):
            # ALCDEFTemporalGNN
            output = self.model(
                lightcurve=batch["lightcurve"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_embeddings=True,
            )
        elif hasattr(self.model, "component_heads"):
            # AstroPhotGNN
            output = self.model(
                x=batch["x"],
                edge_index=batch["edge_index"],
                batch=batch.get("batch"),
                return_embeddings=True,
            )
        else:
            # Generic fallback
            output = self.model(batch["x"], batch["edge_index"])

        # Extract embeddings from output with type guards
        if isinstance(output, dict):
            if "embeddings" in output:
                return output["embeddings"]
            elif "node_embeddings" in output:
                return output["node_embeddings"]
            else:
                # If dict has no embeddings, take first tensor value
                for value in output.values():
                    if isinstance(value, torch.Tensor):
                        return value
                # Fallback: convert to tensor
                return torch.tensor(list(output.values())[0])
        elif isinstance(output, torch.Tensor):
            return output
        else:
            # Convert any other type to tensor
            return torch.tensor(output)

    def _contrastive_loss(self, embeddings: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Compute contrastive loss for unsupervised learning."""
        # Project embeddings
        projections = self.projection_head(embeddings)
        projections = F.normalize(projections, dim=1)
        
        # Create positive pairs (augmented versions of same sample)
        # For now, use simple noise augmentation
        noise = torch.randn_like(projections) * 0.1
        aug_projections = F.normalize(projections + noise, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(projections, aug_projections.t()) / self.contrastive_temperature
        
        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=projections.device)
        
        # Contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def _compute_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, torch.Tensor]:
        """Unified computation for all steps to reduce code duplication."""
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

            # Compute loss with type guards
            if isinstance(predictions, dict) and isinstance(targets, dict):
                loss = self._multi_task_loss(predictions, targets)
                results["loss"] = loss
            else:
                # Ensure predictions is a tensor for supervised learning
                if isinstance(predictions, dict):
                    # Extract main prediction tensor from dict
                    pred_tensor = next(iter(predictions.values()))
                else:
                    pred_tensor = predictions
                
                if self.task_type == "classification":
                    loss = F.cross_entropy(pred_tensor, targets)
                    acc = (pred_tensor.argmax(dim=1) == targets).float().mean()
                    results["loss"] = loss
                    results["acc"] = acc
                elif self.task_type == "regression":
                    loss = F.mse_loss(pred_tensor, targets)
                    mae = F.l1_loss(pred_tensor, targets)
                    results["loss"] = loss
                    results["mae"] = mae
                else:
                    loss = F.mse_loss(pred_tensor, targets)
                    results["loss"] = loss

        return results

    def _multi_task_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Multi-task loss with learned weighting."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for key in predictions:
            if key in targets:
                if "classification" in key:
                    loss = F.cross_entropy(predictions[key], targets[key])
                else:
                    loss = F.mse_loss(predictions[key], targets[key])
                total_loss = total_loss + loss
                
        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Optimized training step."""
        results = self._compute_step(batch, "train")
        
        # Log metrics
        for key, value in results.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=True)
            
        return results["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Optimized validation step."""
        results = self._compute_step(batch, "val")
        
        # Log metrics
        for key, value in results.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=True)
            
        return results["loss"]

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Optimized test step."""
        results = self._compute_step(batch, "test")
        
        # Log metrics
        for key, value in results.items():
            self.log(f"test_{key}", value, on_step=False, on_epoch=True)
            
        return results["loss"]

    def configure_optimizers(self) -> Union[AdamW, Dict[str, Any]]:
        """Configure optimizer with modern best practices."""
        # Optimizer with parameter groups
        param_groups = [
            {
                "params": [p for n, p in self.named_parameters() if "bias" not in n],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if "bias" in n],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            param_groups, 
            lr=self.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        # Scheduler configuration
        if self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        elif self.scheduler == "onecycle" and self.total_steps:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.total_steps,
                pct_start=self.warmup_epochs / self.max_epochs,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=10000.0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif self.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
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
