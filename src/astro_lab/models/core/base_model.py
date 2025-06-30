"""
Base Model for AstroLab
======================

Enhanced base class for astronomical neural networks with mixin support.
"""

import logging
from typing import Any, Dict, Union, Optional

import lightning as L
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

# Import mixins
from ..components.mixins import (
    ExplainabilityMixin,
    HPOResetMixin,
    MetricsMixin,
    OptimizationMixin,
    VisualizationMixin,
    MLflowMixin,
)

logger = logging.getLogger(__name__)


class AstroBaseModel(
    L.LightningModule,
    MetricsMixin,
    OptimizationMixin,
    VisualizationMixin,
    HPOResetMixin,
    ExplainabilityMixin,
    MLflowMixin,
):
    """
    Enhanced base model for astronomical neural networks.

    Features:
    - Pure PyTorch metrics computation
    - Advanced optimization strategies
    - Visualization capabilities
    - HPO-ready with efficient parameter reset
    - MLflow integration
    - torch.compile support for production
    - Dynamic shape support for variable batch sizes
    """

    def __init__(
        self,
        task: str = "node_classification",
        num_features: int = 3,
        num_classes: int = 2,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        mlflow_logging: bool = True,
        # Compilation settings
        compile_model: bool = False,
        compile_mode: str = "default",  # "default", "reduce-overhead", "max-autotune"
        compile_dynamic: bool = True,  # Dynamic shapes for variable batch sizes
        **kwargs,
    ):
        super().__init__()
        
        # Initialize MLflow mixin
        MLflowMixin.__init__(self)

        # Core configuration
        self.task = task
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mlflow_logging = mlflow_logging
        
        # Compilation settings
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.compile_dynamic = compile_dynamic

        # Save hyperparameters for Lightning
        self.save_hyperparameters()

        # Configure loss function
        self.loss_fn = self._configure_loss_function()

        # Initialize metrics storage
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        # Compiled model placeholder
        self._compiled_forward = None

    def _configure_loss_function(self) -> nn.Module:
        """Configure appropriate loss function for the task."""
        # Handle various task name formats
        task_lower = self.task.lower()
        
        if any(name in task_lower for name in ["node_classification", "graph_classification", "classification"]):
            if self.num_classes == 2:
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss(label_smoothing=0.1)
        elif any(name in task_lower for name in ["node_regression", "graph_regression", "regression"]):
            return nn.SmoothL1Loss()
        elif "link_prediction" in task_lower:
            return nn.BCEWithLogitsLoss()
        else:
            logger.warning(f"Unknown task type: {self.task}, using CrossEntropyLoss")
            return nn.CrossEntropyLoss()

    def forward(self, batch: Union[Data, HeteroData, Batch]) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def on_fit_start(self) -> None:
        """Called at the beginning of fit."""
        # Compile model if requested
        if self.compile_model and self._compiled_forward is None:
            logger.info(f"Compiling model with mode={self.compile_mode}, dynamic={self.compile_dynamic}")
            try:
                self._compiled_forward = torch.compile(
                    self.forward,
                    mode=self.compile_mode,
                    dynamic=self.compile_dynamic,
                )
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
                self.compile_model = False
        
        # MLflow logging
        if self.mlflow_logging and mlflow.active_run():
            mlflow.log_params(
                {
                    "model_type": self.__class__.__name__,
                    "task": self.task,
                    "num_features": self.num_features,
                    "num_classes": self.num_classes,
                    "hidden_dim": self.hidden_dim,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "compiled": self.compile_model,
                    "compile_mode": self.compile_mode if self.compile_model else "none",
                }
            )

    def _forward_wrapper(self, batch: Union[Data, HeteroData, Batch]) -> Union[Tensor, Dict[str, Tensor]]:
        """Wrapper for forward pass that handles compilation."""
        if self.compile_model and self._compiled_forward is not None:
            return self._compiled_forward(batch)
        else:
            return self.forward(batch)

    def training_step(
        self, batch: Union[Data, HeteroData, Batch], batch_idx: int
    ) -> STEP_OUTPUT:
        """Training step with metrics tracking."""
        # Forward pass with potential compilation
        logits = self._forward_wrapper(batch)

        # Calculate loss
        loss = self._calculate_loss(logits, batch)

        # Calculate metrics using mixin - avoid device sync
        with torch.no_grad():
            metrics = self.calculate_metrics(logits, batch, stage="train")

        # Log metrics - accumulate on GPU
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)

        for metric_name, metric_value in metrics.items():
            self.log(f"train_{metric_name}", metric_value, on_step=False, on_epoch=True, sync_dist=False)

        # Store for epoch-end logging
        self.train_metrics.update(metrics)

        return loss

    def validation_step(
        self, batch: Union[Data, HeteroData, Batch], batch_idx: int
    ) -> STEP_OUTPUT:
        """Validation step with metrics tracking."""
        # Forward pass with potential compilation
        logits = self._forward_wrapper(batch)

        # Calculate loss
        loss = self._calculate_loss(logits, batch)

        # Calculate metrics using mixin
        with torch.no_grad():
            metrics = self.calculate_metrics(logits, batch, stage="val")

        # Log metrics
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_value in metrics.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Store for epoch-end logging
        self.val_metrics.update(metrics)

        return loss

    def test_step(
        self, batch: Union[Data, HeteroData, Batch], batch_idx: int
    ) -> STEP_OUTPUT:
        """Test step with metrics tracking."""
        # Forward pass with potential compilation
        logits = self._forward_wrapper(batch)

        # Calculate loss
        loss = self._calculate_loss(logits, batch)

        # Calculate metrics using mixin
        with torch.no_grad():
            metrics = self.calculate_metrics(logits, batch, stage="test")

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for metric_name, metric_value in metrics.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Store for final logging
        self.test_metrics.update(metrics)

        return loss

    def _calculate_loss(
        self, logits: Union[Tensor, Dict[str, Tensor]], batch: Union[Data, HeteroData, Batch]
    ) -> Tensor:
        """Calculate loss based on task type."""
        # Handle dictionary outputs (e.g., from ShapeModelingHead)
        if isinstance(logits, dict):
            return self._calculate_dict_loss(logits, batch)
        
        y = getattr(batch, "y", None)
        
        # Handle different loss scenarios
        if y is None:
            # Self-supervised loss for unlabeled data
            return self._self_supervised_loss(logits, batch)
        
        if self.task in ["node_classification", "node_regression"]:
            train_mask = getattr(batch, "train_mask", None)
            if train_mask is not None and self.training:
                return self.loss_fn(logits[train_mask], y[train_mask])
            else:
                return self.loss_fn(logits, y)
        elif self.task in ["graph_classification", "graph_regression"]:
            return self.loss_fn(logits, y)
        elif self.task == "link_prediction":
            pos_edge_index = getattr(batch, "pos_edge_index", None)
            if pos_edge_index is None:
                raise ValueError(
                    "Batch must have 'pos_edge_index' for link prediction!"
                )
            pos_logits = logits[: pos_edge_index.size(1)]
            neg_logits = logits[pos_edge_index.size(1) :]
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            all_logits = torch.cat([pos_logits, neg_logits])
            all_labels = torch.cat([pos_labels, neg_labels])
            return self.loss_fn(all_logits, all_labels)
        else:
            # Fallback
            return self.loss_fn(logits, y)
    
    def _calculate_dict_loss(self, outputs: Dict[str, Tensor], batch: Union[Data, HeteroData, Batch]) -> Tensor:
        """Calculate loss for dictionary outputs (e.g., ShapeModelingHead)."""
        if "real_coeffs" in outputs and "imag_coeffs" in outputs:
            # Shape modeling loss - reconstruct from coefficients
            real_coeffs = outputs["real_coeffs"]
            imag_coeffs = outputs["imag_coeffs"]
            
            # If we have ground truth coefficients
            if hasattr(batch, "real_coeffs_gt") and hasattr(batch, "imag_coeffs_gt"):
                loss_real = F.mse_loss(real_coeffs, batch.real_coeffs_gt)
                loss_imag = F.mse_loss(imag_coeffs, batch.imag_coeffs_gt)
                return loss_real + loss_imag
            else:
                # Self-supervised: encourage orthogonality and unit norm
                coeffs_complex = torch.complex(real_coeffs, imag_coeffs)
                coeffs_norm = torch.norm(coeffs_complex, dim=-1)
                
                # Loss 1: Encourage unit norm
                norm_loss = F.mse_loss(coeffs_norm, torch.ones_like(coeffs_norm))
                
                # Loss 2: Orthogonality between coefficients
                gram_matrix = torch.matmul(real_coeffs, real_coeffs.t()) + torch.matmul(imag_coeffs, imag_coeffs.t())
                eye = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
                ortho_loss = F.mse_loss(gram_matrix, eye)
                
                return norm_loss + 0.1 * ortho_loss
        
        elif "period" in outputs and "uncertainty" in outputs:
            # Period detection loss
            period = outputs["period"]
            uncertainty = outputs["uncertainty"]
            
            if hasattr(batch, "period_gt"):
                # Negative log-likelihood with uncertainty
                diff = (period - batch.period_gt) ** 2
                nll_loss = 0.5 * torch.log(uncertainty + 1e-6) + 0.5 * diff / (uncertainty + 1e-6)
                return nll_loss.mean()
            else:
                # Self-supervised: encourage reasonable periods
                # Penalize very small or very large periods
                period_regularization = torch.relu(0.1 - period) + torch.relu(period - 100.0)
                uncertainty_regularization = torch.relu(uncertainty - 10.0)  # Avoid too high uncertainty
                return period_regularization.mean() + 0.1 * uncertainty_regularization.mean()
        
        else:
            raise ValueError(f"Unknown dictionary output format: {outputs.keys()}")
    
    def _self_supervised_loss(self, logits: Tensor, batch: Union[Data, HeteroData, Batch]) -> Tensor:
        """
        Self-supervised loss for unlabeled data.
        
        Uses contrastive learning or reconstruction objectives.
        """
        # Option 1: Contrastive loss (SimCLR-style)
        if hasattr(batch, 'edge_index') and batch.edge_index is not None:
            # Use graph structure for positive pairs
            edge_index = batch.edge_index
            
            # Normalize embeddings
            logits_norm = F.normalize(logits, p=2, dim=-1)
            
            # Compute similarity between connected nodes
            src_emb = logits_norm[edge_index[0]]
            dst_emb = logits_norm[edge_index[1]]
            
            # Positive pairs: connected nodes
            pos_sim = (src_emb * dst_emb).sum(dim=-1)
            
            # Negative pairs: random sampling
            num_neg = min(edge_index.size(1), 100)
            neg_idx = torch.randint(0, logits.size(0), (num_neg,), device=logits.device)
            neg_emb = logits_norm[neg_idx]
            
            # Compute negative similarities
            neg_sim = torch.mm(src_emb[:num_neg], neg_emb.t())
            
            # InfoNCE loss
            temperature = 0.1
            pos_loss = -torch.log(torch.exp(pos_sim / temperature).mean() + 1e-8)
            neg_loss = torch.log(torch.exp(neg_sim / temperature).sum(dim=1).mean() + 1e-8)
            
            return pos_loss + neg_loss
        
        # Option 2: Feature reconstruction loss
        elif hasattr(batch, 'x') and batch.x is not None:
            # Predict original features from embeddings
            if not hasattr(self, 'reconstruction_head'):
                self.reconstruction_head = nn.Linear(
                    logits.size(-1), 
                    batch.x.size(-1)
                ).to(logits.device)
            
            reconstructed = self.reconstruction_head(logits)
            return F.mse_loss(reconstructed, batch.x)
        
        # Option 3: Entropy maximization (encourage diverse predictions)
        else:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            return -entropy  # Negative because we want to maximize entropy

    def on_train_epoch_end(self) -> None:
        """End of training epoch - MLflow logging."""
        if self.mlflow_logging and mlflow.active_run():
            for metric_name, metric_value in self.train_metrics.items():
                mlflow.log_metric(
                    f"train_{metric_name}", metric_value, step=self.current_epoch
                )
        self.train_metrics.clear()

    def on_validation_epoch_end(self) -> None:
        """End of validation epoch - MLflow logging."""
        if self.mlflow_logging and mlflow.active_run():
            for metric_name, metric_value in self.val_metrics.items():
                mlflow.log_metric(
                    f"val_{metric_name}", metric_value, step=self.current_epoch
                )
        self.val_metrics.clear()

    def on_test_epoch_end(self) -> None:
        """End of test epoch - final MLflow logging."""
        if self.mlflow_logging and mlflow.active_run():
            for metric_name, metric_value in self.test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
        self.test_metrics.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer using mixin."""
        # Use the optimization mixin method
        return self.configure_astro_optimizers()

    def predict_step(
        self, batch: Union[Data, HeteroData, Batch], batch_idx: int
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Prediction step for inference."""
        logits = self._forward_wrapper(batch)
        
        # Handle dictionary outputs
        if isinstance(logits, dict):
            return logits  # Return as-is for structured outputs

        if self.task in ["node_classification", "graph_classification"]:
            if self.num_classes == 2:
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=-1)
        elif self.task == "link_prediction":
            return torch.sigmoid(logits)
        else:
            return logits

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_class": self.__class__.__name__,
            "task": self.task,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": next(self.parameters()).device.type,
            "compiled": self.compile_model,
        }
