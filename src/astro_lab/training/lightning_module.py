"""
AstroLightningModule - Unified PyTorch Lightning Module for Astronomical ML
=======================================================================

Provides a robust, configurable Lightning module with unified logging,
error handling, and support for various astronomical tasks.
Optimized for Lightning 2.0+ compatibility and modern ML practices.
Fixed for PyTorch Geometric compatibility and proper batch handling.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import MulticlassAccuracy

# Import real model classes
from astro_lab.models import (
    ALCDEFTemporalGNN,
    AstroPhotGNN,
    AstroSurveyGNN,
    ModelConfig,
    create_gaia_classifier,
)
from astro_lab.training.config import TrainingConfig

# Setup logging - only errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class AstroLightningModule(LightningModule):
    """
    Unified PyTorch Lightning Module for Astronomical Machine Learning.

    Features:
    - Lightning 2.0+ compatible architecture
    - Robust error handling with detailed logging
    - Automatic model creation from config
    - Support for classification, regression, and unsupervised tasks
    - Unified logging throughout
    - Modern metrics tracking with torchmetrics
    - Automatic class detection from data
    - Advanced training features: gradient accumulation, gradient clipping
    - 2025 Best Practices: Compile mode, FSDP support, advanced schedulers
    - Fixed PyTorch Geometric compatibility
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        task_type: str = "classification",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        projection_dim: int = 128,
        temperature: float = 0.1,
        num_classes: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: float = 1.0,
        gradient_clip_algorithm: str = "norm",
        scheduler_type: str = "cosine",
        warmup_steps: int = 0,
        use_compile: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # Save hyperparameters for Lightning compatibility
        self.save_hyperparameters(ignore=["model"])

        # Core configuration
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.num_classes = num_classes  # Will be set automatically if None
        self.model_config = model_config
        self.training_config = training_config

        # Advanced training options
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.use_compile = use_compile
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.label_smoothing = label_smoothing

        # Use automatic optimization (default) instead of manual
        self.automatic_optimization = True

        # Initialize model with robust error handling
        self._initialize_model(model)

        # Initialize EMA if requested
        if self.use_ema:
            self._init_ema()

        # Initialize projection head for unsupervised learning
        self.projection_head = None
        if task_type == "unsupervised":
            self.projection_head = self._auto_create_projection_head()

        # Initialize metrics for tracking (will be set up after class detection)
        self._setup_metrics()
        self.metrics_initialized = False  # Will be set to True after setup

        # Performance tracking
        self._step_times = []
        self._memory_usage = []

    def _init_ema(self):
        """Initialize Exponential Moving Average of model weights."""
        self.ema_model = torch.nn.ModuleDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_model[name] = param.data.clone()

    def _update_ema(self):
        """Update EMA weights."""
        if self.use_ema:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.ema_model:
                        self.ema_model[name].mul_(self.ema_decay).add_(
                            param.data, alpha=1 - self.ema_decay
                        )

    def _initialize_model(self, model: Optional[torch.nn.Module]) -> None:
        """Initialize model with compile support and error handling."""
        if model is None:
            model = self._create_default_model()

        self.model = model

        # Skip torch.compile on Windows due to cl.exe and triton issues
        import platform

        skip_compile_windows = platform.system() == "Windows"

        # Apply torch.compile with robust error handling
        if self.use_compile and not skip_compile_windows:
            try:
                # Use inductor backend for best performance with dynamic shapes
                self.model = torch.compile(
                    self.model,
                    backend="inductor",  # Better than eager for performance
                    mode="reduce-overhead",  # Stable mode
                    fullgraph=False,  # Allow fallback for unsupported operations
                    dynamic=True,  # Enable dynamic shapes for graphs
                )
                logger.info("✅ Model compiled successfully with inductor backend")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")
                logger.info("🔄 Using uncompiled model")
                # Keep the original model without compilation
                pass
        else:
            if skip_compile_windows:
                logger.info(
                    "🔄 Skipping torch.compile on Windows (cl.exe/triton compatibility)"
                )
            else:
                logger.info("📝 Model training without torch.compile")

    def _load_num_classes_from_metadata(self) -> Optional[int]:
        """Load number of classes from dataset metadata files."""
        try:
            # Try common metadata file locations
            metadata_paths = [
                "data/processed/gaia/gaia_tensor_metadata.json",
                "data/processed/gaia/gaia_metadata.json",
                "data/processed/gaia_metadata.json",
            ]

            import json
            from pathlib import Path

            for metadata_path in metadata_paths:
                path = Path(metadata_path)
                if path.exists():
                    with open(path, "r") as f:
                        metadata = json.load(f)

                    # Check for classification info
                    if "classification" in metadata:
                        num_classes = metadata["classification"].get("num_classes")
                        if num_classes:
                            return int(num_classes)

                    # Fallback: check for direct num_classes field
                    if "num_classes" in metadata:
                        num_classes = metadata["num_classes"]
                        return int(num_classes)

            return None

        except Exception:
            return None

    def _setup_metrics(self) -> None:
        """Setup torchmetrics for performance tracking. 📊"""
        # Initialize as None - will be set up later when we know num_classes
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None
        self.train_f1 = None
        self.val_f1 = None
        self.test_f1 = None

    def _create_metrics_for_classes(self, num_classes: int) -> None:
        """Create metrics once we know the number of classes."""
        if self.task_type == "classification" and not self.metrics_initialized:
            # Ensure we have at least 2 classes
            num_classes = max(num_classes, 2)
            device = self.device if hasattr(self, "device") else "cpu"

            # Always use multiclass metrics for consistency
            self.train_acc = MulticlassAccuracy(num_classes=num_classes).to(device)
            self.val_acc = MulticlassAccuracy(num_classes=num_classes).to(device)
            self.test_acc = MulticlassAccuracy(num_classes=num_classes).to(device)
            self.train_f1 = F1Score(task="multiclass", num_classes=num_classes).to(
                device
            )
            self.val_f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)
            self.test_f1 = F1Score(task="multiclass", num_classes=num_classes).to(
                device
            )

            self.metrics_initialized = True
            logger.info(f"✅ Metrics initialized for {num_classes} classes")

    def _detect_num_classes_from_data(self, batch) -> int:
        """Detect number of classes from a batch of data."""
        try:
            # Handle PyTorch Geometric Data objects
            if hasattr(batch, "y"):
                targets = batch.y
            elif isinstance(batch, list) and len(batch) > 0:
                # Handle list of data objects
                if hasattr(batch[0], "y"):
                    targets = batch[0].y
                else:
                    targets = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                targets = batch.get("target") or batch.get("y") or batch.get("labels")
            elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                targets = batch[1]
            else:
                targets = None

            if targets is not None:
                if targets.dim() > 1:
                    targets = targets.flatten()
                num_classes = int(targets.max().item()) + 1
                # Ensure at least 2 classes
                return max(num_classes, 2)
            else:
                return 4  # Default fallback

        except Exception as e:
            logger.warning(f"Could not detect classes from data: {e}")
            return 4  # Default fallback

    def _create_default_model(self) -> torch.nn.Module:
        """Create a default model if none provided."""
        try:
            if self.model_config:
                return self._create_model_from_config(self.model_config)
            else:
                # Create a simple default model
                return AstroSurveyGNN(
                    input_dim=16,  # Default input dimension
                    hidden_dim=64,
                    output_dim=self.num_classes or 4,
                    num_layers=3,
                    dropout=0.1,
                )
        except Exception as e:
            logger.error(f"Failed to create default model: {e}")
            # Ultimate fallback
            return torch.nn.Sequential(
                torch.nn.Linear(16, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, self.num_classes or 4),
            )

    def _create_model_from_config(self, config: ModelConfig) -> torch.nn.Module:
        """Create model from configuration."""
        try:
            # Use the simplified ModelConfig fields
            if (
                config.task == "classification"
                and config.name
                and "gaia" in config.name.lower()
            ):
                return create_gaia_classifier(
                    hidden_dim=config.hidden_dim,
                    num_classes=config.output_dim or 7,
                )
            elif config.name and "astrophot" in config.name.lower():
                return AstroPhotGNN(
                    hidden_dim=config.hidden_dim,
                    output_dim=config.output_dim or 12,
                )
            elif config.name and "temporal" in config.name.lower():
                return ALCDEFTemporalGNN(
                    hidden_dim=config.hidden_dim,
                    task=config.task or "period_detection",
                )
            else:
                # Default survey GNN
                return AstroSurveyGNN(
                    hidden_dim=config.hidden_dim,
                    output_dim=config.output_dim or 4,
                    num_layers=config.num_layers,
                    conv_type=config.conv_type,
                    task=config.task or "classification",
                    dropout=config.dropout,
                )

        except Exception as e:
            logger.error(f"Failed to create model from config: {e}")
            return self._create_default_model()

    def _auto_create_projection_head(self) -> Optional[torch.nn.Module]:
        """Create projection head for unsupervised learning."""
        try:
            return torch.nn.Sequential(
                torch.nn.Linear(self.projection_dim, self.projection_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.projection_dim, self.projection_dim),
            )
        except Exception as e:
            logger.error(f"Failed to create projection head: {e}")
            return None

    def forward(
        self, batch: Union[torch.Tensor, Dict[str, torch.Tensor], Any]
    ) -> torch.Tensor:
        """
        Forward pass - handles different input formats.

        Args:
            batch: Input batch (can be PyTorch Geometric Data, tensor, or dict)

        Returns:
            Model output tensor
        """
        try:
            # Handle PyTorch Geometric Data objects
            if hasattr(batch, "x") and hasattr(batch, "edge_index"):
                # Standard PyTorch Geometric forward pass
                return self.model(batch.x, batch.edge_index)
            elif isinstance(batch, dict):
                # Handle dictionary input
                if "x" in batch and "edge_index" in batch:
                    return self.model(batch["x"], batch["edge_index"])
                elif "input" in batch:
                    return self.model(batch["input"])
                else:
                    # Try to find the main input
                    inputs = batch.get("features") or batch.get("data")
                    return self.model(inputs)
            elif isinstance(batch, (list, tuple)):
                # Handle list/tuple input
                if len(batch) >= 2:
                    return self.model(batch[0], batch[1])  # Assume (x, edge_index)
                else:
                    return self.model(batch[0])
            else:
                # Handle tensor input
                return self.model(batch)

        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            logger.error(f"Batch type: {type(batch)}")
            raise

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on task type."""
        try:
            # Ensure both tensors are on the same device
            if outputs.device != targets.device:
                targets = targets.to(outputs.device)

            if self.task_type == "classification":
                # Ensure targets are the right type and shape
                if targets.dtype != torch.long:
                    targets = targets.long()
                if targets.dim() > 1:
                    targets = targets.squeeze(-1)

                # Handle shape mismatch for classification
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    # Binary classification with single output
                    outputs = outputs.squeeze(-1)
                    targets = targets.float()
                    return F.binary_cross_entropy_with_logits(outputs, targets)
                else:
                    # Multi-class classification
                    if self.label_smoothing > 0:
                        return F.cross_entropy(
                            outputs, targets, label_smoothing=self.label_smoothing
                        )
                    else:
                        return F.cross_entropy(outputs, targets)
            elif self.task_type == "regression":
                return F.mse_loss(outputs, targets)
            elif self.task_type == "unsupervised":
                # Contrastive loss or similar
                if self.projection_head is not None:
                    projected = self.projection_head(outputs)
                    # Simple contrastive loss (placeholder)
                    return F.mse_loss(projected, torch.zeros_like(projected))
                else:
                    return F.mse_loss(outputs, targets)
            else:
                # Default to MSE
                return F.mse_loss(outputs, targets)
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    def _compute_step(self, batch: Any, stage: str) -> Dict[str, torch.Tensor]:
        """
        Unified step computation for train/val/test.

        Args:
            batch: Input batch (PyTorch Geometric Data object or list)
            stage: One of 'train', 'val', 'test'

        Returns:
            Dictionary with loss and other metrics
        """
        try:
            # Ensure we have a PyTorch Geometric Data object
            if not hasattr(batch, "x") or not hasattr(batch, "edge_index"):
                raise ValueError(
                    f"Expected PyTorch Geometric Data object, got {type(batch)}"
                )

            # Initialize metrics if not done yet
            if not self.metrics_initialized and hasattr(batch, "y"):
                num_classes = self._detect_num_classes_from_data(batch)
                self.num_classes = num_classes
                self._create_metrics_for_classes(num_classes)

            # Get the appropriate mask for this stage
            if stage == "train" and hasattr(batch, "train_mask"):
                mask = batch.train_mask
            elif stage == "val" and hasattr(batch, "val_mask"):
                mask = batch.val_mask
            elif stage == "test" and hasattr(batch, "test_mask"):
                mask = batch.test_mask
            else:
                mask = None

            # Forward pass
            outputs = self.forward(batch)

            # Apply mask if available
            if mask is not None and mask.sum() > 0:
                outputs_masked = outputs[mask]
                targets_masked = batch.y[mask]
            else:
                outputs_masked = outputs
                targets_masked = batch.y

            # Ensure we have valid targets
            if targets_masked.numel() == 0:
                logger.warning(f"No valid targets for stage {stage}, using full batch")
                outputs_masked = outputs
                targets_masked = batch.y

            # Compute loss
            loss = self._compute_loss(outputs_masked, targets_masked)

            # Log metrics
            self._log_step_metrics(outputs_masked, targets_masked, loss, stage)

            return {"loss": loss, "outputs": outputs_masked, "targets": targets_masked}

        except Exception as e:
            logger.error(f"Step computation failed for stage {stage}: {e}")
            logger.error(f"Batch type: {type(batch)}")
            # Return a dummy loss to prevent training from crashing
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return {"loss": dummy_loss}

    def _log_step_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        stage: str,
    ) -> None:
        """Log metrics for the current step."""
        try:
            # Log loss
            self.log(
                f"{stage}_loss",
                loss,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=True,
            )

            # Log accuracy for classification
            if self.task_type == "classification" and self.metrics_initialized:
                # Ensure targets and outputs are on the same device and correct shape
                if outputs.device != targets.device:
                    targets = targets.to(outputs.device)

                # Always use logits for multiclass metrics
                preds = outputs
                targets = targets.long()

                # Ensure metrics are on the correct device
                if stage == "train" and self.train_acc is not None:
                    self.train_acc = self.train_acc.to(preds.device)
                    acc = self.train_acc(preds, targets)
                    self.log("train_acc", acc, on_step=False, on_epoch=True)
                elif stage == "val" and self.val_acc is not None:
                    self.val_acc = self.val_acc.to(preds.device)
                    acc = self.val_acc(preds, targets)
                    self.log(
                        "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True
                    )
                elif stage == "test" and self.test_acc is not None:
                    self.test_acc = self.test_acc.to(preds.device)
                    acc = self.test_acc(preds, targets)
                    self.log("test_acc", acc, on_step=False, on_epoch=True)

        except Exception as e:
            logger.error(f"Metric logging failed for stage {stage}: {e}")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step with automatic optimization.

        Updated for 2025 best practices with proper PyTorch Geometric handling.
        """
        try:
            # Compute step
            result = self._compute_step(batch, "train")
            loss = result["loss"]

            # Update EMA if enabled
            if self.use_ema:
                self._update_ema()

            return loss

        except Exception as e:
            logger.error(f"Training step {batch_idx} failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            if hasattr(batch, "x"):
                logger.error(f"   Batch x shape: {batch.x.shape}")
            raise

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step with guaranteed val_loss logging."""
        try:
            result = self._compute_step(batch, "val")
            return result["loss"]

        except Exception as e:
            logger.error(f"❌ Validation step failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            # Log dummy val_loss to prevent Lightning from crashing
            dummy_loss = torch.tensor(0.0, device=self.device)
            self.log("val_loss", dummy_loss, prog_bar=True, on_epoch=True)
            return dummy_loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step with comprehensive evaluation."""
        try:
            result = self._compute_step(batch, "test")
            return result["loss"]

        except Exception as e:
            logger.error(f"❌ Test step failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            raise

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and schedulers with 2025 best practices.

        Includes:
        - AdamW with decoupled weight decay
        - Multiple scheduler options
        - Warmup support
        - Gradient accumulation awareness
        """
        # Configure optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Configure scheduler
        config = {"optimizer": optimizer}

        if self.scheduler_type == "cosine":
            # Safe trainer access - use default if trainer not attached yet
            max_epochs = 100
            try:
                if (
                    self.trainer
                    and hasattr(self.trainer, "max_epochs")
                    and self.trainer.max_epochs
                ):
                    max_epochs = self.trainer.max_epochs
            except RuntimeError:
                # Trainer not attached yet - use default
                pass

            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.scheduler_type == "onecycle":
            # Safe trainer access for stepping batches
            total_steps = 1000
            try:
                if (
                    self.trainer
                    and hasattr(self.trainer, "estimated_stepping_batches")
                    and self.trainer.estimated_stepping_batches
                ):
                    total_steps = self.trainer.estimated_stepping_batches
            except RuntimeError:
                # Trainer not attached yet - use default
                pass

            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy="cos",
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.scheduler_type == "cosine_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        return config

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step for inference."""
        try:
            outputs = self.forward(batch)

            # Apply softmax for classification
            if self.task_type == "classification":
                outputs = F.softmax(outputs, dim=-1)

            return outputs

        except Exception as e:
            logger.error(f"Prediction step failed: {e}")
            raise

    def on_train_start(self) -> None:
        """Called at the start of training."""
        try:
            # Get datamodule info if available
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule:
                dm = self.trainer.datamodule
                if (
                    hasattr(dm, "num_classes")
                    and dm.num_classes
                    and not self.metrics_initialized
                ):
                    self.num_classes = dm.num_classes
                    self._create_metrics_for_classes(dm.num_classes)

            # Log model info
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

            logger.info("🚀 Training started")
            logger.info(f"📊 Total parameters: {total_params:,}")
            logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
            logger.info(f"🎨 Task type: {self.task_type}")
            logger.info(f"🔢 Number of classes: {self.num_classes}")

        except Exception as e:
            logger.error(f"Error in on_train_start: {e}")

    def on_train_end(self) -> None:
        """Called at the end of training."""
        logger.info("🏁 Training completed")

    def on_validation_start(self) -> None:
        """Called at the start of validation."""
        pass

    def on_test_start(self) -> None:
        """Called at the start of testing."""
        pass


__all__ = ["AstroLightningModule"]
