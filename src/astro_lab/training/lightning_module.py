"""
AstroLightningModule - Unified PyTorch Lightning Module for Astronomical ML
=======================================================================

Provides a robust, configurable Lightning module with unified logging,
error handling, and support for various astronomical tasks.
Optimized for Lightning 2.0+ compatibility and modern ML practices.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import MulticlassAccuracy

# Import real model classes
from astro_lab.models.astro import AstroSurveyGNN
from astro_lab.models.astrophot_models import AstroPhotGNN
from astro_lab.models.config import ModelConfig
from astro_lab.models.factory import create_gaia_classifier
from astro_lab.models.tgnn import ALCDEFTemporalGNN
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

        # Initialize model with robust error handling
        self._initialize_model(model)

        # Initialize projection head for unsupervised learning
        self.projection_head = None
        if task_type == "unsupervised":
            self.projection_head = self._auto_create_projection_head()

        # Initialize metrics for tracking (will be set up after class detection)
        self._setup_metrics()
        self.metrics_initialized = True

        # Performance tracking
        self._step_times = []
        self._memory_usage = []

    def _initialize_model(self, model: Optional[torch.nn.Module]) -> None:
        """Initialize model with robust error handling."""
        try:
            if model is not None:
                self.model = model
            elif self.model_config is not None:
                self.model = self._create_model_from_config(self.model_config)
            else:
                # Create default model
                self.model = self._create_default_model()
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

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
                    with open(path, 'r') as f:
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
        if self.task_type == "classification":
            # Don't initialize metrics here - we'll do it dynamically
            # when we see the actual data
            self.train_acc = None
            self.val_acc = None
            self.test_acc = None
            self.train_f1 = None
            self.val_f1 = None
            self.test_f1 = None
        else:
            # For regression tasks
            self.train_acc = None
            self.val_acc = None
            self.test_acc = None
            self.train_f1 = None
            self.val_f1 = None
            self.test_f1 = None
            
    def _create_metrics_for_classes(self, num_classes: int, device: torch.device) -> None:
        """Create metrics with the correct number of classes. 🎯"""
        if self.task_type == "classification":
            # Create metrics with detected number of classes
            self.train_acc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.val_acc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.test_acc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            ).to(device)

            self.train_f1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(device)
            self.val_f1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(device)
            self.test_f1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(device)

    def _detect_num_classes_from_data(self, dataloader) -> int:
        """
        Automatically detect number of classes from the dataloader.

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            Number of unique classes
        """
        try:
            all_targets = []

            # Sample a few batches to determine class count
            for i, batch in enumerate(dataloader):
                if i >= 10:  # Limit to first 10 batches for efficiency
                    break

                # Handle PyTorch Geometric Data objects specifically
                if hasattr(batch, '__class__') and batch.__class__.__name__ == 'DataBatch':
                    # This is a batched PyG data object
                    if hasattr(batch, 'y'):
                        targets = batch.y
                    else:
                        continue
                elif hasattr(batch, 'y') and hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                    # Single PyG Data object
                    targets = batch.y
                elif isinstance(batch, list) and len(batch) > 0:
                    # List of PyG Data objects
                    if hasattr(batch[0], 'y'):
                        targets = batch[0].y
                    else:
                        continue
                elif isinstance(batch, dict):
                    targets = batch.get("target") or batch.get("y")
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    targets = batch[1]  # Assume (data, target) format
                else:
                    targets = batch

                if targets is not None:
                    if isinstance(targets, torch.Tensor):
                        all_targets.append(targets.flatten())
                    else:
                        all_targets.append(torch.tensor(targets).flatten())

            if not all_targets:
                return 7

            # Concatenate all targets and find unique values
            all_targets = torch.cat(all_targets)
            unique_classes = torch.unique(all_targets)
            num_classes = len(unique_classes)

            return num_classes

        except Exception as e:
            logger.error(f"Error detecting classes from data: {e}")
            return 7

    def _create_default_model(self) -> torch.nn.Module:
        """Create default model with logging."""
        try:
            from astro_lab.models.config import (
                EncoderConfig,
                GraphConfig,
                ModelConfig,
                OutputConfig,
            )

            # Use detected num_classes or fallback
            output_dim = self.num_classes or 7

            default_config = ModelConfig(
                name="default_gaia",
                description="Default Gaia survey configuration",
                encoder=EncoderConfig(
                    use_photometry=True,
                    use_astrometry=True,
                    use_spectroscopy=False,
                    photometry_dim=64,
                    astrometry_dim=64,
                ),
                graph=GraphConfig(
                    conv_type="gcn",
                    hidden_dim=128,
                    num_layers=3,
                    dropout=0.1,
                ),
                output=OutputConfig(
                    task="stellar_classification",
                    output_dim=output_dim,
                    pooling="mean",
                ),
            )
            self.model_config = default_config
            return self._create_model_from_config(default_config)
        except Exception as e:
            logger.error(f"Failed to create default model: {e}")
            raise

    def _create_model_from_config(self, config: ModelConfig) -> torch.nn.Module:
        """Create model from ModelConfig with error handling."""
        try:
            # Calculate input_dim from EncoderConfig
            encoder = config.encoder
            input_dim = 0
            if encoder.use_photometry:
                input_dim += encoder.photometry_dim
            if encoder.use_astrometry:
                input_dim += encoder.astrometry_dim
            if encoder.use_spectroscopy:
                input_dim += encoder.spectroscopy_dim
            if encoder.use_lightcurve:
                input_dim += encoder.lightcurve_dim
            if encoder.use_spatial_3d:
                input_dim += encoder.spatial_3d_dim
            if input_dim == 0:
                input_dim = 16  # Fallback

            # Use detected num_classes for classification tasks
            output_dim = config.output.output_dim
            if config.output.task == "stellar_classification":
                # Try to load num_classes from dataset metadata first
                metadata_classes = self._load_num_classes_from_metadata()
                if metadata_classes is not None:
                    output_dim = metadata_classes
                    self.num_classes = metadata_classes
                elif self.num_classes is not None:
                    # Use automatically detected number of classes
                    output_dim = self.num_classes
                else:
                    # Fallback: use config value
                    self.num_classes = output_dim

            return AstroSurveyGNN(
                input_dim=input_dim,
                hidden_dim=config.graph.hidden_dim,
                output_dim=output_dim,
                conv_type=config.graph.conv_type,
                num_layers=config.graph.num_layers,
                dropout=config.graph.dropout,
                task=config.output.task,
                use_photometry=config.encoder.use_photometry,
                use_astrometry=config.encoder.use_astrometry,
                use_spectroscopy=config.encoder.use_spectroscopy,
                use_lightcurve=config.encoder.use_lightcurve,
                use_spatial_3d=config.encoder.use_spatial_3d,
            )
        except Exception as e:
            logger.error(f"Failed to create model from config: {e}")
            raise

    def _auto_create_projection_head(self) -> Optional[torch.nn.Module]:
        """Create projection head for unsupervised learning."""
        try:
            if self.model is None:
                return None
            
            # Get input dimension from model
            input_dim = 128  # Default
            if hasattr(self.model, 'input_dim'):
                input_dim = int(self.model.input_dim)
            elif hasattr(self.model, 'hidden_dim'):
                input_dim = int(self.model.hidden_dim)
            
            # Create simple projection head
            projection_head = torch.nn.Sequential(
                torch.nn.Linear(input_dim, self.projection_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.projection_dim, self.projection_dim)
            )
            
            return projection_head
            
        except Exception as e:
            logger.error(f"Failed to create projection head: {e}")
            return None

    def forward(
        self, batch: Union[torch.Tensor, Dict[str, torch.Tensor], Any]
    ) -> torch.Tensor:
        """Forward pass with robust error handling and logging."""
        try:
            # Handle different batch formats
            if isinstance(batch, torch.Tensor):
                x = batch
                edge_index = None
                batch_tensor = None
            elif hasattr(batch, "x"):
                x, edge_index = batch.x, batch.edge_index
                batch_tensor = getattr(batch, "batch", None)
            elif isinstance(batch, dict):
                x = batch.get("x")
                edge_index = batch.get("edge_index", None)
                batch_tensor = batch.get("batch", None)
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            # Forward pass through model
            if edge_index is not None:
                output = self.model(x, edge_index, batch_tensor)
            else:
                output = self.model(x)

            return output

        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            logger.error(f"   Batch shape: {getattr(batch, 'shape', 'N/A')}")
            raise

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on task type."""
        # Ensure both tensors are on the same device
        if outputs.device != targets.device:
            targets = targets.to(outputs.device)
            
        if self.task_type == "classification":
            # Ensure targets are long for classification
            if targets.dtype != torch.long:
                targets = targets.long()
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            # Check for invalid class indices
            if torch.any(targets < 0) or torch.any(targets >= outputs.shape[1]):
                logger.error(
                    f"Target contains invalid class indices: min={targets.min().item()}, max={targets.max().item()}, num_classes={outputs.shape[1]}"
                )
                raise ValueError(
                    f"Target contains invalid class indices: min={targets.min().item()}, max={targets.max().item()}, num_classes={outputs.shape[1]}"
                )
            if outputs.shape[0] != targets.shape[0]:
                logger.error(
                    f"Output/target batch size mismatch: outputs={outputs.shape}, targets={targets.shape}"
                )
                raise ValueError(
                    f"Output/target batch size mismatch: outputs={outputs.shape}, targets={targets.shape}"
                )
            return F.cross_entropy(outputs, targets)
        elif self.task_type == "regression":
            return F.mse_loss(outputs, targets)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _compute_step(self, batch: Any, stage: str) -> Dict[str, torch.Tensor]:
        """Compute training/validation/test step with unified logic."""
        try:
            # Extract data from batch
            if hasattr(batch, "x"):
                x, edge_index = batch.x, batch.edge_index
                batch_tensor = getattr(batch, "batch", None)
                targets = getattr(batch, "y", None)
            elif isinstance(batch, dict):
                x = batch["x"]
                edge_index = batch.get("edge_index", None)
                batch_tensor = batch.get("batch", None)
                targets = batch.get("y", None)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            # Forward pass
            if edge_index is not None:
                outputs = self.model(x, edge_index, batch_tensor)
            else:
                outputs = self.model(x)

            # Handle missing targets for unsupervised tasks
            if targets is None:
                loss = torch.tensor(0.0, device=self.device)
                return {"loss": loss}

            # Ensure targets have correct shape and type
            if targets.dim() > 1 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            if targets.dim() > 1 and outputs.dim() > 1:
                # Multiple targets per sample - take mean (regression fallback)
                targets = targets.mean(dim=1)

            # Compute loss
            loss = self._compute_loss(outputs, targets)

            # Log metrics
            self._log_step_metrics(outputs, targets, loss, stage)

            return {
                "loss": loss,
                "outputs": outputs,
                "targets": targets,
            }

        except Exception as e:
            logger.error(f"{stage} step failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            logger.error(
                f"   Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'}"
            )
            if "targets" in locals():
                logger.error(
                    f"   targets.shape: {getattr(targets, 'shape', None)}, targets: {targets}"
                )
            if "outputs" in locals():
                logger.error(f"   outputs.shape: {getattr(outputs, 'shape', None)}")
            raise

    def _log_step_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        stage: str,
    ) -> None:
        """Log metrics for the current step."""
        # Log loss
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)

        # Log accuracy for classification
        if self.task_type == "classification":
            # Auto-detect and setup metrics if needed
            if not self.metrics_initialized or outputs.dim() > 1 and outputs.size(1) != self.num_classes:
                detected_classes = outputs.size(1) if outputs.dim() > 1 else 2
                if self.num_classes != detected_classes:
                    logger.info(f"Auto-detected {detected_classes} classes, updating metrics")
                    self.num_classes = detected_classes
                
                # Setup metrics on correct device
                self._create_metrics_for_classes(self.num_classes, self.device)
                self.metrics_initialized = True
            
            # Use metrics if they exist
            if hasattr(self, f"{stage}_acc"):
                try:
                    acc_metric = getattr(self, f"{stage}_acc")
                    acc = acc_metric(outputs, targets)
                    self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

                    # Log F1 score
                    if hasattr(self, f"{stage}_f1"):
                        f1_metric = getattr(self, f"{stage}_f1")
                        f1 = f1_metric(outputs, targets)
                        self.log(f"{stage}_f1", f1, sync_dist=True)
                except Exception as e:
                    logger.warning(f"Metrics computation failed: {e}")
                    # Continue without failing

    def training_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor], Any], batch_idx: int) -> torch.Tensor:
        """Training step with proper gradient handling. 🚂"""
        try:
            # Handle PyTorch Geometric Data objects
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index') and hasattr(batch, 'y'):
                # Direct PyG Data object
                x = batch.x
                edge_index = batch.edge_index
                y = batch.y
                
                # For node classification with masks
                if hasattr(batch, 'train_mask'):
                    # Apply mask to get only training nodes
                    train_mask = batch.train_mask
                    x_train = x[train_mask]
                    y_train = y[train_mask]
                    
                    # Forward pass through model
                    output = self.model(x, edge_index)
                    
                    # Get predictions for training nodes only
                    output_train = output[train_mask]
                    
                    # Compute loss on training nodes
                    if self.task_type == "classification":
                        loss = F.cross_entropy(output_train, y_train)
                    else:
                        loss = F.mse_loss(output_train, y_train)
                else:
                    # Full graph without masks
                    output = self.model(x, edge_index)
                    if self.task_type == "classification":
                        loss = F.cross_entropy(output, y)
                    else:
                        loss = F.mse_loss(output, y)
                        
                # Log metrics
                self.log("train_loss", loss, prog_bar=True)
                if self.task_type == "classification":
                    with torch.no_grad():
                        # Lazy initialize metrics if needed
                        if self.train_acc is None:
                            if hasattr(batch, 'train_mask'):
                                num_classes = output_train.shape[1]
                            else:
                                num_classes = output.shape[1]
                            self._create_metrics_for_classes(num_classes, loss.device)
                            self.metrics_initialized = True
                            
                        if self.train_acc:
                            if hasattr(batch, 'train_mask'):
                                acc = self.train_acc(output_train, y_train)
                            else:
                                acc = self.train_acc(output, y)
                            self.log("train_acc", acc, prog_bar=True)
                        
                return loss
                
            else:
                # Fallback to generic compute_step
                result = self._compute_step(batch, "train")
                # IMPORTANT: Always return just the loss tensor
                return result["loss"]
                
        except Exception as e:
            logger.error(f"❌ Training step failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            if hasattr(batch, '__dict__'):
                logger.error(f"   Batch attributes: {list(batch.__dict__.keys())}")
            raise

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step with mask handling. 🧪"""
        try:
            # Handle PyTorch Geometric Data objects with validation masks
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index') and hasattr(batch, 'y'):
                x = batch.x
                edge_index = batch.edge_index
                y = batch.y
                
                # For node classification with masks
                if hasattr(batch, 'val_mask'):
                    val_mask = batch.val_mask
                    
                    # Forward pass through entire graph
                    output = self.model(x, edge_index)
                    
                    # Get predictions and targets for validation nodes only
                    output_val = output[val_mask]
                    y_val = y[val_mask]
                    
                    # Compute loss on validation nodes
                    if self.task_type == "classification":
                        loss = F.cross_entropy(output_val, y_val)
                    else:
                        loss = F.mse_loss(output_val, y_val)
                    
                    # Log metrics
                    self.log("val_loss", loss, prog_bar=True)
                    if self.task_type == "classification" and self.val_acc:
                        with torch.no_grad():
                            acc = self.val_acc(output_val, y_val)
                            self.log("val_acc", acc, prog_bar=True)
                            if self.val_f1:
                                f1 = self.val_f1(output_val, y_val)
                                self.log("val_f1", f1)
                else:
                    # Full graph without masks
                    output = self.model(x, edge_index)
                    if self.task_type == "classification":
                        loss = F.cross_entropy(output, y)
                    else:
                        loss = F.mse_loss(output, y)
                    
                    # Log metrics
                    self.log("val_loss", loss, prog_bar=True)
                    if self.task_type == "classification" and self.val_acc:
                        with torch.no_grad():
                            acc = self.val_acc(output, y)
                            self.log("val_acc", acc, prog_bar=True)
                            
                return loss
                
            else:
                # Fallback to generic compute_step
                result = self._compute_step(batch, "val")
                return result["loss"]
                
        except Exception as e:
            logger.error(f"❌ Validation step failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            raise

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step with comprehensive evaluation. 🎯"""
        try:
            # Handle PyTorch Geometric Data objects with test masks
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index') and hasattr(batch, 'y'):
                x = batch.x
                edge_index = batch.edge_index
                y = batch.y
                
                # For node classification with masks
                if hasattr(batch, 'test_mask'):
                    test_mask = batch.test_mask
                    
                    # Forward pass through entire graph
                    output = self.model(x, edge_index)
                    
                    # Get predictions and targets for test nodes only
                    output_test = output[test_mask]
                    y_test = y[test_mask]
                    
                    # Compute loss on test nodes
                    if self.task_type == "classification":
                        loss = F.cross_entropy(output_test, y_test)
                    else:
                        loss = F.mse_loss(output_test, y_test)
                    
                    # Log metrics
                    self.log("test_loss", loss)
                    if self.task_type == "classification" and self.test_acc:
                        with torch.no_grad():
                            acc = self.test_acc(output_test, y_test)
                            self.log("test_acc", acc)
                            if self.test_f1:
                                f1 = self.test_f1(output_test, y_test)
                                self.log("test_f1", f1)
                else:
                    # Full graph without masks
                    output = self.model(x, edge_index)
                    if self.task_type == "classification":
                        loss = F.cross_entropy(output, y)
                    else:
                        loss = F.mse_loss(output, y)
                    
                    # Log metrics
                    self.log("test_loss", loss)
                    if self.task_type == "classification" and self.test_acc:
                        with torch.no_grad():
                            acc = self.test_acc(output, y)
                            self.log("test_acc", acc)
                            
                return loss
                
            else:
                # Fallback to generic compute_step
                result = self._compute_step(batch, "test")
                return result["loss"]
                
        except Exception as e:
            logger.error(f"❌ Test step failed: {e}")
            logger.error(f"   Batch type: {type(batch)}")
            raise

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and learning rate schedulers. 🔧"""
        # Simple optimizer configuration to avoid "No inf checks" error
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step with error handling."""
        try:
            with torch.no_grad():
                if hasattr(batch, "x"):
                    x, edge_index = batch.x, batch.edge_index
                    batch_tensor = getattr(batch, "batch", None)
                elif isinstance(batch, dict):
                    x = batch["x"]
                    edge_index = batch.get("edge_index", None)
                    batch_tensor = batch.get("batch", None)
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                if edge_index is not None:
                    outputs = self.model(x, edge_index, batch_tensor)
                else:
                    outputs = self.model(x)

                return outputs

        except Exception as e:
            logger.error(f"Prediction step failed: {e}")
            raise

    def on_train_start(self) -> None:
        """Called when training starts - detect classes if needed."""
        super().on_train_start()

        # Auto-detect classes if not set and we have a dataloader
        if self.num_classes is None and self.task_type == "classification":
            try:
                if hasattr(self, "trainer") and self.trainer is not None:
                    dataloader = self.trainer.train_dataloader
                    if dataloader is not None:
                        detected_classes = self._detect_num_classes_from_data(
                            dataloader
                        )
                        self.num_classes = detected_classes

                        # Recreate model with correct number of classes
                        if self.model_config is not None:
                            self.model = self._create_model_from_config(
                                self.model_config
                            )
                            logger.info(
                                f"Recreated model with {detected_classes} classes"
                            )
                        else:
                            # Recreate default model with correct classes
                            self.model = self._create_default_model()
                            logger.info(
                                f"Recreated default model with {detected_classes} classes"
                            )

                            # Recreate metrics with correct number of classes
                            self._create_metrics_for_classes(self.num_classes, self.device)
                            logger.info(
                                f"Recreated metrics with {detected_classes} classes"
                            )

                            # Move model to correct device
                            if hasattr(self, "device"):
                                self.model = self.model.to(self.device)
                                logger.info(f"Moved model to device: {self.device}")
            except Exception as e:
                logger.error(f"Error during class detection: {e}")

        logger.info("Training started")
        logger.info(f"   Model: {type(self.model).__name__}")
        logger.info(f"   Task: {self.task_type}")
        logger.info(f"   Device: {self.device}")
        if self.num_classes is not None:
            logger.info(f"   Classes: {self.num_classes}")

    def on_train_end(self) -> None:
        """Called when training ends."""
        logger.info("Training completed")

    def on_validation_start(self) -> None:
        """Called when validation starts."""
        logger.info("Validation started")

    def on_test_start(self) -> None:
        """Called when testing starts."""
        logger.info("Testing started")

__all__ = ["AstroLightningModule"]
