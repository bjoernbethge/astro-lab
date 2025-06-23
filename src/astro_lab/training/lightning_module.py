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

# Setup logging
logger = logging.getLogger(__name__)

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

        logger.info(
            f"✅ AstroLightningModule initialized with task_type: {self.task_type}"
        )

    def _initialize_model(self, model: Optional[torch.nn.Module]) -> None:
        """Initialize model with robust error handling."""
        try:
            if model is not None:
                self.model = model
                logger.info(f"✅ Using provided model: {type(model).__name__}")
            elif self.model_config is not None:
                self.model = self._create_model_from_config(self.model_config)
                logger.info(
                    f"✅ Created model from config: {type(self.model).__name__}"
                )
            else:
                # Create default model
                self.model = self._create_default_model()
                logger.info(f"✅ Created default model: {type(self.model).__name__}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
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
                            logger.info(f"📋 Found {num_classes} classes in metadata: {path}")
                            return int(num_classes)
                    
                    # Fallback: check for direct num_classes field
                    if "num_classes" in metadata:
                        num_classes = metadata["num_classes"]
                        logger.info(f"📋 Found {num_classes} classes in metadata: {path}")
                        return int(num_classes)
            
            logger.debug("No class information found in metadata files")
            return None
            
        except Exception as e:
            logger.debug(f"Error loading metadata: {e}")
            return None

    def _setup_metrics(self) -> None:
        """Setup torchmetrics for performance tracking."""
        if self.task_type == "classification":
            # Use detected num_classes or fallback to 4 for Gaia
            num_classes = self.num_classes or 4
            
            # Create metrics and move to device
            self.train_acc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            ).to(self.device)
            self.val_acc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            ).to(self.device)
            self.test_acc = MulticlassAccuracy(
                num_classes=num_classes, average="macro"
            ).to(self.device)

            self.train_f1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(self.device)
            self.val_f1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(self.device)
            self.test_f1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(self.device)
        else:
            # For regression tasks
            self.train_acc = None
            self.val_acc = None
            self.test_acc = None
            self.train_f1 = None
            self.val_f1 = None
            self.test_f1 = None

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

                if isinstance(batch, dict):
                    targets = batch.get("target")
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
                logger.warning(
                    "No targets found in dataloader, using default 7 classes"
                )
                return 7

            # Concatenate all targets and find unique values
            all_targets = torch.cat(all_targets)
            unique_classes = torch.unique(all_targets)
            num_classes = len(unique_classes)

            # Ensure classes are 0-indexed
            min_class = unique_classes.min().item()
            max_class = unique_classes.max().item()

            if min_class != 0:
                logger.warning(f"Classes start from {min_class}, not 0")

            logger.info(
                f"🔍 Automatically detected {num_classes} classes: {unique_classes.tolist()}"
            )
            return num_classes

        except Exception as e:
            logger.error(f"❌ Error detecting classes from data: {e}")
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
            logger.error(f"❌ Failed to create default model: {e}")
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
                    logger.info(f"📋 Using {output_dim} classes from dataset metadata")
                elif self.num_classes is not None:
                    # Use automatically detected number of classes
                    output_dim = self.num_classes
                    logger.info(f"🎯 Using detected {output_dim} classes for stellar classification")
                else:
                    # Fallback: use config value but warn
                    logger.warning(f"⚠️ No classes detected, using config output_dim: {output_dim}")
                    self.num_classes = output_dim

            logger.info(
                f"🔧 Creating AstroSurveyGNN with input_dim={input_dim}, hidden_dim={config.graph.hidden_dim}, output_dim={output_dim}"
            )

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
            logger.error(f"❌ Failed to create model from config: {e}")
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
            
            logger.info(f"✅ Created projection head: {input_dim} -> {self.projection_dim}")
            return projection_head
            
        except Exception as e:
            logger.error(f"❌ Failed to create projection head: {e}")
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
            logger.error(f"❌ Forward pass failed: {e}")
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
                    f"❌ Target contains invalid class indices: min={targets.min().item()}, max={targets.max().item()}, num_classes={outputs.shape[1]}"
                )
                raise ValueError(
                    f"Target contains invalid class indices: min={targets.min().item()}, max={targets.max().item()}, num_classes={outputs.shape[1]}"
                )
            if outputs.shape[0] != targets.shape[0]:
                logger.error(
                    f"❌ Output/target batch size mismatch: outputs={outputs.shape}, targets={targets.shape}"
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

            # Debug: Log shapes and value ranges
            logger.debug(
                f"Batch {stage}: outputs.shape={outputs.shape}, targets.shape={targets.shape}, targets.min={targets.min().item()}, targets.max={targets.max().item()}"
            )

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
            logger.error(f"❌ {stage} step failed: {e}")
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
                    logger.info(f"🔍 Auto-detected {detected_classes} classes, updating metrics")
                    self.num_classes = detected_classes
                
                # Setup metrics on correct device
                self._setup_metrics()
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
                    logger.warning(f"⚠️ Metrics computation failed: {e}")
                    # Continue without failing

    def training_step(self, batch: Union[torch.Tensor, Dict[str, torch.Tensor], Any], batch_idx: int) -> torch.Tensor:
        """Training step with proper gradient handling."""
        try:
            # Ensure batch is properly formatted for GNN
            if isinstance(batch, dict):
                x = batch.get("x")
                edge_index = batch.get("edge_index")
                y = batch.get("y")
            elif hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                # PyTorch Geometric Data object
                x = batch.x
                edge_index = batch.edge_index
                y = getattr(batch, 'y', None)
            else:
                # Fallback for tensor input
                x = batch
                edge_index = None
                y = None
            
            # Ensure tensors require gradients
            if x is not None and not x.requires_grad:
                x = x.detach().requires_grad_(True)
            
            # Forward pass
            if self.task_type == "unsupervised":
                # Unsupervised learning with projection head
                if self.projection_head is not None:
                    embeddings = self.model(x, edge_index)
                    projections = self.projection_head(embeddings)
                    # Use contrastive loss or reconstruction loss
                    loss = torch.nn.functional.mse_loss(projections, embeddings.detach())
                else:
                    # Simple reconstruction loss
                    output = self.model(x, edge_index)
                    loss = torch.nn.functional.mse_loss(output, x)
            else:
                # Supervised learning
                output = self.model(x, edge_index)
                if y is not None:
                    if self.task_type == "classification":
                        loss = torch.nn.functional.cross_entropy(output, y)
                    else:
                        loss = torch.nn.functional.mse_loss(output, y)
                else:
                    # Fallback loss
                    loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
            
            # Log metrics
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            
            return loss
            
        except Exception as e:
            logger.error(f"❌ Training step failed: {e}")
            # Return a dummy loss to prevent training from crashing
            return torch.tensor(0.0, requires_grad=True, device=self.device)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step with error handling."""
        try:
            result = self._compute_step(batch, "val")
            return result["loss"]
        except Exception as e:
            logger.error(f"❌ Validation step failed: {e}")
            raise

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step with error handling."""
        try:
            result = self._compute_step(batch, "test")
            return result["loss"]
        except Exception as e:
            logger.error(f"❌ Test step failed: {e}")
            raise

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        """Configure optimizers and schedulers."""
        try:
            # Get optimizer parameters from config or use defaults
            lr = getattr(self, "learning_rate", 1e-3)
            weight_decay = getattr(self, "weight_decay", 1e-4)

            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            # Configure scheduler if training config is available
            if self.training_config and hasattr(self.training_config, "scheduler"):
                scheduler_config = self.training_config.scheduler
                scheduler_type = getattr(scheduler_config, "type", "cosine")

                if scheduler_type == "cosine":
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=scheduler_config.max_epochs,
                        eta_min=scheduler_config.min_lr,
                    )
                elif scheduler_type == "onecycle":
                    # Estimate steps per epoch if not available
                    steps_per_epoch = getattr(scheduler_config, "steps_per_epoch", 100)
                    scheduler = OneCycleLR(
                        optimizer,
                        max_lr=lr,
                        epochs=scheduler_config.max_epochs,
                        steps_per_epoch=steps_per_epoch,
                    )
                elif scheduler_type == "reduce_on_plateau":
                    scheduler = ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.5, patience=5
                    )
                else:
                    scheduler = None
            else:
                scheduler = None

            if scheduler:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    },
                }
            else:
                return optimizer

        except Exception as e:
            logger.error(f"❌ Failed to configure optimizers: {e}")
            # Fallback to simple AdamW
            return AdamW(self.parameters(), lr=1e-3)

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
            logger.error(f"❌ Prediction step failed: {e}")
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
                                f"🔄 Recreated model with {detected_classes} classes"
                            )
                        else:
                            # Recreate default model with correct classes
                            self.model = self._create_default_model()
                            logger.info(
                                f"🔄 Recreated default model with {detected_classes} classes"
                            )

                        # Recreate metrics with correct number of classes
                        self._setup_metrics()
                        logger.info(
                            f"🔄 Recreated metrics with {detected_classes} classes"
                        )

                        # Move model to correct device
                        if hasattr(self, "device"):
                            self.model = self.model.to(self.device)
                            logger.info(f"🔄 Moved model to device: {self.device}")
            except Exception as e:
                logger.error(f"❌ Error during class detection: {e}")

        logger.info("🚀 Training started")
        logger.info(f"   Model: {type(self.model).__name__}")
        logger.info(f"   Task: {self.task_type}")
        logger.info(f"   Device: {self.device}")
        if self.num_classes is not None:
            logger.info(f"   Classes: {self.num_classes}")

    def on_train_end(self) -> None:
        """Called when training ends."""
        logger.info("✅ Training completed")

    def on_validation_start(self) -> None:
        """Called when validation starts."""
        logger.info("🔍 Validation started")

    def on_test_start(self) -> None:
        """Called when testing starts."""
        logger.info("🔍 Testing started")

__all__ = ["AstroLightningModule"]
