"""
AstroLab Trainer for Lightning Models - Optimized 2025
=====================================================

Unified trainer class for training AstroLab Lightning models.
Optimized for PyTorch Lightning 2.x and modern GPUs with best practices from 2025.

Implements:
- Automatic memory pinning and DataLoader optimization
- Smart model-data compatibility checks
- Unified model type mapping
- GPU-aware training configuration
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.strategies import DDPStrategy

from astro_lab.data import AstroDataModule
from astro_lab.models import (
    create_model_for_task,
    create_model_from_preset,
    get_available_presets,
    get_model_type_for_task,
)

from .mlflow_logger import LightningMLflowLogger

logger = logging.getLogger(__name__)


class AstroTrainer:
    """
    Unified trainer for AstroLab Lightning models with 2025 optimizations.

    Features:
    - Automatic DataLoader optimization (pin_memory, persistent_workers)
    - Smart GPU detection and configuration
    - Unified model type handling
    - Memory-efficient training setup
    """

    # Unified model type mapping
    MODEL_TYPE_MAPPING = {
        # Canonical names
        "node": "node",
        "graph": "graph",
        "temporal": "temporal",
        "point": "point",
        # Alternative names
        "astro_node_gnn": "node",
        "astro_graph_gnn": "graph",
        "astro_temporal_gnn": "temporal",
        "astro_pointnet": "point",
        "AstroNodeGNN": "node",
        "AstroGraphGNN": "graph",
        "AstroTemporalGNN": "temporal",
        "AstroPointNet": "point",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AstroTrainer with configuration.

        Args:
            config: Training configuration dictionary
        """
        self.config = config.copy()

        # Normalize model type
        if "model_type" in self.config:
            original_type = self.config["model_type"]
            self.config["model_type"] = self.MODEL_TYPE_MAPPING.get(
                original_type, original_type
            )
            if original_type != self.config["model_type"]:
                logger.info(
                    f"Normalized model_type: {original_type} -> {self.config['model_type']}"
                )

        # Auto-detect GPU and set optimal defaults
        self.use_gpu = (
            torch.cuda.is_available() and self.config.get("accelerator", "gpu") == "gpu"
        )
        self._set_optimal_defaults()

        logger.info(
            f"[AstroTrainer] Initialized with config keys: {list(self.config.keys())}"
        )
        logger.info(f"[AstroTrainer] GPU available: {self.use_gpu}")

        self.model = None
        self.datamodule = None
        self.trainer = None
        self.mlflow_logger = None

    def _set_optimal_defaults(self):
        """Set optimal defaults based on 2025 best practices."""
        # DataLoader optimization defaults
        if self.use_gpu:
            # GPU-optimized settings
            if "num_workers" not in self.config:
                # Optimal workers: 4-8 for consumer GPUs, 8-16 for workstation
                self.config["num_workers"] = min(8, torch.get_num_threads())
            if "pin_memory" not in self.config:
                self.config["pin_memory"] = True
            if "persistent_workers" not in self.config:
                self.config["persistent_workers"] = True
            if "prefetch_factor" not in self.config:
                self.config["prefetch_factor"] = 4
        else:
            # CPU settings
            if "num_workers" not in self.config:
                self.config["num_workers"] = 0
            self.config["pin_memory"] = False
            self.config["persistent_workers"] = False
            self.config["prefetch_factor"] = None

        # Training defaults
        if "drop_last" not in self.config:
            self.config["drop_last"] = True  # Better batch consistency
        if "precision" not in self.config:
            self.config["precision"] = "16-mixed" if self.use_gpu else "32"

        logger.info(
            f"[AstroTrainer] Optimized settings: "
            f"num_workers={self.config['num_workers']}, "
            f"pin_memory={self.config.get('pin_memory', False)}, "
            f"persistent_workers={self.config.get('persistent_workers', False)}"
        )

    def create_model(self) -> pl.LightningModule:
        """Create Lightning model based on configuration."""
        # First, create datamodule to get data info
        if not hasattr(self, "datamodule"):
            self.create_datamodule()

        # Get actual feature dimensions from data
        num_features = self.datamodule.num_features
        num_classes = self.datamodule.num_classes

        logger.info(f"📊 Data info: {num_features} features, {num_classes} classes")

        # Build model kwargs from config
        model_kwargs = {}
        param_keys = [
            "learning_rate",
            "optimizer",
            "scheduler",
            "num_layers",
            "dropout",
            "weight_decay",
            "warmup_epochs",
            "min_lr",
            "loss_function",
            "conv_type",
            "temporal_model",
            "sequence_length",
            "pooling",
            "use_batch_norm",
        ]

        for key in param_keys:
            if key in self.config and self.config[key] is not None:
                model_kwargs[key] = self.config[key]

        # Create model using improved config logic
        if self.config.get("preset"):
            preset_name = self.config["preset"]
            logger.info(f"Creating model with preset: {preset_name}")

            # Validate preset exists
            available_presets = get_available_presets()
            if preset_name not in available_presets:
                raise ValueError(
                    f"Unknown preset '{preset_name}'. Available: {available_presets}"
                )

            # Pass actual data dimensions to preset, let factory handle the rest
            model = create_model_from_preset(
                preset_name,
                num_features=num_features,
                num_classes=num_classes,
                **model_kwargs,
            )

        elif self.config.get("model_type"):
            # Use model_type-based creation (preferred approach)
            model_type = self.config["model_type"]
            logger.info(f"Creating model of type: {model_type}")

            task = self.config.get("task", None)
            hidden_dim = self.config.get(
                "hidden_dim", max(32, num_features)
            )  # At least 32, or data features

            logger.info(
                f"Model params: num_features={num_features}, "
                f"hidden_dim={hidden_dim}, num_classes={num_classes}"
            )

            from astro_lab.models.core import create_model

            model = create_model(
                model_type=model_type,
                num_features=num_features,
                num_classes=num_classes,
                task=task,
                hidden_dim=hidden_dim,
                **model_kwargs,
            )

        elif self.config.get("task"):
            # Fallback to task-based model creation
            task = self.config["task"]
            logger.info(f"Creating model for task: {task}")

            hidden_dim = self.config.get(
                "hidden_dim", max(32, num_features)
            )  # At least 32, or data features

            model = create_model_for_task(
                task=task,
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                **model_kwargs,
            )
        else:
            raise ValueError(
                "Must specify either 'preset', 'model_type', or 'task' in config"
            )

        self.model = model
        return model

    def create_datamodule(self) -> AstroDataModule:
        """Create optimized data module for training."""
        # Use consistent naming: dataset
        dataset = self.config.get("dataset", self.config.get("survey", "gaia"))
        batch_size = self.config.get("batch_size", 32)
        max_samples = self.config.get("max_samples")

        # Get normalized model type
        model_type = self.config.get("model_type")
        if model_type:
            model_type = self.MODEL_TYPE_MAPPING.get(model_type, model_type)

        logger.info(f"Setting up data for dataset: {dataset}")
        logger.info(f"Batch size: {batch_size}, Model type: {model_type}")

        # Get optimized DataLoader settings from config
        dataloader_kwargs = {
            "num_workers": self.config.get("num_workers", 0),
            "pin_memory": self.config.get("pin_memory", False),
            "persistent_workers": self.config.get("persistent_workers", False),
            "prefetch_factor": self.config.get("prefetch_factor", 2),
        }

        logger.info(f"DataLoader settings: {dataloader_kwargs}")

        # Create DataModule with optimized settings
        datamodule = AstroDataModule(
            survey=dataset,
            batch_size=batch_size,
            max_samples=max_samples,
            model_type=model_type,
            train_ratio=self.config.get("train_ratio", 0.7),
            val_ratio=self.config.get("val_ratio", 0.15),
            **dataloader_kwargs,
        )

        self.datamodule = datamodule
        return datamodule

    def create_trainer(self) -> pl.Trainer:
        """Create Lightning trainer with optimized configuration."""
        logger.info("Configuring Lightning trainer with 2025 optimizations")

        # Use data_config for correct paths
        from astro_lab.data.config import data_config

        # Ensure experiment directories exist
        experiment_name = self.config.get("experiment_name", "default")
        data_config.ensure_experiment_directories(experiment_name)

        # Get correct paths from data_config
        checkpoint_dir = data_config.checkpoints_dir / experiment_name
        logs_dir = data_config.logs_dir / experiment_name

        logger.info(f"📁 Checkpoint directory: {checkpoint_dir}")
        logger.info(f"📁 Logs directory: {logs_dir}")

        # Setup callbacks
        callbacks = [
            # Early Stopping - weniger aggressiv
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.get(
                    "early_stopping_patience", 20
                ),  # Erhöht von 10 auf 20
                verbose=True,
                mode="min",
            ),
            # Model Checkpointing mit sehr einfachen Namen
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="checkpoint",  # Sehr einfacher Name ohne Versionierung
                monitor="val_loss",
                save_top_k=3,
                save_last=False,  # Deaktiviert um MLflow-Probleme zu vermeiden
                mode="min",
                verbose=True,
            ),
            # Learning Rate Monitor
            LearningRateMonitor(logging_interval="step"),
            # Rich Progress Bar
            RichProgressBar(
                refresh_rate=1,
                leave=True,
            ),
        ]

        # Setup MLflow logger
        model_name = (
            self.config.get("model_type")
            or self.config.get("preset")
            or self.config.get("task", "unknown")
        )
        dataset = self.config.get("dataset", self.config.get("survey", "gaia"))

        # Ensure MLruns directory exists
        mlruns_dir = data_config.mlruns_dir
        mlruns_dir.mkdir(parents=True, exist_ok=True)

        # Debug MLflow setup
        logger.info("🔍 MLflow setup:")
        logger.info(f"   MLruns directory: {mlruns_dir}")
        logger.info(f"   Directory exists: {mlruns_dir.exists()}")

        try:
            # Fix tracking URI format for Windows - use relative path
            tracking_uri = str(mlruns_dir.absolute()).replace("\\", "/")
            if tracking_uri.startswith("D:"):
                tracking_uri = tracking_uri[2:]  # Remove drive letter
            tracking_uri = f"file://{tracking_uri}"

            logger.info(f"   Tracking URI: {tracking_uri}")
            logger.info(f"   Experiment name: {experiment_name}")
            logger.info(f"   Run name: {model_name}_{dataset}")

            # Import Lightning MLflow Logger
            from lightning.pytorch.loggers import MLFlowLogger

            # Create custom MLflow logger that fixes artifact path issues
            class FixedMLFlowLogger(MLFlowLogger):
                def _scan_and_log_checkpoints(self, checkpoint_callback):
                    """Override to fix artifact path issues."""
                    if checkpoint_callback is None:
                        return

                    # Get checkpoint files
                    checkpoint_files = []
                    if (
                        hasattr(checkpoint_callback, "best_model_path")
                        and checkpoint_callback.best_model_path
                    ):
                        checkpoint_files.append(checkpoint_callback.best_model_path)
                    if (
                        hasattr(checkpoint_callback, "last_model_path")
                        and checkpoint_callback.last_model_path
                    ):
                        checkpoint_files.append(checkpoint_callback.last_model_path)

                    # Log with fixed artifact paths
                    for checkpoint_file in checkpoint_files:
                        if checkpoint_file and Path(checkpoint_file).exists():
                            # Create MLflow-compatible artifact path
                            filename = Path(checkpoint_file).name
                            # Replace problematic characters
                            safe_name = (
                                filename.replace("=", "_")
                                .replace("-", "_")
                                .replace(".", "_")
                            )
                            artifact_path = f"checkpoints/{safe_name}"

                            try:
                                self.experiment.log_artifact(
                                    self._run_id, checkpoint_file, artifact_path
                                )
                                logger.info(f"✅ Logged checkpoint: {artifact_path}")
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to log checkpoint {checkpoint_file}: {e}"
                                )

            mlflow_logger = FixedMLFlowLogger(
                experiment_name=experiment_name,
                run_name=f"{model_name}_{dataset}",
                tracking_uri=tracking_uri,
                tags={
                    "model": model_name,
                    "dataset": dataset,
                    "preset": self.config.get("preset", "custom"),
                    "task": self.config.get("task", "unknown"),
                    "batch_size": str(self.config.get("batch_size", 32)),
                    "learning_rate": str(self.config.get("learning_rate", 0.001)),
                    "num_workers": str(self.config.get("num_workers", 0)),
                    "pin_memory": str(self.config.get("pin_memory", False)),
                },
                log_model=True,  # Re-enable model logging with fixed checkpoint names
            )
            logger.info("✅ MLflow logger created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create MLflow logger: {e}")
            logger.warning("   Using default logger instead")
            mlflow_logger = None

        self.mlflow_logger = mlflow_logger

        # Trainer configuration with GPU optimizations
        max_epochs = self.config.get("epochs", self.config.get("max_epochs", 50))
        trainer_kwargs = {
            "max_epochs": max_epochs,
            "accelerator": "gpu" if self.use_gpu else "cpu",
            "devices": 1,
            "precision": self.config.get("precision", "16-mixed"),
            "callbacks": callbacks,
            "logger": mlflow_logger,
            "gradient_clip_val": self.config.get("gradient_clip_val", 1.0),
            "accumulate_grad_batches": self.config.get("accumulate_grad_batches", 1),
            "log_every_n_steps": 1,
            "val_check_interval": self.config.get("val_check_interval", 1.0),
            "check_val_every_n_epoch": self.config.get("check_val_every_n_epoch", 1),
            "num_sanity_val_steps": 2,
            "enable_model_summary": True,
            "enable_checkpointing": True,
            "deterministic": self.config.get("deterministic", True),
        }

        # Add GPU-specific optimizations
        if self.use_gpu:
            trainer_kwargs.update(
                {
                    "sync_batchnorm": True,  # For better batch norm with small batches
                    "detect_anomaly": False,  # Disable for performance (enable for debugging)
                }
            )

        self.trainer = pl.Trainer(**trainer_kwargs)
        return self.trainer

    def setup_gpu_optimizations(self):
        """Setup GPU optimizations based on 2025 best practices."""
        if torch.cuda.is_available():
            # Enable cuDNN autotuner for variable input sizes
            torch.backends.cudnn.benchmark = True
            # Deterministic mode for reproducibility (slight performance cost)
            torch.backends.cudnn.deterministic = self.config.get("deterministic", True)

            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(
                self.config.get("gpu_memory_fraction", 0.95)
            )

            # Clear cache before training
            torch.cuda.empty_cache()

            device_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 Using GPU: {device_name}")
            logger.info(
                f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
            )
            logger.info(
                f"   Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
            )

    def validate_model_data_compatibility(self) -> bool:
        """Validate model and data compatibility with detailed checks."""
        if self.model is None or self.datamodule is None:
            logger.error("Model or datamodule not initialized")
            return False

        # Check feature dimensions
        model_features = getattr(self.model, "num_features", None)
        data_features = getattr(self.datamodule, "num_features", None)

        if model_features and data_features:
            if model_features != data_features:
                logger.error(
                    f"Feature dimension mismatch: model expects {model_features}, "
                    f"data has {data_features}"
                )
                return False
            else:
                logger.info(f"✅ Feature dimensions match: {model_features}")

        # Check number of classes
        model_classes = getattr(self.model, "num_classes", None)
        data_classes = getattr(self.datamodule, "num_classes", None)

        if model_classes and data_classes:
            if model_classes != data_classes:
                logger.warning(
                    f"Class number mismatch: model expects {model_classes}, "
                    f"data has {data_classes}. This might be OK for some tasks."
                )

        # Check model type compatibility
        model_type = self.config.get("model_type")
        if model_type and hasattr(self.datamodule, "model_type"):
            normalized_type = self.MODEL_TYPE_MAPPING.get(model_type, model_type)
            if self.datamodule.model_type != normalized_type:
                logger.warning(
                    f"Model type mismatch: trainer has {normalized_type}, "
                    f"datamodule has {self.datamodule.model_type}"
                )

        logger.info("✅ Model-data compatibility check passed")
        return True

    def train(self) -> bool:
        """Execute the complete training pipeline with 2025 optimizations."""
        try:
            logger.info("🚀 Starting AstroLab training pipeline (2025 optimized)")
            logger.info(f"   Config: {self.config}")

            # Setup GPU optimizations
            self.setup_gpu_optimizations()

            # Create datamodule and setup
            datamodule = self.create_datamodule()
            datamodule.setup()

            # Update config with actual features from data
            self.config["num_features"] = datamodule.num_features
            self.config["num_classes"] = datamodule.num_classes
            if "hidden_dim" not in self.config:
                self.config["hidden_dim"] = datamodule.num_features

            logger.info(
                f"📊 Data loaded: {datamodule.num_features} features, "
                f"{datamodule.num_classes} classes"
            )

            # Create model with updated config
            model = self.create_model()

            # Validate compatibility
            if not self.validate_model_data_compatibility():
                return False

            # Create trainer
            trainer = self.create_trainer()

            # Debug MLflow status
            if self.mlflow_logger:
                logger.info("✅ MLflow logger is active")
            else:
                logger.warning("⚠️ No MLflow logger available")

            # Log training start
            logger.info("🏃 Starting training...")
            logger.info(f"   Epochs: {trainer.max_epochs}")
            logger.info(f"   Batch size: {self.config.get('batch_size', 32)}")
            logger.info(f"   Learning rate: {self.config.get('learning_rate', 0.001)}")
            logger.info(f"   Precision: {trainer.precision}")

            # Start training
            trainer.fit(model, datamodule)

            return True

        except KeyboardInterrupt:
            logger.warning("⚠️ Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # Cleanup GPU memory
            if self.use_gpu and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("🧹 Cleaned up GPU memory")
                except Exception as e:
                    logger.warning(f"⚠️ GPU cleanup failed: {e}")

    def get_model(self) -> Optional[pl.LightningModule]:
        """Get the trained model."""
        return self.model

    def get_trainer(self) -> Optional[pl.Trainer]:
        """Get the Lightning trainer."""
        return self.trainer


def train_model(config: Dict[str, Any]) -> bool:
    """
    Convenience function to train a model with given configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        True if training completed successfully, False otherwise
    """
    trainer = AstroTrainer(config)
    return trainer.train()
