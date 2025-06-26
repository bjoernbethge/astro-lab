"""
AstroLab Trainer for Lightning Models
====================================

Unified trainer class for training AstroLab Lightning models.
Optimized for PyTorch Lightning 2.x and RTX 4070 Mobile GPU.

Now supports the consolidated 4-model architecture with Lightning Mixins:
- AstroNodeGNN: Node-level tasks (classification, regression, segmentation)
- AstroGraphGNN: Graph-level tasks (survey classification, cluster analysis)
- AstroTemporalGNN: Temporal tasks (lightcurves, time series)
- AstroPointNet: Point cloud tasks (classification, segmentation, registration)
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

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
    Unified trainer for AstroLab Lightning models.

    Handles model creation, data loading, training setup, and execution.
    Optimized for modern GPUs with mixed precision and efficient training.

    Now supports the consolidated 4-model architecture with Lightning Mixins.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AstroTrainer with configuration.

        Args:
            config: Training configuration dictionary
        """
        self.config = config.copy()

        logger = logging.getLogger(__name__)
        logger.info(f"[AstroTrainer] Init: config keys: {list(self.config.keys())}")
        if "num_features" in self.config:
            logger.info(
                f"[AstroTrainer] Init: num_features={self.config['num_features']}"
            )
        else:
            logger.info("[AstroTrainer] Init: num_features not in config")

        self.model = None
        self.datamodule = None
        self.trainer = None
        self.mlflow_logger = None

    def create_model(self) -> pl.LightningModule:
        """Create Lightning model based on configuration."""
        # Build model kwargs from config - support all parameters
        model_kwargs = {}
        for key in [
            "learning_rate",
            "optimizer",
            "scheduler",
            "num_layers",
            "num_classes",
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
        ]:
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

            model = create_model_from_preset(preset_name, **model_kwargs)
        elif self.config.get("model_type"):
            # Use model_type-based creation (preferred approach)
            model_type = self.config["model_type"]
            logger.info(f"Creating model of type: {model_type}")

            num_features = self.config.get("num_features", 64)
            num_classes = self.config.get("num_classes", 2)
            task = self.config.get("task", None)  # Will be auto-assigned by factory

            # Ensure hidden_dim is passed correctly
            hidden_dim = self.config.get(
                "hidden_dim", num_features
            )  # Default to num_features
            logger.info(
                f"Creating model with num_features={num_features}, hidden_dim={hidden_dim}"
            )

            from astro_lab.models.core import create_model

            model = create_model(
                model_type=model_type,
                num_features=num_features,
                num_classes=num_classes,
                task=task,  # Can be None, factory will auto-assign
                hidden_dim=hidden_dim,  # Explicitly pass hidden_dim
                **model_kwargs,
            )
        elif self.config.get("task"):
            # Fallback to task-based model creation
            task = self.config["task"]
            logger.info(f"Creating model for task: {task}")

            num_features = self.config.get("num_features", 64)
            num_classes = self.config.get("num_classes", 2)
            hidden_dim = self.config.get(
                "hidden_dim", num_features
            )  # Default to num_features

            model = create_model_for_task(
                task=task,
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,  # Explicitly pass hidden_dim
                **model_kwargs,
            )
        else:
            raise ValueError(
                "Must specify either 'preset', 'model_type', or 'task' in config"
            )

        self.model = model
        return model

    def create_datamodule(self) -> AstroDataModule:
        """Create data module for training."""
        # Fix: dataset statt survey für Konsistenz
        survey = self.config.get("dataset", self.config.get("survey", "gaia"))
        batch_size = self.config.get("batch_size", 32)
        max_samples = self.config.get("max_samples")

        logger.info(f"Setting up data for survey: {survey}")
        logger.info(f"Batch size: {batch_size}")

        # Optimierte Dataloader-Einstellungen für RTX 4070
        # Pin memory nur wenn GPU verfügbar und num_workers > 0
        use_gpu = torch.cuda.is_available()
        num_workers = self.config.get("num_workers", 4 if use_gpu else 0)

        datamodule = AstroDataModule(
            survey=survey,
            batch_size=batch_size,
            max_samples=max_samples,
            pin_memory=use_gpu and num_workers > 0,  # Pin memory nur mit workers
            persistent_workers=num_workers > 0,  # Nur mit workers sinnvoll
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,  # Nur mit workers
        )

        self.datamodule = datamodule
        return datamodule

    def create_trainer(self) -> pl.Trainer:
        """Create Lightning trainer with configuration."""
        logger.info("Configuring Lightning trainer")

        # Checkpoint directory
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup callbacks - vereinfacht um Fehler zu vermeiden
        callbacks = [
            # Early Stopping
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.get("early_stopping_patience", 10),
                verbose=True,
                mode="min",
            ),
            # Model Checkpointing
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=3,
                save_last=True,
                mode="min",
                verbose=True,
            ),
            # Learning Rate Monitor
            LearningRateMonitor(logging_interval="step"),
            # Rich Progress Bar mit optimierten Einstellungen
            RichProgressBar(
                refresh_rate=1,  # Häufigere Updates für bessere Sichtbarkeit
                leave=True,  # Progress Bar nach Training sichtbar lassen
            ),
        ]

        # Setup MLflow logger
        experiment_name = self.config.get("experiment_name", "astrolab_experiment")
        model_name = (
            self.config.get("model")
            or self.config.get("preset")
            or self.config.get("task_type", "unknown")
        )
        survey = self.config.get("dataset", self.config.get("survey", "gaia"))

        mlflow_logger = LightningMLflowLogger(
            experiment_name=experiment_name,
            run_name=f"{model_name}_{survey}",
            tags={
                "model": model_name,
                "survey": survey,
                "preset": self.config.get("preset") or "custom",
                "task_type": self.config.get("task_type") or "unknown",
                "batch_size": str(self.config.get("batch_size", 32)),
                "learning_rate": str(self.config.get("learning_rate", 0.001)),
            },
            log_model=True,  # Modell automatisch in MLflow speichern
        )

        self.mlflow_logger = mlflow_logger

        # Optimierte RTX 4070 Mobile Konfiguration
        # For performance: prefer benchmark=True, deterministic=False
        # For reproducibility: prefer deterministic=True, benchmark=False
        use_deterministic = self.config.get(
            "deterministic", False
        )  # Default to performance

        trainer_kwargs = {
            "max_epochs": self.config.get("max_epochs", self.config.get("epochs", 50)),
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,  # RTX 4070 Mobile ist single GPU
            "precision": self._get_precision_config(),
            "callbacks": callbacks,
            "logger": mlflow_logger,
            "gradient_clip_val": self.config.get("gradient_clip_val", 1.0),
            "accumulate_grad_batches": self.config.get("accumulate_grad_batches", 1),
            "log_every_n_steps": 1,  # Set to 1 for single-batch training to avoid warnings
            "val_check_interval": self.config.get("val_check_interval", 1.0),
            "check_val_every_n_epoch": self.config.get("check_val_every_n_epoch", 1),
            "num_sanity_val_steps": 2,
            "enable_model_summary": True,
            "enable_checkpointing": True,
            "deterministic": use_deterministic,  # Reproduzierbarkeit vs Performance
            "benchmark": not use_deterministic,  # Optimierung für Performance (wenn nicht deterministic)
        }

        # Strategy für Multi-GPU falls gewünscht
        if self.config.get("use_ddp", False) and torch.cuda.device_count() > 1:
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
            trainer_kwargs["devices"] = torch.cuda.device_count()

        self.trainer = pl.Trainer(**trainer_kwargs)
        return self.trainer

    def _get_precision_config(self) -> str:
        """Get precision configuration based on GPU capabilities."""
        precision = self.config.get("precision", "auto")

        if precision == "auto":
            if torch.cuda.is_available():
                # Check if GPU supports mixed precision
                gpu_name = torch.cuda.get_device_name(0).lower()
                if "rtx" in gpu_name or "gtx" in gpu_name:
                    return "16-mixed"  # Mixed precision for modern GPUs
                else:
                    return "32"  # Full precision for older GPUs
            else:
                return "32"  # Full precision for CPU

        return precision

    def setup_gpu_optimizations(self):
        """Setup GPU optimizations for better performance."""
        if torch.cuda.is_available():
            # Enable cudnn benchmarking for better performance
            torch.backends.cudnn.benchmark = True

            # Set memory fraction if specified
            memory_fraction = self.config.get("gpu_memory_fraction")
            if memory_fraction:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)

            # Log GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            # Optimize for RTX 4070 Mobile
            if "rtx 4070" in gpu_name.lower():
                logger.info("Optimizing for RTX 4070 Mobile")
                # Set optimal settings for RTX 4070 Mobile
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True

    def train(self) -> bool:
        """
        Execute the complete training pipeline.

        Returns:
            True if training completed successfully, False otherwise
        """
        try:
            logger.info("Starting AstroLab training pipeline")

            # Setup GPU optimizations
            self.setup_gpu_optimizations()

            # Create datamodule FIRST
            datamodule = self.create_datamodule()
            logger.info("Created datamodule")

            # FORCE DataModule setup to get num_features
            datamodule.setup()
            logger.info("DataModule setup complete")

            # Get num_features from DataModule
            actual_features = datamodule.num_features
            logger.info(f"DataModule num_features: {actual_features}")

            # ALWAYS use DataModule num_features
            self.config["num_features"] = actual_features
            logger.info(f"Set config num_features to: {actual_features}")

            # Set hidden_dim = num_features, falls nicht explizit gesetzt
            if (
                "hidden_dim" not in self.config or self.config["hidden_dim"] is None
            ) and "num_features" in self.config:
                self.config["hidden_dim"] = self.config["num_features"]
                logger.info(
                    f"Set hidden_dim to num_features: {self.config['hidden_dim']}"
                )

            logger.info(
                f"[AstroTrainer] Final config before model creation: num_features={self.config.get('num_features')}, hidden_dim={self.config.get('hidden_dim')}"
            )

            # Create model AFTER config is updated
            model = self.create_model()
            logger.info(f"Created model: {model.__class__.__name__}")

            # Create trainer
            trainer = self.create_trainer()
            logger.info("Created trainer")

            # Start training
            logger.info("Starting training...")
            trainer.fit(model, datamodule)
            logger.info("Training completed successfully")

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Traceback: {e.__traceback__}")
            return False

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
