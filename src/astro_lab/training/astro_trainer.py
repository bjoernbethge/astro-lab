"""
AstroLab Trainer for Lightning Models
====================================

Unified trainer class for training AstroLab Lightning models.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping

from astro_lab.data import AstroDataModule
from astro_lab.models.lightning import (
    create_lightning_model,
    create_preset_model,
)

from .callbacks import SafeModelCheckpoint
from .mlflow_logger import LightningMLflowLogger

logger = logging.getLogger(__name__)


class AstroTrainer:
    """
    Unified trainer for AstroLab Lightning models.

    Handles model creation, data loading, training setup, and execution.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AstroTrainer with configuration.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.model = None
        self.datamodule = None
        self.trainer = None
        self.mlflow_logger = None

    def create_model(self) -> pl.LightningModule:
        """Create Lightning model based on configuration."""
        # Build model kwargs from config
        model_kwargs = {}
        for key in [
            "learning_rate",
            "optimizer",
            "scheduler",
            "hidden_dim",
            "num_layers",
            "num_classes",
        ]:
            if key in self.config and self.config[key] is not None:
                model_kwargs[key] = self.config[key]

        # Create model
        if self.config.get("preset"):
            logger.info(f"Creating model with preset: {self.config['preset']}")
            model = create_preset_model(self.config["preset"], **model_kwargs)
        elif self.config.get("model"):
            logger.info(f"Creating model: {self.config['model']}")
            model = create_lightning_model(self.config["model"], **model_kwargs)
        else:
            raise ValueError("Must specify either 'model' or 'preset' in config")

        self.model = model
        return model

    def create_datamodule(self) -> AstroDataModule:
        """Create data module for training."""
        survey = self.config.get("survey", "gaia")
        batch_size = self.config.get("batch_size", 32)
        max_samples = self.config.get("max_samples")

        logger.info(f"Setting up data for survey: {survey}")
        datamodule = AstroDataModule(
            survey=survey,
            batch_size=batch_size,
            max_samples=max_samples,
            pin_memory=False,  # PyG handles memory management
            persistent_workers=True,  # Fix: Enable for better performance
            num_workers=4,  # Fix: Use more workers for better performance
        )

        self.datamodule = datamodule
        return datamodule

    def create_trainer(self) -> pl.Trainer:
        """Create Lightning trainer with configuration."""
        logger.info("Configuring Lightning trainer")

        # Setup callbacks
        checkpoint_dir = self.config.get("checkpoint_dir", Path("checkpoints"))
        callbacks = [
            SafeModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="model_epoch{epoch:02d}_valloss{val_loss:.3f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                auto_insert_metric_name=False,  # Prevent metric name duplication
                save_weights_only=False,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
            ),
        ]

        # Setup MLflow logger
        experiment_name = self.config.get("experiment_name", "astrolab_experiment")
        model_name = self.config.get("model") or self.config.get("preset", "unknown")
        survey = self.config.get("survey", "gaia")

        mlflow_logger = LightningMLflowLogger(
            experiment_name=experiment_name,
            run_name=f"{model_name}_{survey}",
            tags={
                "model": model_name,
                "survey": survey,
                "preset": self.config.get("preset") or "custom",
            },
        )

        self.mlflow_logger = mlflow_logger

        # Optimized RTX 4070 configuration
        trainer_kwargs = {
            "max_epochs": self.config.get("max_epochs", self.config.get("epochs", 50)),
            "accelerator": "gpu" if self.config.get("devices", 1) > 0 else "cpu",
            "devices": self.config.get("devices", 1),
            "precision": "16-mixed",  # RTX 4070 has excellent Tensor Core support
            "strategy": "auto",
            "callbacks": callbacks,
            "logger": mlflow_logger,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,  # RTX 4070 has enough VRAM for full batches
            "enable_progress_bar": True,
            "log_every_n_steps": 1,  # Fix: Log every step for small datasets
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 2,  # Quick sanity check
        }

        # Development options
        if self.config.get("overfit_batches"):
            trainer_kwargs["overfit_batches"] = self.config["overfit_batches"]

        trainer = pl.Trainer(**trainer_kwargs)
        self.trainer = trainer
        return trainer

    def train(self) -> bool:
        """
        Execute complete training pipeline.

        Returns:
            True if training succeeded, False otherwise
        """
        try:
            logger.info("🚀 Starting AstroLab Lightning training")

            # RTX 4070 optimizations

            if torch.cuda.is_available():
                # Enable Tensor Core optimizations for RTX 4070
                torch.set_float32_matmul_precision("medium")
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = (
                    True  # Optimize for consistent input sizes
                )
                logger.info("✅ RTX 4070 optimizations enabled")

            # Create components
            self.create_model()
            self.create_datamodule()
            self.create_trainer()

            # Training
            logger.info("Starting training...")
            if self.model is None or self.datamodule is None:
                raise RuntimeError(
                    "Model and datamodule must be created before training"
                )
            self.trainer.fit(self.model, self.datamodule)

            # Testing
            logger.info("Running final evaluation...")
            self.trainer.test(self.model, self.datamodule)

            logger.info("✅ Training completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def get_model(self) -> Optional[pl.LightningModule]:
        """Get the created model."""
        return self.model

    def get_trainer(self) -> Optional[pl.Trainer]:
        """Get the created trainer."""
        return self.trainer


def train_model(config: Dict[str, Any]) -> bool:
    """
    Convenience function to train a model with configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        True if training succeeded, False otherwise
    """
    trainer = AstroTrainer(config)
    return trainer.train()
