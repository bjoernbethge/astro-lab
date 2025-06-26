"""
AstroLab Trainer for Lightning Models
====================================

Unified trainer class for training AstroLab Lightning models.
Optimized for PyTorch Lightning 2.x and RTX 4070 Mobile GPU.
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
from astro_lab.models.lightning import (
    create_lightning_model,
    create_preset_model,
)

from .mlflow_logger import LightningMLflowLogger

logger = logging.getLogger(__name__)


class AstroTrainer:
    """
    Unified trainer for AstroLab Lightning models.
    
    Handles model creation, data loading, training setup, and execution.
    Optimized for modern GPUs with mixed precision and efficient training.
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
        # Build model kwargs from config - mehr Parameter unterstützen
        model_kwargs = {}
        for key in [
            "learning_rate", "optimizer", "scheduler", "hidden_dim", 
            "num_layers", "num_classes", "dropout", "weight_decay",
            "warmup_epochs", "min_lr", "task", "loss_function"
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
        
        # Setup callbacks
        callbacks = [
            # Verbessertes Checkpoint-Management
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                auto_insert_metric_name=False,
                every_n_epochs=1,
            ),
            # Early Stopping mit Patience
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.get("early_stopping_patience", 10),
                mode="min",
                verbose=True,
            ),
            # Learning Rate Monitor
            LearningRateMonitor(logging_interval="step"),
            # Rich Progress Bar für bessere Visualisierung
            RichProgressBar(refresh_rate=10),
        ]
        
        # Setup MLflow logger
        experiment_name = self.config.get("experiment_name", "astrolab_experiment")
        model_name = self.config.get("model") or self.config.get("preset", "unknown")
        survey = self.config.get("dataset", self.config.get("survey", "gaia"))
        
        mlflow_logger = LightningMLflowLogger(
            experiment_name=experiment_name,
            run_name=f"{model_name}_{survey}",
            tags={
                "model": model_name,
                "survey": survey,
                "preset": self.config.get("preset") or "custom",
                "batch_size": str(self.config.get("batch_size", 32)),
                "learning_rate": str(self.config.get("learning_rate", 0.001)),
            },
            log_model=True,  # Modell automatisch in MLflow speichern
        )
        
        self.mlflow_logger = mlflow_logger
        
        # Optimierte RTX 4070 Mobile Konfiguration
        trainer_kwargs = {
            "max_epochs": self.config.get("max_epochs", self.config.get("epochs", 50)),
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,  # RTX 4070 Mobile ist single GPU
            "precision": self._get_precision_config(),
            "callbacks": callbacks,
            "logger": mlflow_logger,
            "gradient_clip_val": self.config.get("gradient_clip_val", 1.0),
            "accumulate_grad_batches": self.config.get("accumulate_grad_batches", 1),
            "log_every_n_steps": max(1, self.config.get("log_every_n_steps", 10)),
            "val_check_interval": self.config.get("val_check_interval", 1.0),
            "check_val_every_n_epoch": self.config.get("check_val_every_n_epoch", 1),
            "num_sanity_val_steps": 2,
            "enable_model_summary": True,
            "enable_checkpointing": True,
            "deterministic": self.config.get("deterministic", True),  # Reproduzierbarkeit
            "benchmark": True,  # Optimierung für konsistente Input-Größen
        }
        
        # Strategy für Multi-GPU falls gewünscht
        if self.config.get("strategy"):
            trainer_kwargs["strategy"] = self.config["strategy"]
        
        # Development/Debug Optionen
        if self.config.get("fast_dev_run"):
            trainer_kwargs["fast_dev_run"] = self.config["fast_dev_run"]
        if self.config.get("overfit_batches"):
            trainer_kwargs["overfit_batches"] = self.config["overfit_batches"]
        if self.config.get("limit_train_batches"):
            trainer_kwargs["limit_train_batches"] = self.config["limit_train_batches"]
        if self.config.get("limit_val_batches"):
            trainer_kwargs["limit_val_batches"] = self.config["limit_val_batches"]
            
        # Resume from checkpoint
        if self.config.get("resume"):
            trainer_kwargs["resume_from_checkpoint"] = self.config["resume"]
            
        trainer = pl.Trainer(**trainer_kwargs)
        self.trainer = trainer
        return trainer
        
    def _get_precision_config(self) -> str:
        """
        Bestimme die optimale Precision-Konfiguration für RTX 4070.
        
        RTX 4070 unterstützt:
        - FP16 mit Tensor Cores (schnell, weniger Speicher)
        - BF16 mit Tensor Cores (stabiler als FP16)
        - TF32 für Matmul-Operationen
        """
        precision = self.config.get("precision", "16-mixed")
        
        # Mapping von alten zu neuen Precision-Bezeichnungen
        precision_map = {
            16: "16-mixed",
            "16": "16-mixed",
            32: "32-true",
            "32": "32-true",
            "bf16": "bf16-mixed",
            "fp16": "16-mixed",
        }
        
        return precision_map.get(precision, precision)
        
    def setup_gpu_optimizations(self):
        """RTX 4070 spezifische Optimierungen."""
        if torch.cuda.is_available():
            # Tensor Core Optimierungen
            torch.set_float32_matmul_precision("medium")  # TF32 für bessere Performance
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Optimiert für konsistente Inputs
            torch.backends.cudnn.deterministic = self.config.get("deterministic", False)
            
            # Memory Management
            torch.cuda.empty_cache()
            
            # Device Info logging
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            device_capability = torch.cuda.get_device_capability(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            
            logger.info(f"🎮 GPU Device: {device_name}")
            logger.info(f"   Compute Capability: {device_capability}")
            logger.info(f"   Total Memory: {total_memory:.2f} GB")
            logger.info("✅ RTX 4070 optimizations enabled:")
            logger.info("   - TF32 for matmul operations")
            logger.info("   - Mixed precision training ready")
            logger.info("   - cuDNN benchmark mode enabled")
            
    def train(self) -> bool:
        """
        Execute complete training pipeline.
        
        Returns:
            True if training succeeded, False otherwise
        """
        try:
            logger.info("🚀 Starting AstroLab Lightning training")
            
            # GPU Optimizations
            self.setup_gpu_optimizations()
            
            # Create components
            self.create_model()
            self.create_datamodule()
            self.create_trainer()
            
            # Setup data
            self.datamodule.setup("fit")
            
            # Log configuration
            logger.info(f"📊 Training configuration:")
            logger.info(f"   Model: {self.config.get('model') or self.config.get('preset')}")
            logger.info(f"   Dataset: {self.config.get('dataset', 'gaia')}")
            logger.info(f"   Batch size: {self.config.get('batch_size', 32)}")
            logger.info(f"   Learning rate: {self.config.get('learning_rate', 0.001)}")
            logger.info(f"   Max epochs: {self.config.get('max_epochs', 50)}")
            logger.info(f"   Precision: {self._get_precision_config()}")
            
            # Training
            logger.info("Starting training...")
            self.trainer.fit(
                self.model, 
                self.datamodule,
                ckpt_path=self.config.get("resume")  # Resume if specified
            )
            
            # Testing nur wenn Test-Daten vorhanden
            if hasattr(self.datamodule, 'test_dataloader') and self.datamodule.test_dataloader() is not None:
                logger.info("Running final evaluation...")
                self.trainer.test(self.model, self.datamodule)
            
            # Cleanup
            torch.cuda.empty_cache()
            
            logger.info("✅ Training completed successfully!")
            return True
            
        except KeyboardInterrupt:
            logger.warning("⚠️ Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False
        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
