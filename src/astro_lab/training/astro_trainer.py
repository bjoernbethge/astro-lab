"""
AstroLab Trainer - High-Level Training API
==========================================

Main trainer class that provides a unified interface for training
astronomical ML models with integrated optimizations and monitoring.
"""

import logging
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.optim as optim

from astro_lab.config.defaults import TRAINING_DEFAULTS, get_training_config
from astro_lab.data.datamodules.lightning import AstroLightningDataset
from astro_lab.memory import clear_cuda_cache
from astro_lab.models.core.factory import create_model
from astro_lab.training.trainer_core import TrainingCore
from astro_lab.training.trainer_utils import log_training_info, validate_model_inputs

logger = logging.getLogger(__name__)


class AstroTrainer:
    """
    High-level trainer for AstroLab models.

    Provides a unified interface for training astronomical ML models with
    integrated hardware optimizations and comprehensive monitoring.
    """

    def __init__(self, config: Union[Dict[str, Any], str]):
        """
        Initialize AstroTrainer.

        Args:
            config: Configuration dictionary or path to config file
        """
        # Load configuration
        if isinstance(config, str):
            self.config = get_training_config()
        else:
            base_config = TRAINING_DEFAULTS.copy()
            base_config.update(config)
            self.config = base_config

        # Initialize components
        self.model = None
        self.datamodule = None
        self.trainer_core = None

        # Training state
        self.is_setup = False
        self.is_training = False

        logger.info("AstroTrainer initialized")

    def setup(self):
        """Setup training components."""
        if self.is_setup:
            logger.warning("Trainer already setup")
            return

        logger.info("Setting up training components...")

        # Setup datamodule
        self._setup_datamodule()

        # Setup model
        self._setup_model()

        # Setup optimizations
        self._setup_optimizations()

        # Validate setup
        self._validate_setup()

        self.is_setup = True
        logger.info("Training setup completed")

    def _setup_datamodule(self):
        """Setup data module with large-scale support."""
        # Extract large-scale parameters
        large_scale_params = {
            "sampling_strategy": self.config.get("sampling_strategy", "none"),
            "neighbor_sizes": self.config.get("neighbor_sizes", [25, 10]),
            "num_clusters": self.config.get("num_clusters", 1500),
            "saint_sample_coverage": self.config.get("saint_sample_coverage", 50),
            "saint_walk_length": self.config.get("saint_walk_length", 2),
            "enable_dynamic_batching": self.config.get("enable_dynamic_batching", False),
            "min_batch_size": self.config.get("min_batch_size", 1),
            "max_batch_size": self.config.get("max_batch_size", 512),
            "partition_method": self.config.get("partition_method"),
            "num_partitions": self.config.get("num_partitions", 4),
        }
        
        self.datamodule = AstroLightningDataset(
            survey=self.config["survey"],
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 4),
            max_samples=self.config.get("max_samples"),
            train_ratio=self.config.get("train_ratio", 0.8),
            val_ratio=self.config.get("val_ratio", 0.1),
            **large_scale_params,
        )

        self.datamodule.setup(stage="fit")  # Add stage parameter
        
        # Log data module info
        train_loader = self.datamodule.train_dataloader()
        logger.info(f"DataModule setup:")
        logger.info(f"  Survey: {self.config['survey']}")
        logger.info(f"  Sampling strategy: {large_scale_params['sampling_strategy']}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Batch size: {self.datamodule.current_batch_size}")

    def _setup_model(self):
        """Setup model with compilation support."""
        # Extract model compilation parameters
        compile_params = {
            "compile_model": self.config.get("compile_model", False),
            "compile_mode": self.config.get("compile_mode", "default"),
            "compile_dynamic": self.config.get("compile_dynamic", True),
        }
        
        self.model = create_model(
            model_type=self.config["model_type"],
            num_features=self.datamodule.num_features,
            num_classes=self.datamodule.num_classes,
            hidden_dim=self.config.get("hidden_dim", 64),
            num_layers=self.config.get("num_layers", 3),
            dropout=self.config.get("dropout", 0.1),
            **compile_params,
        )
        
        logger.info(f"Model created: {type(self.model).__name__}")
        if compile_params["compile_model"]:
            logger.info(f"  Compilation enabled: mode={compile_params['compile_mode']}")

    def _setup_optimizations(self):
        """Setup training optimizations."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Get dataloaders
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        test_loader = self.datamodule.test_dataloader()

        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

        # Create scheduler
        scheduler = None
        if self.config.get("use_scheduler", False):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )

        # Create training core
        self.trainer_core = TrainingCore(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,  # type: ignore
            config=self.config,
        )

        logger.info("Training optimizations setup completed")

    def _validate_setup(self):
        """Validate training setup."""
        if self.model is None or self.trainer_core is None:
            raise RuntimeError("Model or trainer core not initialized")

        # Validate model inputs
        train_loader = self.datamodule.train_dataloader()
        if not validate_model_inputs(
            self.model, train_loader, self.trainer_core.device
        ):
            raise ValueError("Model input validation failed")

        # Log training info
        log_training_info(
            self.model, train_loader, self.trainer_core.device, self.config
        )

    def train(self) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            Dictionary with training results
        """
        if not self.is_setup:
            self.setup()

        if self.is_training:
            logger.warning("Training already in progress")
            return {}

        self.is_training = True

        try:
            # Run training
            logger.info("Starting training...")
            results = self._run_training()

            # Run testing
            if self.datamodule.test_dataloader():
                logger.info("Running test evaluation...")
                test_results = self.trainer_core.test()
                results.update(test_results)

            return results

        finally:
            self.is_training = False
            clear_cuda_cache()

    def _run_training(self) -> Dict[str, Any]:
        """Run the main training loop."""
        max_epochs = self.config.get("max_epochs", 100)
        patience = self.config.get("patience", 10)

        best_val_loss = float("inf")
        patience_counter = 0
        
        # Check if this is part of an Optuna trial
        optuna_trial = self.config.get("optuna_trial")

        for epoch in range(max_epochs):
            self.trainer_core.current_epoch = epoch

            # Training
            train_loss, train_acc = self.trainer_core.train_epoch()

            # Validation
            val_loss, val_acc = self.trainer_core.validate_epoch()

            # Update scheduler
            self.trainer_core.update_scheduler(val_loss)
            
            # Report to Optuna if in HPO
            if optuna_trial is not None:
                optuna_trial.report(val_acc, epoch)
                
                # Check if should prune
                if optuna_trial.should_prune():
                    logger.info(f"Trial pruned at epoch {epoch}")
                    import optuna
                    raise optuna.TrialPruned()

            # Update best metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if self.config.get("save_best_model", True):
                    self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Update training history
            self.trainer_core.training_history["train_loss"].append(train_loss)
            self.trainer_core.training_history["val_loss"].append(val_loss)
            self.trainer_core.training_history["train_acc"].append(train_acc)
            self.trainer_core.training_history["val_acc"].append(val_acc)

        # Final results
        results = {
            "best_val_loss": best_val_loss,
            "best_val_acc": max(self.trainer_core.training_history["val_acc"]) if self.trainer_core.training_history["val_acc"] else 0,
            "final_train_loss": train_loss,
            "final_train_acc": train_acc,
            "final_val_loss": val_loss,
            "final_val_acc": val_acc,
            "epochs_trained": epoch + 1,
            "training_history": self.trainer_core.training_history,
        }

        logger.info(f"Training completed. Best val_loss: {best_val_loss:.4f}")

        return results

    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        if self.trainer_core is not None:
            self.trainer_core.save_checkpoint(filepath)

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        if self.trainer_core is not None:
            self.trainer_core.load_checkpoint(filepath)

    def predict(self, data: Any) -> torch.Tensor:
        """
        Make predictions on new data.

        Args:
            data: Input data

        Returns:
            Model predictions
        """
        if not self.is_setup:
            raise RuntimeError("Trainer not setup. Call setup() first.")

        if self.model is None or self.trainer_core is None:
            raise RuntimeError("Model or trainer core not initialized")

        self.model.eval()

        with torch.no_grad():
            if hasattr(data, "to"):
                data = data.to(self.trainer_core.device)

            if self.trainer_core.use_mixed_precision:
                with torch.autocast(device_type="cuda"):
                    predictions = self.model(data)
            else:
                predictions = self.model(data)

        return predictions

    def get_model(self) -> nn.Module:
        """Get the trained model."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model

    def get_config(self) -> Dict[str, Any]:
        """Get the training configuration."""
        return self.config.copy()
