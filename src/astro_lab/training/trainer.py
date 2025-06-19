"""
AstroLab Trainer - High-Performance Training Interface

State-of-the-art training with Lightning + MLflow + Optuna integration.
Optimized for astronomical ML workloads with modern Lightning DataModule support.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from torch.utils.data import DataLoader

from .lightning_module import AstroLightningModule

# Optional imports
try:
    from .mlflow_logger import AstroMLflowLogger, setup_mlflow_experiment

    MLFLOW_AVAILABLE = True
except ImportError:
    AstroMLflowLogger = None
    setup_mlflow_experiment = None
    MLFLOW_AVAILABLE = False

try:
    from .optuna_trainer import OptunaTrainer

    OPTUNA_AVAILABLE = True
except ImportError:
    OptunaTrainer = None
    OPTUNA_AVAILABLE = False

# Type aliases for clarity
DeviceType = Union[int, List[int], Literal["auto"]]
PrecisionType = Union[Literal["16-mixed", "bf16-mixed", "32", "64"], int]


class AstroTrainer:
    """
    High-performance trainer for astronomical ML models.

    Modern Lightning-based trainer with MLflow integration and astronomical optimizations.
    """

    def __init__(
        self,
        lightning_module: AstroLightningModule,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: Union[int, str] = "auto",
        precision: Union[str, int] = "16-mixed",
        gradient_clip_val: Optional[float] = 1.0,
        accumulate_grad_batches: int = 1,
        enable_swa: bool = False,
        patience: int = 10,
        monitor: str = "val_loss",
        mode: str = "min",
        log_every_n_steps: int = 1,
        val_check_interval: Union[int, float] = 1.0,
        num_sanity_val_steps: int = 0,
        experiment_name: str = "astro_experiment",
        **kwargs,
    ):
        """
        Initialize optimized Lightning Trainer.
        """
        self.lightning_module = lightning_module
        self.experiment_name = experiment_name

        # Setup callbacks
        callbacks = self._setup_callbacks(
            enable_swa=enable_swa, patience=patience, monitor=monitor, mode=mode
        )

        # Setup logger
        logger = self._setup_logger()

        # Create Lightning Trainer
        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=val_check_interval,
            num_sanity_val_steps=num_sanity_val_steps,
            enable_progress_bar=True,
            enable_model_summary=True,
            callbacks=callbacks,
            logger=logger,
            **kwargs,
        )

        print(f"🚀 AstroTrainer initialized with {accelerator} acceleration")

    def _setup_callbacks(
        self, enable_swa: bool, patience: int, monitor: str, mode: str
    ) -> List:
        """Setup training callbacks with astronomical ML optimizations."""
        callbacks = []

        # Model checkpointing - save best model
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.2f}",
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping - prevent overfitting
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            verbose=True,
        )
        callbacks.append(early_stopping)

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Stochastic Weight Averaging (if enabled)
        if enable_swa:
            swa_callback = StochasticWeightAveraging(swa_lrs=1e-4)
            callbacks.append(swa_callback)

        # Rich progress bar for better UX
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)

        return callbacks

    def _setup_logger(self):
        """Setup logger with MLflow if available."""
        if MLFLOW_AVAILABLE and AstroMLflowLogger is not None:
            return AstroMLflowLogger(experiment_name=self.experiment_name)
        else:
            return None

    def fit(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        datamodule=None,
    ) -> None:
        """
        Train the model with support for both DataLoaders and DataModules.

        Args:
            train_dataloader: Training DataLoader (optional if datamodule provided)
            val_dataloader: Validation DataLoader (optional if datamodule provided)
            datamodule: Lightning DataModule (alternative to DataLoaders)
        """
        if datamodule is not None:
            # Modern Lightning way with DataModule
            self.trainer.fit(self.lightning_module, datamodule=datamodule)
        elif train_dataloader is not None:
            # Traditional way with DataLoaders
            self.trainer.fit(
                self.lightning_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
        else:
            raise ValueError("Either datamodule or train_dataloader must be provided")

    def test(
        self,
        test_dataloader: Optional[DataLoader] = None,
        datamodule=None,
    ) -> List[Dict[str, float]]:
        """
        Test the model.

        Args:
            test_dataloader: Test DataLoader (optional if datamodule provided)
            datamodule: Lightning DataModule (alternative to DataLoader)

        Returns:
            Test metrics
        """
        if datamodule is not None:
            return self.trainer.test(self.lightning_module, datamodule=datamodule)
        elif test_dataloader is not None:
            return self.trainer.test(self.lightning_module, dataloaders=test_dataloader)
        else:
            raise ValueError("Either datamodule or test_dataloader must be provided")

    def predict(
        self,
        predict_dataloader: Optional[DataLoader] = None,
        datamodule=None,
    ) -> List[Any]:
        """
        Run predictions.

        Args:
            predict_dataloader: Prediction DataLoader (optional if datamodule provided)
            datamodule: Lightning DataModule (alternative to DataLoader)

        Returns:
            Predictions
        """
        if datamodule is not None:
            return self.trainer.predict(self.lightning_module, datamodule=datamodule)
        elif predict_dataloader is not None:
            return self.trainer.predict(
                self.lightning_module, dataloaders=predict_dataloader
            )
        else:
            raise ValueError("Either datamodule or predict_dataloader must be provided")

    def get_metrics(self) -> Dict[str, Any]:
        """Get final training metrics."""
        return {
            k: float(v) if hasattr(v, "item") else v
            for k, v in self.trainer.logged_metrics.items()
        }

    @property
    def best_model_path(self) -> Optional[str]:
        """Path to the best saved model."""
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback.best_model_path
        return None

    def load_best_model(self) -> AstroLightningModule:
        """Load the best model from checkpoint."""
        if self.best_model_path:
            return AstroLightningModule.load_from_checkpoint(
                self.best_model_path,
            )
        return self.lightning_module

    def optimize_hyperparameters(
        self,
        model_factory: Any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        **optuna_kwargs,
    ) -> Any:
        """Run hyperparameter optimization with modern Optuna integration."""
        if not OPTUNA_AVAILABLE or OptunaTrainer is None:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        optuna_trainer = OptunaTrainer(
            model_factory=model_factory,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            study_name=f"{self.experiment_name}_optimization",
            **optuna_kwargs,
        )

        study = optuna_trainer.optimize(n_trials=n_trials, timeout=timeout)
        return study

    def load_from_checkpoint(self, checkpoint_path: str) -> AstroLightningModule:
        """Load model from checkpoint."""
        self.lightning_module = AstroLightningModule.load_from_checkpoint(
            checkpoint_path
        )
        return self.lightning_module

    def save_model(self, path: str) -> None:
        """Save model to path."""
        self.trainer.save_checkpoint(path)


__all__ = ["AstroTrainer"]
