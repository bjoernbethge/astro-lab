"""
AstroLab Trainer - High-Performance Training Interface

State-of-the-art training with Lightning + MLflow + Optuna integration.
Optimized for astronomical ML workloads with modern Lightning DataModule support.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

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
PrecisionType = Union[
    Literal["64", "32", "16"],
    Literal[
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
    ],
    Literal["64", "32", "16", "bf16"],
    int,
]


class AstroTrainer(Trainer):
    """
    High-performance trainer for astronomical ML models.

    Modern Lightning-based trainer with MLflow integration and astronomical optimizations.
    Inherits directly from Lightning Trainer for maximum compatibility.
    """

    def __init__(
        self,
        lightning_module: AstroLightningModule,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: Union[int, str] = "auto",
        precision: Literal[
            "bf16-mixed"
        ] = "bf16-mixed",  # Default to bf16-mixed for stability
        gradient_clip_val: Optional[float] = 1.0,
        accumulate_grad_batches: int = 1,
        enable_swa: bool = False,
        patience: int = 10,
        monitor: str = "val_loss",
        mode: str = "min",
        experiment_name: str = "astro_experiment",
        checkpoint_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize optimized Lightning Trainer with astronomical ML defaults.

        Args:
            checkpoint_dir: Directory to save checkpoints. If None, uses 'checkpoints/{experiment_name}'
        """
        self.astro_module = lightning_module
        self.experiment_name = experiment_name

        # Setup checkpoint directory
        self.checkpoint_dir = self._setup_checkpoint_dir(
            checkpoint_dir, experiment_name
        )

        # Setup callbacks with astronomical optimizations
        callbacks = self._setup_astro_callbacks(
            enable_swa=enable_swa,
            patience=patience,
            monitor=monitor,
            mode=mode,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Setup logger
        logger = self._setup_astro_logger()

        # Initialize parent Lightning Trainer with optimized defaults
        super().__init__(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=1,  # Always log every step for astronomy
            val_check_interval=1.0,  # Always check validation every epoch
            num_sanity_val_steps=0,  # Skip sanity checks for speed
            enable_progress_bar=True,  # Always show progress
            enable_model_summary=True,  # Always show model summary
            enable_checkpointing=True,  # Always enable checkpointing
            default_root_dir=str(
                self.checkpoint_dir.parent
            ),  # Set root dir for Lightning
            callbacks=callbacks,
            logger=logger,
            **kwargs,
        )

        print("🚀 AstroTrainer initialized:")
        print(f"   - Acceleration: {accelerator}")
        print(f"   - Precision: {precision}")
        print(f"   - Checkpoints: {self.checkpoint_dir}")

    def _setup_checkpoint_dir(
        self, checkpoint_dir: Optional[Union[str, Path]], experiment_name: str
    ) -> Path:
        """Setup checkpoint directory with sensible defaults."""
        if checkpoint_dir is None:
            # Default: checkpoints/{experiment_name}
            checkpoint_path = Path("checkpoints") / experiment_name
        else:
            checkpoint_path = Path(checkpoint_dir)

        # Create directory if it doesn't exist
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        return checkpoint_path

    def _setup_astro_callbacks(
        self,
        enable_swa: bool,
        patience: int,
        monitor: str,
        mode: str,
        checkpoint_dir: Path,
    ) -> List:
        """Setup training callbacks with astronomical ML optimizations."""
        callbacks = []

        # Model checkpointing - save best model in specified directory
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=True,  # Always save last checkpoint
            filename="best-{epoch:02d}-{val_loss:.2f}",
            auto_insert_metric_name=False,
            verbose=True,
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

    def _setup_astro_logger(self):
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
        ckpt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Train the model with support for both DataLoaders and DataModules.

        Args:
            train_dataloader: Training DataLoader (optional if datamodule provided)
            val_dataloader: Validation DataLoader (optional if datamodule provided)
            datamodule: Lightning DataModule (alternative to DataLoaders)
            ckpt_path: Path to checkpoint to resume from (optional)
        """
        if datamodule is not None:
            # Modern Lightning way with DataModule
            super().fit(self.astro_module, datamodule=datamodule, ckpt_path=ckpt_path)
        elif train_dataloader is not None:
            # Traditional way with DataLoaders
            super().fit(
                self.astro_module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=ckpt_path,
            )
        else:
            raise ValueError("Either datamodule or train_dataloader must be provided")

    def test(
        self,
        test_dataloader: Optional[DataLoader] = None,
        datamodule=None,
    ) -> List[Mapping[str, float]]:
        """
        Test the model.

        Args:
            test_dataloader: Test DataLoader (optional if datamodule provided)
            datamodule: Lightning DataModule (alternative to DataLoader)

        Returns:
            Test metrics
        """
        if datamodule is not None:
            return super().test(self.astro_module, datamodule=datamodule)
        elif test_dataloader is not None:
            return super().test(self.astro_module, dataloaders=test_dataloader)
        else:
            raise ValueError("Either datamodule or test_dataloader must be provided")

    def predict(
        self,
        predict_dataloader: Optional[DataLoader] = None,
        datamodule=None,
    ) -> Optional[List[Any]]:
        """
        Run predictions.

        Args:
            predict_dataloader: Prediction DataLoader (optional if datamodule provided)
            datamodule: Lightning DataModule (alternative to DataLoader)

        Returns:
            Predictions
        """
        if datamodule is not None:
            return super().predict(self.astro_module, datamodule=datamodule)
        elif predict_dataloader is not None:
            return super().predict(self.astro_module, dataloaders=predict_dataloader)
        else:
            raise ValueError("Either datamodule or predict_dataloader must be provided")

    def get_metrics(self) -> Dict[str, Any]:
        """Get final training metrics."""
        return {
            k: float(v) if hasattr(v, "item") else v
            for k, v in self.logged_metrics.items()
        }

    @property
    def best_model_path(self) -> Optional[str]:
        """Path to the best saved model."""
        # Access callbacks via the trainer state
        try:
            callbacks = getattr(self, "callbacks", [])
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    return callback.best_model_path
        except AttributeError:
            pass
        return None

    @property
    def last_model_path(self) -> Optional[str]:
        """Path to the last saved model."""
        try:
            callbacks = getattr(self, "callbacks", [])
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    return callback.last_model_path
        except AttributeError:
            pass
        return None

    def load_best_model(self) -> AstroLightningModule:
        """Load the best model from checkpoint."""
        if self.best_model_path:
            return AstroLightningModule.load_from_checkpoint(
                self.best_model_path,
            )
        return self.astro_module

    def load_last_model(self) -> AstroLightningModule:
        """Load the last saved model from checkpoint."""
        if self.last_model_path:
            return AstroLightningModule.load_from_checkpoint(
                self.last_model_path,
            )
        return self.astro_module

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Resume training from a specific checkpoint."""
        print(f"🔄 Resuming training from: {checkpoint_path}")
        self.fit(ckpt_path=checkpoint_path)

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
            raise ImportError("Optuna not available. Install with: uv add optuna")

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
        loaded_module = AstroLightningModule.load_from_checkpoint(checkpoint_path)
        self.astro_module = loaded_module
        return loaded_module

    def save_model(self, path: str) -> None:
        """Save model to path."""
        self.save_checkpoint(path)

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for file in self.checkpoint_dir.glob("*.ckpt"):
            checkpoints.append(file)

        return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Clean up old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()

        # Keep best and last checkpoints
        protected_files = set()
        if self.best_model_path:
            protected_files.add(Path(self.best_model_path).name)
        if self.last_model_path:
            protected_files.add(Path(self.last_model_path).name)

        # Remove old checkpoints beyond keep_last_n
        for checkpoint in checkpoints[keep_last_n:]:
            if checkpoint.name not in protected_files:
                try:
                    checkpoint.unlink()
                    print(f"🗑️  Removed old checkpoint: {checkpoint.name}")
                except OSError as e:
                    print(f"⚠️  Could not remove {checkpoint.name}: {e}")


__all__ = ["AstroTrainer"]
