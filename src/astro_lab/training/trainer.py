"""
Main AstroLab Trainer

High-level training interface that combines Lightning, MLflow, and Optuna.
"""

from typing import Any, Dict, List, Optional, Union

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from torch.utils.data import DataLoader

from .lightning_module import AstroLightningModule
from .mlflow_logger import AstroMLflowLogger, setup_mlflow_experiment
from .optuna_trainer import OptunaTrainer


class AstroTrainer:
    """High-level trainer for AstroLab models with integrated logging and optimization."""

    def __init__(
        self,
        model: Any,
        task_type: str = "classification",
        experiment_name: str = "astro_experiment",
        max_epochs: int = 100,
        patience: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        accelerator: str = "auto",
        devices: Union[int, List[int]] = "auto",
        precision: str = "16-mixed",
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        **kwargs,
    ):
        self.model = model
        self.task_type = task_type
        self.experiment_name = experiment_name
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.enable_checkpointing = enable_checkpointing
        self.enable_progress_bar = enable_progress_bar
        self.kwargs = kwargs

        # Setup experiment
        setup_mlflow_experiment(experiment_name)

        # Create Lightning module
        self.lightning_module = AstroLightningModule(
            model=model,
            task_type=task_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler=scheduler,
            max_epochs=max_epochs,
        )

        # Setup callbacks
        self.callbacks = self._setup_callbacks()

        # Setup logger
        self.logger = AstroMLflowLogger(
            experiment_name=experiment_name,
            run_name=f"{model.__class__.__name__}_{task_type}",
        )

        # Create trainer
        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=self.callbacks,
            logger=self.logger,
            enable_progress_bar=enable_progress_bar,
            enable_checkpointing=enable_checkpointing,
            **kwargs,
        )

    def _setup_callbacks(self) -> List[Any]:
        """Setup training callbacks."""
        callbacks = []

        # Early stopping
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                mode="min",
                verbose=True,
            )
        )

        # Model checkpointing
        if self.enable_checkpointing:
            callbacks.append(
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3,
                    filename="{epoch:02d}-{val_loss:.3f}",
                    auto_insert_metric_name=False,
                )
            )

        # Progress bar
        if self.enable_progress_bar:
            callbacks.append(RichProgressBar())

        return callbacks

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        log_hyperparameters: bool = True,
        log_model_architecture: bool = True,
        **fit_kwargs,
    ):
        """Train the model."""
        # Log model info
        if log_model_architecture:
            self.logger.log_model_architecture(self.model)

        if log_hyperparameters:
            hyperparams = {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "scheduler": self.scheduler,
                "max_epochs": self.max_epochs,
                "task_type": self.task_type,
                "model_class": self.model.__class__.__name__,
            }
            # Add model-specific hyperparameters
            if hasattr(self.model, "hidden_dim"):
                hyperparams["hidden_dim"] = self.model.hidden_dim
            if hasattr(self.model, "num_layers"):
                hyperparams["num_layers"] = self.model.num_layers
            if hasattr(self.model, "dropout"):
                hyperparams["dropout"] = self.model.dropout

            self.logger.log_hyperparameters(hyperparams)

        # Train model
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            **fit_kwargs,
        )

        # Test if test dataloader provided
        if test_dataloader is not None:
            self.trainer.test(dataloaders=test_dataloader)

        # Log final model
        self.logger.log_final_model(self.model)

    def test(self, test_dataloader: DataLoader):
        """Test the model."""
        return self.trainer.test(self.lightning_module, dataloaders=test_dataloader)

    def predict(self, dataloader: DataLoader):
        """Make predictions."""
        return self.trainer.predict(self.lightning_module, dataloaders=dataloader)

    def optimize_hyperparameters(
        self,
        model_factory: Any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        **optuna_kwargs,
    ):
        """Run hyperparameter optimization."""
        optuna_trainer = OptunaTrainer(
            model_factory=model_factory,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            study_name=f"{self.experiment_name}_optimization",
            **optuna_kwargs,
        )

        study = optuna_trainer.optimize(n_trials=n_trials, timeout=timeout)

        return study

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        self.lightning_module = AstroLightningModule.load_from_checkpoint(
            checkpoint_path,
            model=self.model,
        )
        return self.lightning_module

    def save_model(self, path: str):
        """Save model to path."""
        self.trainer.save_checkpoint(path)

    @property
    def best_model_path(self) -> Optional[str]:
        """Get path to best model checkpoint."""
        checkpoint_callback = None
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break

        if checkpoint_callback:
            return checkpoint_callback.best_model_path
        return None

    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics."""
        return self.trainer.callback_metrics


__all__ = ["AstroTrainer"]
