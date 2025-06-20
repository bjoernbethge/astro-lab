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

from astro_lab.data.config import data_config

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
        enable_swa: bool = False,
        patience: int = 10,
        experiment_name: str = "astro_experiment",
        survey: str = "gaia",
        checkpoint_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize optimized Lightning Trainer with astronomical ML defaults.
        Automatically detects best hardware configuration.

        Args:
            checkpoint_dir: Directory to save checkpoints. If None, uses data_config system
        """
        import torch

        self.astro_module = lightning_module
        self.experiment_name = experiment_name
        self.survey = survey

        # Automatic hardware detection
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if torch.cuda.is_available() else "auto"
        precision = "16-mixed" if torch.cuda.is_available() else "32"

        # Smart defaults
        monitor = "val_loss"
        mode = "min"
        gradient_clip_val = 1.0
        accumulate_grad_batches = 1

        # Setup results structure
        self.results_structure = data_config.ensure_results_directories(
            survey, experiment_name
        )

        # Setup checkpoint directory using data_config
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

        # Filter out parameters that might conflict with parent class
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["accelerator", "devices", "precision", "max_epochs"]
        }

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
            **filtered_kwargs,
        )

        print("🚀 AstroTrainer initialized:")
        print(f"   - Survey: {survey}")
        print(f"   - Experiment: {experiment_name}")
        print(f"   - Acceleration: {accelerator}")
        print(f"   - Precision: {precision}")
        print(f"   - Checkpoints: {self.checkpoint_dir}")
        print(f"   - Results: {self.results_structure['base']}")

    def _setup_checkpoint_dir(
        self, checkpoint_dir: Optional[Union[str, Path]], experiment_name: str
    ) -> Path:
        """Setup checkpoint directory using data_config system."""
        if checkpoint_dir is None:
            # Use data_config system for organized checkpoint management
            data_config.ensure_experiment_directories(experiment_name)
            checkpoint_path = data_config.checkpoints_dir / experiment_name
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
        # Create descriptive filename with experiment name and timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_short = self.experiment_name.replace("_", "-")[:20]  # Limit length

        # Single optimized checkpoint callback - only best model
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=monitor,
            mode=mode,
            save_top_k=1,  # Keep only the best model
            save_last=True,  # Save last checkpoint as 'last.ckpt'
            filename=f"{experiment_short}_best_{{epoch:02d}}_{{val_loss:.3f}}_{timestamp}",
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
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        search_space: Optional[Dict[str, Any]] = None,
        **optuna_kwargs,
    ) -> Any:
        """Optimize hyperparameters using Optuna (if available)."""
        if not OPTUNA_AVAILABLE or OptunaTrainer is None:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        # Create model factory for Optuna that creates different models based on trial suggestions
        def model_factory(trial):
            # Get the original model configuration
            original_model = self.astro_module.model

            # Create hyperparameter suggestions based on search space
            if search_space:
                # Suggest learning rate
                if "learning_rate" in search_space:
                    lr_config = search_space["learning_rate"]
                    if lr_config["type"] == "loguniform":
                        learning_rate = trial.suggest_float(
                            "learning_rate",
                            lr_config["low"],
                            lr_config["high"],
                            log=True,
                        )
                    else:
                        learning_rate = trial.suggest_float(
                            "learning_rate", lr_config["low"], lr_config["high"]
                        )
                else:
                    learning_rate = trial.suggest_float(
                        "learning_rate", 1e-5, 1e-2, log=True
                    )

                # Suggest hidden_dim
                if "hidden_dim" in search_space:
                    hd_config = search_space["hidden_dim"]
                    if hd_config["type"] == "categorical":
                        hidden_dim = trial.suggest_categorical(
                            "hidden_dim", hd_config["choices"]
                        )
                    else:
                        hidden_dim = trial.suggest_int(
                            "hidden_dim", hd_config["low"], hd_config["high"]
                        )
                else:
                    hidden_dim = trial.suggest_categorical(
                        "hidden_dim", [64, 128, 256, 512]
                    )

                # Suggest dropout
                if "dropout" in search_space:
                    dropout_config = search_space["dropout"]
                    dropout = trial.suggest_float(
                        "dropout", dropout_config["low"], dropout_config["high"]
                    )
                else:
                    dropout = trial.suggest_float("dropout", 0.1, 0.5)

                # Create new model with suggested parameters
                from astro_lab.models import AstroSurveyGNN

                new_model = AstroSurveyGNN(
                    hidden_dim=hidden_dim,
                    output_dim=getattr(original_model, "output_dim", 8),
                    dropout=dropout,
                    num_layers=getattr(original_model, "num_layers", 3),
                    conv_type=getattr(original_model, "conv_type", "gcn"),
                    task=getattr(original_model, "task", "node_classification"),
                )

                # Create new Lightning module with suggested learning rate
                lightning_module = AstroLightningModule(
                    model=new_model, learning_rate=learning_rate
                )

                return lightning_module
            else:
                # Default search space if none provided
                learning_rate = trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                )
                hidden_dim = trial.suggest_categorical(
                    "hidden_dim", [64, 128, 256, 512]
                )
                dropout = trial.suggest_float("dropout", 0.1, 0.5)

                from astro_lab.models import AstroSurveyGNN

                new_model = AstroSurveyGNN(
                    hidden_dim=hidden_dim,
                    output_dim=getattr(original_model, "output_dim", 8),
                    dropout=dropout,
                    num_layers=getattr(original_model, "num_layers", 3),
                    conv_type=getattr(original_model, "conv_type", "gcn"),
                    task=getattr(original_model, "task", "node_classification"),
                )

                lightning_module = AstroLightningModule(
                    model=new_model, learning_rate=learning_rate
                )

                return lightning_module

        # Create Optuna trainer
        optuna_trainer = OptunaTrainer(
            model_factory=model_factory,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            mlflow_experiment=f"{self.experiment_name}_optuna",
            **optuna_kwargs,
        )

        # Run optimization
        return optuna_trainer.optimize(n_trials=n_trials, timeout=timeout)

    def load_from_checkpoint(self, checkpoint_path: str) -> AstroLightningModule:
        """Load model from checkpoint."""
        return AstroLightningModule.load_from_checkpoint(checkpoint_path)

    def save_model(self, path: str) -> None:
        """Save model to specified path."""
        import torch

        torch.save(self.astro_module.state_dict(), path)

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for file in self.checkpoint_dir.iterdir():
            if file.suffix == ".ckpt":
                checkpoints.append(file)

        return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Clean up old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()

        # Keep best and last checkpoints
        protected_files = set()
        if self.best_model_path:
            protected_files.add(Path(self.best_model_path))
        if self.last_model_path:
            protected_files.add(Path(self.last_model_path))

        # Remove old checkpoints beyond keep_last_n
        for checkpoint in checkpoints[keep_last_n:]:
            if checkpoint not in protected_files:
                checkpoint.unlink()
                print(f"🗑️  Removed old checkpoint: {checkpoint.name}")

    def save_best_models_to_results(self, top_k: int = 3) -> Dict[str, Path]:
        """Save top K best models to organized results structure."""
        import shutil
        from datetime import datetime

        if not self.results_structure:
            print("❌ No results structure available")
            return {}

        models_dir = self.results_structure["models"]

        # Get all checkpoints sorted by validation loss (best first)
        checkpoints = []
        for checkpoint_file in self.list_checkpoints():
            # Extract validation loss from filename if available
            name = checkpoint_file.name
            if "best_" in name and "_" in name:
                try:
                    # Pattern: experiment_best_epoch_loss_timestamp.ckpt
                    parts = name.split("_")
                    for i, part in enumerate(parts):
                        if part == "best" and i + 2 < len(parts):
                            loss_str = parts[i + 2].replace(".ckpt", "").split("_")[0]
                            loss = float(loss_str)
                            checkpoints.append((loss, checkpoint_file))
                            break
                except (ValueError, IndexError):
                    continue

        # Sort by loss (best first) and take top K
        checkpoints.sort(key=lambda x: x[0])
        best_checkpoints = checkpoints[:top_k]

        saved_models = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for rank, (loss, checkpoint_file) in enumerate(best_checkpoints):
            # Create clean, descriptive filename
            if rank == 0:
                target_name = "best_model.ckpt"
            else:
                target_name = f"model_rank_{rank + 1}.ckpt"
            target_path = models_dir / target_name

            # Copy checkpoint to results
            shutil.copy2(checkpoint_file, target_path)
            saved_models[f"rank_{rank}"] = target_path

            print(f"💾 Saved {target_name}: val_loss={loss:.3f}")

        # Create README for models
        self._create_models_readme(saved_models, best_checkpoints)

        return saved_models

    def _create_models_readme(
        self, saved_models: Dict[str, Path], checkpoints_info: List
    ):
        """Create README for saved models."""
        readme_path = self.results_structure["models"] / "README.md"
        from datetime import datetime

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# Best Models - {self.survey.upper()} {self.experiment_name}\n\n")
            f.write(f"**Survey**: {self.survey}\n")
            f.write(f"**Experiment**: {self.experiment_name}\n")
            f.write(
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write("## Models\n\n")

            for rank, (loss, original_path) in enumerate(checkpoints_info):
                if f"rank_{rank}" in saved_models:
                    model_path = saved_models[f"rank_{rank}"]
                    f.write(f"### Rank {rank} - {model_path.name}\n")
                    f.write(f"- **Validation Loss**: {loss:.6f}\n")
                    f.write(f"- **Original**: {original_path.name}\n")
                    f.write(
                        f"- **Size**: {model_path.stat().st_size / 1024:.1f} KB\n\n"
                    )

            f.write("## Usage\n\n")
            f.write("```python\n")
            f.write("from astro_lab.training.trainer import AstroTrainer\n\n")
            f.write("# Load best model\n")
            f.write(
                f'model = AstroTrainer.load_from_checkpoint("{saved_models.get("rank_0", "")}")\n'
            )
            f.write("```\n")

        print(f"📄 Created models README: {readme_path}")

    def get_results_summary(self) -> Dict[str, Any]:
        """Get comprehensive results summary."""
        from datetime import datetime

        return {
            "survey": self.survey,
            "experiment": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "results_structure": {k: str(v) for k, v in self.results_structure.items()},
            "best_model_path": self.best_model_path,
            "last_model_path": self.last_model_path,
            "total_checkpoints": len(self.list_checkpoints()),
        }

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], lightning_module: AstroLightningModule
    ) -> "AstroTrainer":
        """Create AstroTrainer from configuration dictionary."""
        training_config = config.get("training", {})
        callbacks_config = config.get("callbacks", {})
        early_stopping_config = callbacks_config.get("early_stopping", {})

        return cls(
            lightning_module=lightning_module,
            max_epochs=training_config.get("max_epochs", 100),
            enable_swa=training_config.get("enable_swa", False),
            patience=early_stopping_config.get(
                "patience", training_config.get("patience", 10)
            ),
            experiment_name=config.get("mlflow", {}).get(
                "experiment_name", "astro_experiment"
            ),
        )


__all__ = ["AstroTrainer"]
