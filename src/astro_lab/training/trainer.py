"""
AstroLab Trainer - High-Performance Training Interface

State-of-the-art training with Lightning + MLflow + Optuna integration.
Optimized for astronomical ML workloads with modern Lightning DataModule support.
Updated for Lightning 2.0+ compatibility and modern ML practices.
"""

import gc
import logging
import os
import shutil
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

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

from astro_lab.data.config import data_config
from astro_lab.models.config import (
    EncoderConfig,
    GraphConfig,
    ModelConfig,
    OutputConfig,
)
from astro_lab.training.config import TrainingConfig as FullTrainingConfig

# Removed memory.py - using simple context managers
from .lightning_module import AstroLightningModule

# Optional imports
from .mlflow_logger import AstroMLflowLogger, setup_mlflow_experiment




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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings - import warnings was already done above
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AstroTrainer(Trainer):
    """
    High-performance trainer for astronomical ML models.

    Modern Lightning-based trainer with MLflow integration and astronomical optimizations.
    Inherits directly from Lightning Trainer for maximum compatibility.
    Updated for Lightning 2.0+ and modern ML practices.
    """

    def __init__(
        self,
        lightning_module: Optional[AstroLightningModule] = None,
        training_config: Optional[FullTrainingConfig] = None,
        **kwargs,
    ):
        """
        Initialize AstroTrainer with modern Lightning 2.0+ compatibility.

        Args:
            lightning_module: Pre-configured Lightning module
            training_config: Training configuration
            **kwargs: Additional trainer parameters
        """
        # Simple initialization without complex context manager
        # Store configurations
        self.training_config = training_config
        self._lightning_module = lightning_module

        # Create default training config if none provided
        if training_config is None:
            from astro_lab.models.config import (
                EncoderConfig,
                GraphConfig,
                ModelConfig,
                OutputConfig,
            )
            from astro_lab.training.config import TrainingConfig

            # Create a minimal model config
            model_config = ModelConfig(
                name="default_model",
                encoder=EncoderConfig(),
                graph=GraphConfig(),
                output=OutputConfig(),
            )

            training_config = TrainingConfig(
                name="default_training", model=model_config
            )
            self.training_config = training_config

        # Validate training config
        assert isinstance(self.training_config, FullTrainingConfig), (
            "training_config must be a TrainingConfig instance"
        )

        # Get model config from training config
        model_config = self.training_config.model

        # Create Lightning module if not provided
        if lightning_module is None:
            lightning_module = AstroLightningModule(
                model_config=model_config, training_config=training_config
            )

        self.astro_module = lightning_module
        self.experiment_name = self.training_config.logging.experiment_name
        self.survey = model_config.name if hasattr(model_config, "name") else "unknown"

        # Extract hardware configuration
        accelerator = self.training_config.hardware.accelerator
        devices = self.training_config.hardware.devices
        precision = self.training_config.hardware.precision

        # Setup checkpoint directory
        self.checkpoint_dir = self._setup_checkpoint_dir(
            checkpoint_dir=None,  # Use default from data_config
            experiment_name=self.experiment_name,
        )

        # Setup callbacks and logger
        callbacks = self._setup_astro_callbacks(
            enable_swa=self.training_config.callbacks.swa,
            patience=self.training_config.callbacks.early_stopping_patience,
            monitor=self.training_config.callbacks.monitor,
            mode=self.training_config.callbacks.mode,
            checkpoint_dir=self.checkpoint_dir,
        )
        logger = self._setup_astro_logger()

        # Filter kwargs to avoid conflicts
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["accelerator", "devices", "precision", "max_epochs"]
        }

        # Determine if we should disable Lightning's default logging
        disable_lightning_logs = (
            getattr(self.training_config.logging, "use_mlflow", False)
            and logger is not None
        )

        # Remove UI parameters from filtered_kwargs to avoid duplication - these are always set
        ui_params = ['enable_progress_bar', 'enable_model_summary', 'enable_checkpointing']
        for param in ui_params:
            filtered_kwargs.pop(param, None)
        
        # Initialize parent Trainer with modern parameters
        super().__init__(
            max_epochs=self.training_config.scheduler.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            logger=logger,
            # UI parameters - always enabled, not configurable
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            # Disable default Lightning logs directory when using MLflow
            default_root_dir=None if disable_lightning_logs else None,
            **filtered_kwargs,
        )

        print(f"🚀 AstroTrainer (Lightning 2.0+) for {self.survey} initialized!")

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

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=True,
        )
        callbacks.append(early_stopping)

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Rich progress bar for better UX
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)

        # Stochastic Weight Averaging (optional)
        if enable_swa:
            swa = StochasticWeightAveraging(swa_lrs=1e-2)
            callbacks.append(swa)

        return callbacks

    def _setup_astro_logger(self):
        """Setup logging with MLflow integration."""
        if getattr(self.training_config.logging, "use_mlflow", False):
            try:
                tracking_uri = getattr(
                    self.training_config.logging, "mlflow_tracking_uri", None
                )
                if not tracking_uri:
                    tracking_uri = "file:./mlruns"
                    print(
                        "⚠️ No tracking_uri found in Config, use Default: file:./mlruns"
                    )
                logger = AstroMLflowLogger(
                    experiment_name=self.experiment_name,
                    tracking_uri=tracking_uri,
                )
                return logger
            except Exception as e:
                print(f"⚠️ MLflow logger setup failed: {e}")
                return None
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
        Fit the model with modern Lightning 2.0+ compatibility.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            datamodule: Lightning DataModule
            ckpt_path: Checkpoint path for resuming
        """
        try:
            # Automatische Klassenableitung vor dem Training
            self._auto_detect_classes(train_dataloader, val_dataloader, datamodule)

            # Use the lightning module directly
            if datamodule is not None:
                # Use DataModule (recommended approach)
                super().fit(
                    model=self.astro_module,
                    datamodule=datamodule,
                    ckpt_path=ckpt_path,
                )
            else:
                # Use individual dataloaders
                super().fit(
                    model=self.astro_module,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path=ckpt_path,
                )

            # Cleanup after training
            self._cleanup_after_training()

        except Exception as e:
            print(f"❌ Training failed: {e}")
            # Cleanup even on error
            self._cleanup_after_training()
            raise

    def _cleanup_after_training(self):
        """Clean up memory after training to prevent leaks."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 CUDA cache cleared")

            # Force garbage collection
            gc.collect()

            # Clear any cached tensors in the module
            if hasattr(self.astro_module, "model"):
                for param in self.astro_module.model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

            print("🧹 Memory cleanup completed")

        except Exception as e:
            print(f"⚠️ Memory cleanup failed: {e}")

    def _auto_detect_classes(self, train_dataloader, val_dataloader, datamodule):
        """Automatic class detection from data."""
        try:
            # Determine DataLoader for class detection
            target_dataloader = None
            if datamodule is not None and hasattr(datamodule, "train_dataloader"):
                target_dataloader = datamodule.train_dataloader()
            elif train_dataloader is not None:
                target_dataloader = train_dataloader
            elif val_dataloader is not None:
                target_dataloader = val_dataloader

            if target_dataloader is None:
                print("⚠️ No DataLoader available for class detection")
                return

            # Class detection from data
            targets = []

            for i, batch in enumerate(target_dataloader):
                t = None
                if isinstance(batch, dict):
                    t = batch.get("target") or batch.get("y")
                elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                    t = batch[1]
                elif hasattr(batch, "y"):  # PyTorch Geometric DataBatch
                    t = batch.y
                elif hasattr(batch, "target"):  # Alternative target attribute
                    t = batch.target
                else:
                    # Try various possible target attributes
                    for attr_name in ["y", "target", "labels", "class", "classes"]:
                        if hasattr(batch, attr_name):
                            attr_value = getattr(batch, attr_name)
                            if attr_value is not None and hasattr(attr_value, "shape"):
                                t = attr_value
                                break
                    else:
                        t = None

                if t is not None:
                    targets.append(t.flatten())
                if i > 5:  # Limit for efficiency
                    break

            if targets:
                all_targets = torch.cat(targets)
                num_classes = int(all_targets.max().item()) + 1
                print(
                    f"🔍 Automatically detected {num_classes} classes from training data (min={all_targets.min().item()}, max={all_targets.max().item()})."
                )

                # Update Lightning module with correct number of classes
                if (
                    hasattr(self.astro_module, "num_classes")
                    and self.astro_module.num_classes != num_classes
                ):
                    print(
                        f"🔄 Updating Lightning module from {self.astro_module.num_classes} to {num_classes} classes"
                    )
                    self.astro_module.num_classes = num_classes

                    # Recreate model with correct number of classes
                    if (
                        hasattr(self.astro_module, "model_config")
                        and self.astro_module.model_config is not None
                    ):
                        # Update model config
                        if hasattr(self.astro_module.model_config, "output"):
                            self.astro_module.model_config.output.output_dim = (
                                num_classes
                            )

                        # Recreate model
                        try:
                            self.astro_module.model = (
                                self.astro_module._create_model_from_config(
                                    self.astro_module.model_config
                                )
                            )
                            print(f"🔄 Model recreated with {num_classes} classes")
                        except Exception as e:
                            print(f"❌ Error recreating model: {e}")
                            # Fallback
                            from astro_lab.models.astro import AstroSurveyGNN

                            self.astro_module.model = AstroSurveyGNN(
                                input_dim=16,  # Default
                                hidden_dim=128,  # Default
                                output_dim=num_classes,
                                conv_type="gcn",
                                num_layers=3,
                                dropout=0.1,
                                task="stellar_classification",
                            )
                            print(
                                f"🔄 Fallback model created with {num_classes} classes"
                            )
                    else:
                        # Fallback: Create new model directly
                        from astro_lab.models.astro import AstroSurveyGNN

                        self.astro_module.model = AstroSurveyGNN(
                            input_dim=16,  # Default
                            hidden_dim=128,  # Default
                            output_dim=num_classes,
                            conv_type="gcn",
                            num_layers=3,
                            dropout=0.1,
                            task="stellar_classification",
                        )
                        print(f"🔄 New model created with {num_classes} classes")

                    # Recreate metrics
                    self.astro_module._setup_metrics()
                    print(f"🔄 Metrics recreated with {num_classes} classes")
            else:
                print("⚠️ Could not detect classes from training data")

        except Exception as e:
            print(f"❌ Error during automatic class detection: {e}")
            print("⚠️ Using default number of classes")

    def test(
        self,
        test_dataloader: Optional[DataLoader] = None,
        datamodule=None,
    ) -> List[Mapping[str, float]]:
        """
        Test the model with modern Lightning 2.0+ compatibility.

        Args:
            test_dataloader: Test data loader
            datamodule: Lightning DataModule

        Returns:
            Test results
        """
        try:
            if datamodule is not None:
                results = super().test(
                    model=self.astro_module,
                    datamodule=datamodule,
                )
            else:
                results = super().test(
                    model=self.astro_module,
                    dataloaders=test_dataloader,
                )

            # Cleanup after testing
            self._cleanup_after_training()
            return results

        except Exception as e:
            print(f"❌ Testing failed: {e}")
            self._cleanup_after_training()
            raise

    def predict(
        self,
        predict_dataloader: Optional[DataLoader] = None,
        datamodule=None,
    ) -> Optional[List[Any]]:
        """
        Make predictions with modern Lightning 2.0+ compatibility.

        Args:
            predict_dataloader: Prediction data loader
            datamodule: Lightning DataModule

        Returns:
            Predictions
        """
        try:
            if datamodule is not None:
                return super().predict(
                    model=self.astro_module,
                    datamodule=datamodule,
                )
            else:
                return super().predict(
                    model=self.astro_module,
                    dataloaders=predict_dataloader,
                )
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return {
            "best_model_path": self.best_model_path,
            "last_model_path": self.last_model_path,
            "experiment_name": self.experiment_name,
            "survey": self.survey,
        }

    @property
    def best_model_path(self) -> Optional[str]:
        """Get path to best model checkpoint."""
        if (
            hasattr(self, "checkpoint_callback")
            and self.checkpoint_callback is not None
        ):
            return self.checkpoint_callback.best_model_path
        return None

    @property
    def last_model_path(self) -> Optional[str]:
        """Get path to last model checkpoint."""
        if (
            hasattr(self, "checkpoint_callback")
            and self.checkpoint_callback is not None
        ):
            return self.checkpoint_callback.last_model_path
        return None

    def load_best_model(self) -> AstroLightningModule:
        """Load the best model from checkpoint."""
        if self.best_model_path is None:
            raise ValueError("No best model checkpoint found")

        return AstroLightningModule.load_from_checkpoint(self.best_model_path)

    def load_last_model(self) -> AstroLightningModule:
        """Load the last model from checkpoint."""
        if self.last_model_path is None:
            raise ValueError("No last model checkpoint found")

        return AstroLightningModule.load_from_checkpoint(self.last_model_path)

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Resume training from checkpoint."""
        self.fit(ckpt_path=str(checkpoint_path))

    def optimize_hyperparameters(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        search_space: Optional[Dict[str, Any]] = None,
        **optuna_kwargs,
    ) -> Any:
        """
        Optimize hyperparameters using Optuna.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            n_trials: Number of optimization trials
            timeout: Optimization timeout
            search_space: Hyperparameter search space
            **optuna_kwargs: Additional Optuna parameters

        Returns:
            Optimization results
        """
        # Optuna is now always available

        def model_factory(trial):
            # Get the original model configuration
            original_config = self.training_config.model

            # Create new config with trial parameters
            if search_space:
                # Use provided search space
                trial_params = {}
                for param_name, param_config in search_space.items():
                    if param_config["type"] == "float":
                        trial_params[param_name] = trial.suggest_float(
                            param_name, param_config["low"], param_config["high"]
                        )
                    elif param_config["type"] == "int":
                        trial_params[param_name] = trial.suggest_int(
                            param_name, param_config["low"], param_config["high"]
                        )
                    elif param_config["type"] == "categorical":
                        trial_params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )
            else:
                # Use default search space
                trial_params = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-2, log=True
                    ),
                    "hidden_dim": trial.suggest_int("hidden_dim", 64, 512),
                    "num_layers": trial.suggest_int("num_layers", 2, 6),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                }

            # Create new model config with trial parameters
            new_config = ModelConfig(
                name=original_config.name,
                encoder=original_config.encoder,
                graph=GraphConfig(
                    conv_type=original_config.graph.conv_type,
                    hidden_dim=trial_params.get(
                        "hidden_dim", original_config.graph.hidden_dim
                    ),
                    num_layers=trial_params.get(
                        "num_layers", original_config.graph.num_layers
                    ),
                    dropout=trial_params.get("dropout", original_config.graph.dropout),
                ),
                output=original_config.output,
            )

            # Create new training config
            new_training_config = FullTrainingConfig(
                name=f"{self.training_config.name}_optuna",
                model=new_config,
                scheduler=self.training_config.scheduler,
                hardware=self.training_config.hardware,
                callbacks=self.training_config.callbacks,
                logging=self.training_config.logging,
            )

            # Create new lightning module
            return AstroLightningModule(
                model_config=new_config,
                training_config=new_training_config,
                learning_rate=trial_params.get("learning_rate", 1e-3),
            )

        # Create Optuna trainer
        from .optuna_trainer import OptunaTrainer
        optuna_trainer = OptunaTrainer(
            model_factory=model_factory,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            **optuna_kwargs,
        )

        # Run optimization
        return optuna_trainer.optimize(
            n_trials=n_trials,
            timeout=timeout,
        )

    def load_from_checkpoint(self, checkpoint_path: str) -> AstroLightningModule:
        """Load model from checkpoint."""
        return AstroLightningModule.load_from_checkpoint(checkpoint_path)

    def save_model(self, path: str) -> None:
        """Save model to path."""
        if self.astro_module is not None:
            torch.save(self.astro_module.state_dict(), path)

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoints in checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Clean up old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > keep_last_n:
            checkpoints_to_remove = checkpoints[keep_last_n:]
            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint.unlink()
                    print(f"🗑️ Removed old checkpoint: {checkpoint.name}")
                except Exception as e:
                    print(f"⚠️ Failed to remove checkpoint {checkpoint.name}: {e}")

    def save_best_models_to_results(self, top_k: int = 3) -> Dict[str, Path]:
        """
        Save best models to results directory with organized structure.

        Args:
            top_k: Number of best models to save

        Returns:
            Dictionary mapping model names to saved paths
        """
        try:
            # Get survey name for better organization
            survey = getattr(self, "survey", "gaia")

            # Create organized results structure
            results_dir = Path(f"./results/{survey}")
            models_dir = results_dir / "models"
            plots_dir = results_dir / "plots"

            models_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Get all checkpoints
            checkpoint_dir = Path(f"./experiments/{survey}/checkpoints")
            if not checkpoint_dir.exists():
                print(f"⚠️ No checkpoint directory found: {checkpoint_dir}")
                return {}

            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                print(f"⚠️ No checkpoints found in: {checkpoint_dir}")
                return {}

            # Sort by validation loss (best first)
            def extract_val_loss(checkpoint_path):
                try:
                    # Extract validation loss from filename
                    filename = checkpoint_path.stem
                    if "val_loss=" in filename:
                        loss_str = filename.split("val_loss=")[1].split("_")[0]
                        return float(loss_str)
                    return float("inf")  # Put files without loss info at the end
                except:
                    return float("inf")

            checkpoints.sort(key=extract_val_loss)

            # Save top-k models with descriptive names
            saved_models = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, checkpoint_path in enumerate(checkpoints[:top_k]):
                # Create descriptive filename
                val_loss = extract_val_loss(checkpoint_path)
                if val_loss == float("inf"):
                    model_name = f"{survey}_model_{i + 1}_{timestamp}"
                else:
                    model_name = (
                        f"{survey}_model_{i + 1}_val_loss_{val_loss:.4f}_{timestamp}"
                    )

                # Save to results/models
                target_path = models_dir / f"{model_name}.ckpt"

                try:
                    shutil.copy2(checkpoint_path, target_path)
                    saved_models[model_name] = target_path
                    print(f"💾 Saved model {i + 1}: {target_path.name}")
                except Exception as e:
                    print(f"❌ Failed to save model {i + 1}: {e}")

            # Create README with model information
            self._create_models_readme(saved_models, checkpoints[:top_k])

            print(f"✅ Saved {len(saved_models)} models to: {models_dir}")
            return saved_models

        except Exception as e:
            print(f"❌ Failed to save models to results: {e}")
            return {}

    def _create_models_readme(
        self, saved_models: Dict[str, Path], checkpoints_info: List
    ):
        """Create README file with model information."""
        try:
            results_dir = data_config.results_dir / self.experiment_name
            readme_path = results_dir / "README.md"

            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"# {self.experiment_name} - Model Results\n\n")
                f.write(f"**Survey:** {self.survey}\n")
                f.write(f"**Experiment:** {self.experiment_name}\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("## Saved Models\n\n")
                for model_name, model_path in saved_models.items():
                    f.write(f"- **{model_name}**: `{model_path.name}`\n")

                f.write("\n## Training Information\n\n")
                f.write(f"- **Model Type:** {type(self.astro_module.model).__name__}\n")
                f.write(f"- **Task Type:** {self.astro_module.task_type}\n")
                f.write(
                    f"- **Number of Classes:** {getattr(self.astro_module, 'num_classes', 'N/A')}\n"
                )

                if self.training_config:
                    f.write(
                        f"- **Max Epochs:** {self.training_config.scheduler.max_epochs}\n"
                    )
                    f.write(
                        f"- **Learning Rate:** {getattr(self.astro_module, 'learning_rate', 'N/A')}\n"
                    )

                f.write("\n## Checkpoints\n\n")
                for i, checkpoint in enumerate(checkpoints_info[:10]):  # Show top 10
                    f.write(f"{i + 1}. `{checkpoint.name}`\n")

            print(f"📝 Created README at {readme_path}")

        except Exception as e:
            print(f"⚠️ Failed to create README: {e}")

    def get_results_summary(self) -> Dict[str, Any]:
        """Get comprehensive results summary."""
        return {
            "experiment_name": self.experiment_name,
            "survey": self.survey,
            "best_model_path": self.best_model_path,
            "last_model_path": self.last_model_path,
            "checkpoint_dir": str(self.checkpoint_dir),
            "results_structure": {
                "base": str(data_config.results_dir / self.experiment_name),
                "checkpoints": str(self.checkpoint_dir),
                "logs": str(data_config.logs_dir / self.experiment_name),
            },
            "model_info": {
                "type": type(self.astro_module.model).__name__,
                "task_type": self.astro_module.task_type,
                "num_classes": getattr(self.astro_module, "num_classes", "N/A"),
            },
            "training_info": {
                "max_epochs": self.training_config.scheduler.max_epochs
                if self.training_config
                else "N/A",
                "learning_rate": getattr(self.astro_module, "learning_rate", "N/A"),
            },
        }

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], lightning_module: AstroLightningModule
    ) -> "AstroTrainer":
        """
        Create AstroTrainer from configuration dictionary.

        Args:
            config: Configuration dictionary
            lightning_module: Pre-configured Lightning module

        Returns:
            AstroTrainer instance
        """
        # Extract trainer-specific parameters
        trainer_params = config.get("trainer", {})

        # Create trainer
        return cls(
            lightning_module=lightning_module,
            **trainer_params,
        )

    @property
    def lightning_module(self) -> Optional[AstroLightningModule]:
        """Get the lightning module."""
        return self._lightning_module


__all__ = ["AstroTrainer"]
