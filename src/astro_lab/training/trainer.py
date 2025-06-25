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

# MLflow and Optuna imports
import mlflow
import optuna
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

from astro_lab.data.config import data_config
from astro_lab.models.config import (
    EncoderConfig,
    GraphConfig,
    ModelConfig,
    OutputConfig,
)
from astro_lab.training.config import TrainingConfig

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

# Configure logging - only errors
logging.basicConfig(level=logging.ERROR)
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
        training_config: Optional[TrainingConfig] = None,
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
        self._optuna_study = None  # Will be set during hyperparameter optimization

        # Create default training config if none provided
        if training_config is None:
            # Only pass valid ModelConfig fields after refactoring.
            model_config = ModelConfig(name="default_model")

            training_config = TrainingConfig(
                name="default_training", model=model_config
            )
            self.training_config = training_config

        # Validate training config
        assert isinstance(self.training_config, TrainingConfig), (
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
        logger_instance = self._setup_astro_logger()

        # Filter kwargs to avoid conflicts
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["accelerator", "devices", "precision", "max_epochs"]
        }

        # Determine logging configuration based on Lightning best practices
        # Reference: https://lightning.ai/docs/pytorch/stable/api_references.html
        use_mlflow = getattr(self.training_config.logging, "use_mlflow", False)

        # Configure default_root_dir properly
        if use_mlflow and logger_instance is not None:
            # When using MLflow, disable Lightning's default logs by setting default_root_dir=None
            # This prevents the lightning_logs directory from being created
            default_root_dir = None

            # Also ensure we don't pass logger=False, keep our MLflow logger
            final_logger = logger_instance
        else:
            # When not using MLflow, use checkpoint_dir as root for Lightning's default TensorBoardLogger
            default_root_dir = str(self.checkpoint_dir)
            final_logger = True  # Use Lightning's default TensorBoardLogger

        # Remove UI parameters from filtered_kwargs to avoid duplication - these are always set
        ui_params = [
            "enable_progress_bar",
            "enable_model_summary",
            "enable_checkpointing",
            "default_root_dir",  # Also remove this to avoid conflicts
        ]
        for param in ui_params:
            filtered_kwargs.pop(param, None)

        # Initialize parent Trainer with modern parameters
        super().__init__(
            max_epochs=self.training_config.scheduler.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            logger=final_logger,
            # UI parameters - always enabled, not configurable
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            # Set default_root_dir according to Lightning best practices
            default_root_dir=default_root_dir,
            **filtered_kwargs,
        )

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
            verbose=False,  # Changed to False
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=False,  # Changed to False
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
        if getattr(self.training_config.logging, "use_mlflow", True):  # Default to True
            try:
                tracking_uri = getattr(
                    self.training_config.logging, "tracking_uri", None
                )
                if not tracking_uri:
                    # Use data_config system for organized MLflow storage
                    data_config.ensure_experiment_directories(self.experiment_name)
                    exp_paths = data_config.get_experiment_paths(self.experiment_name)
                    # Ensure absolute path before using as_uri() for Windows compatibility
                    tracking_uri = exp_paths["mlruns"].resolve().as_uri()

                logger_instance = AstroMLflowLogger(
                    experiment_name=self.experiment_name,
                    tracking_uri=tracking_uri,
                    artifact_location=getattr(
                        self.training_config.logging, "tracking_uri", None
                    ),
                    enable_system_metrics=getattr(
                        self.training_config.logging, "enable_system_metrics", True
                    ),
                )
                return logger_instance
            except Exception as e:
                logger.error(f"MLflow logger setup failed: {e}")
                return None
        else:
            return None

    def fit(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        datamodule: Any = None,
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
            # Extract survey name from datamodule if available
            if datamodule and hasattr(datamodule, "survey"):
                self.survey = datamodule.survey
            else:
                self.survey = "unknown"

            # Use the lightning module directly - Lightning expects the LightningModule as model parameter
            if datamodule is not None:
                # Use DataModule (recommended approach)
                # Pass the LightningModule as model parameter to the parent Trainer
                super().fit(
                    model=self.astro_module,  # This is the LightningModule
                    datamodule=datamodule,
                    ckpt_path=ckpt_path,
                )
            elif train_dataloader is not None or val_dataloader is not None:
                # Use individual dataloaders
                super().fit(
                    model=self.astro_module,  # This is the LightningModule
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path=ckpt_path,
                )
            else:
                # No dataloaders provided - this should not happen with proper DataModule
                raise ValueError(
                    "No dataloaders or datamodule provided. Cannot train without data."
                )

            # Cleanup after training
            self._cleanup_after_training()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Cleanup even on error
            self._cleanup_after_training()
            raise

    def _cleanup_after_training(self):
        """Clean up memory after training to prevent leaks."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Clear any cached tensors in the module
            if hasattr(self.astro_module, "model"):
                for param in self.astro_module.model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

            # Automatically save results and create plots if enabled
            if getattr(self.training_config.logging, "save_results_to_disk", True):
                self._save_training_results()

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def _save_training_results(self):
        """Save training results to organized directory structure."""
        try:
            # Use the survey name extracted during fit()
            survey = getattr(self, "survey", "unknown")

            # Extract model name from training config
            model_name = "unknown"
            if self.training_config and hasattr(self.training_config, "model"):
                model_name = getattr(self.training_config.model, "name", "unknown")

            # Create organized results structure: results/survey/model/
            results_structure = data_config.ensure_results_directories(
                survey, model_name
            )

            models_dir = results_structure["models"]

            # Save best models if enabled
            if getattr(self.training_config.logging, "save_best_models", True):
                top_k = getattr(self.training_config.logging, "top_k_models", 3)
                saved_models = self.save_best_models_to_results(top_k)
                logger.info(f"✅ Saved {len(saved_models)} best models to results")

            # Create MLflow and Optuna plots using their built-in functions
            if getattr(self.training_config.logging, "create_plots", True):
                self._create_mlflow_optuna_plots(results_structure["plots"])

            # Create comprehensive results summary
            self._create_results_summary(results_structure["base"])

            logger.info(f"📊 Training results saved to: {results_structure['base']}")

        except Exception as e:
            logger.error(f"Failed to save training results: {e}")

    def _create_mlflow_optuna_plots(self, plots_dir: Path):
        """Create plots using MLflow and Optuna built-in visualization functions (2025 Best Practices)."""
        try:
            import mlflow
            import optuna

            # Create plots directory
            plots_dir.mkdir(parents=True, exist_ok=True)

            # 1. MLflow built-in plots (if we have an active run)
            if mlflow.active_run():
                self._create_mlflow_plots(plots_dir)

            # 2. Optuna plots (if we have optimization results)
            if hasattr(self, "_optuna_study") and self._optuna_study is not None:
                self._create_optuna_plots(plots_dir)

            logger.info(f"📈 MLflow/Optuna plots created in: {plots_dir}")

        except Exception as e:
            logger.error(f"Failed to create MLflow/Optuna plots: {e}")

    def _create_mlflow_plots(self, plots_dir: Path):
        """Create plots using MLflow's built-in visualization functions (2025 Best Practices)."""
        try:
            import mlflow

            # Get current run
            run = mlflow.active_run()
            if not run:
                return

            # MLflow automatically creates these plots in the UI
            # We can also export them programmatically
            logger.info(f"📊 MLflow run {run.info.run_id} has built-in visualizations")
            logger.info(
                f"   View at: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}"
            )

            # Create a simple summary of available metrics
            client = mlflow.tracking.MlflowClient()
            metrics = client.get_metric_history(run.info.run_id, "val_loss")

            if metrics:
                summary_file = plots_dir / "mlflow_summary.txt"
                with open(summary_file, "w") as f:
                    f.write("MLflow Run Summary\n")
                    f.write(f"Run ID: {run.info.run_id}\n")
                    f.write(f"Experiment: {run.info.experiment_id}\n")
                    f.write(f"Status: {run.info.status}\n")
                    f.write(f"Validation Loss Metrics: {len(metrics)}\n")
                    f.write(f"Best Loss: {min(m.value for m in metrics):.4f}\n")
                    f.write(f"Last Loss: {metrics[-1].value:.4f}\n")
                    f.write(
                        f"\nView plots at: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}\n"
                    )
                    f.write("   - Training curves (loss, accuracy, learning rate)\n")
                    f.write("   - System metrics (CPU, GPU, memory)\n")
                    f.write("   - Parameter tracking and comparison\n")
                    f.write("   - Model artifacts and versions\n")

        except Exception as e:
            logger.warning(f"Could not create MLflow plots: {e}")

    def _create_optuna_plots(self, plots_dir: Path):
        """Create plots using Optuna's built-in visualization functions (2025 Best Practices)."""
        try:
            import optuna

            study = getattr(self, "_optuna_study", None)
            if not study:
                return

            # Create Optuna plots using their built-in functions (2025 Best Practices)
            # These are interactive HTML plots that can be viewed in browser

            # 1. Optimization History - shows how the objective value improved over trials
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_html(str(plots_dir / "optuna_optimization_history.html"))
            fig1.write_image(str(plots_dir / "optuna_optimization_history.png"))

            # 2. Parameter Importances - shows which hyperparameters are most important
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html(str(plots_dir / "optuna_param_importances.html"))
            fig2.write_image(str(plots_dir / "optuna_param_importances.png"))

            # 3. Parallel Coordinate Plot - shows relationships between parameters
            fig3 = optuna.visualization.plot_parallel_coordinate(study)
            fig3.write_html(str(plots_dir / "optuna_parallel_coordinate.html"))

            # 4. Contour Plot - shows 2D parameter relationships
            fig4 = optuna.visualization.plot_contour(study)
            fig4.write_html(str(plots_dir / "optuna_contour.html"))

            # 5. Slice Plot - shows individual parameter distributions
            fig5 = optuna.visualization.plot_slice(study)
            fig5.write_html(str(plots_dir / "optuna_slice.html"))

            # 6. Timeline Plot - shows trial duration and timing
            fig6 = optuna.visualization.plot_timeline(study)
            fig6.write_html(str(plots_dir / "optuna_timeline.html"))

            # Create a summary file with instructions
            summary_file = plots_dir / "optuna_visualization_guide.md"
            with open(summary_file, "w") as f:
                f.write("# Optuna Visualization Guide\n\n")
                f.write(f"Study Name: {study.study_name}\n")
                f.write(f"Number of Trials: {len(study.trials)}\n")
                f.write(f"Best Value: {study.best_value:.4f}\n")
                f.write(f"Best Parameters: {study.best_params}\n\n")

                f.write("## Available Plots\n\n")
                f.write(
                    "1. **optimization_history.html** - How objective value improved over trials\n"
                )
                f.write(
                    "2. **param_importances.html** - Which hyperparameters matter most\n"
                )
                f.write(
                    "3. **parallel_coordinate.html** - Parameter relationships and correlations\n"
                )
                f.write("4. **contour.html** - 2D parameter space visualization\n")
                f.write("5. **slice.html** - Individual parameter distributions\n")
                f.write("6. **timeline.html** - Trial timing and duration analysis\n\n")

                f.write("## How to View\n\n")
                f.write(
                    "Open any `.html` file in your web browser for interactive visualizations.\n"
                )
                f.write(
                    "The plots are interactive - you can zoom, hover, and explore the data.\n\n"
                )

                f.write("## Best Practices (2025)\n\n")
                f.write("- Use HTML plots for detailed analysis (interactive)\n")
                f.write("- Use PNG plots for reports and documentation\n")
                f.write("- Parameter importance helps focus optimization efforts\n")
                f.write("- Parallel coordinate plots reveal parameter interactions\n")
                f.write("- Timeline plots help identify performance bottlenecks\n")

            logger.info(f"📊 Created {len(study.trials)} Optuna visualization plots")
            logger.info("   HTML plots: Open in browser for interactive analysis")
            logger.info("   PNG plots: Use for reports and documentation")

        except Exception as e:
            logger.warning(f"Could not create Optuna plots: {e}")

    def _create_results_summary(self, results_dir: Path):
        """Create comprehensive results summary."""
        try:
            from datetime import datetime

            summary_file = results_dir / "training_summary.md"

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"# Training Summary: {self.experiment_name}\n\n")
                f.write(
                    f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"**Survey:** {getattr(self, 'survey', 'unknown')}\n\n")

                # Model information
                f.write("## Model Information\n\n")
                f.write(f"- **Model Type:** {type(self.astro_module.model).__name__}\n")
                f.write(
                    f"- **Task Type:** {getattr(self.astro_module, 'task_type', 'unknown')}\n"
                )
                f.write(
                    f"- **Number of Classes:** {getattr(self.astro_module, 'num_classes', 'N/A')}\n"
                )
                f.write(
                    f"- **Parameters:** {sum(p.numel() for p in self.astro_module.model.parameters()):,}\n"
                )
                f.write(
                    f"- **Trainable Parameters:** {sum(p.numel() for p in self.astro_module.model.parameters() if p.requires_grad):,}\n\n"
                )

                # Training configuration
                f.write("## Training Configuration\n\n")
                if self.training_config:
                    f.write(
                        f"- **Max Epochs:** {self.training_config.scheduler.max_epochs}\n"
                    )
                    f.write(
                        f"- **Learning Rate:** {self.training_config.scheduler.learning_rate}\n"
                    )
                    f.write(
                        f"- **Batch Size:** {getattr(self.training_config.data, 'batch_size', 'N/A')}\n"
                    )
                    f.write(
                        f"- **Hardware:** {self.training_config.hardware.accelerator}\n"
                    )
                    f.write(
                        f"- **Precision:** {self.training_config.hardware.precision}\n\n"
                    )

                # Results
                f.write("## Results\n\n")
                f.write(f"- **Best Model Path:** {self.best_model_path or 'N/A'}\n")
                f.write(f"- **Last Model Path:** {self.last_model_path or 'N/A'}\n")
                f.write(f"- **Checkpoint Directory:** {self.checkpoint_dir}\n\n")

                # MLflow information
                f.write("## MLflow Integration\n\n")
                f.write(f"- **Experiment Name:** {self.experiment_name}\n")
                f.write(
                    f"- **MLflow Enabled:** {getattr(self.training_config.logging, 'use_mlflow', True)}\n"
                )
                if hasattr(self, "logger") and self.logger:
                    f.write(f"- **Logger Type:** {type(self.logger).__name__}\n")

                # Directory structure
                f.write("\n## Directory Structure\n\n")
                f.write("```\n")
                f.write(f"{results_dir}/\n")
                f.write("├── models/          # Saved best models\n")
                f.write("├── plots/           # Training visualizations\n")
                f.write("│   ├── training_curves.png\n")
                f.write("│   ├── model_architecture.png\n")
                f.write("│   ├── confusion_matrix.png\n")
                f.write("│   └── feature_importance.png\n")
                f.write("└── training_summary.md  # This file\n")
                f.write("```\n")

            logger.info(f"📝 Training summary created: {summary_file}")

        except Exception as e:
            logger.error(f"Failed to create results summary: {e}")

    def test(
        self,
        test_dataloader: Optional[DataLoader] = None,
        datamodule: Any = None,
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
            logger.error(f"Testing failed: {e}")
            self._cleanup_after_training()
            raise

    def predict(
        self,
        predict_dataloader: Optional[DataLoader] = None,
        datamodule: Any = None,
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
            logger.error(f"Prediction failed: {e}")
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
        # Return the best model path from the ModelCheckpoint callback if available
        cb = getattr(self, "checkpoint_callback", None)
        if cb is not None and hasattr(cb, "best_model_path"):
            return getattr(cb, "best_model_path", None)
        # Fallback: try to find latest .ckpt file in checkpoint_dir
        if hasattr(self, "checkpoint_dir") and self.checkpoint_dir:
            ckpts = list(self.checkpoint_dir.glob("*.ckpt"))
            if ckpts:
                return str(
                    sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                )
        return None

    @property
    def last_model_path(self) -> Optional[str]:
        # Return the last model path from the ModelCheckpoint callback if available
        cb = getattr(self, "checkpoint_callback", None)
        if cb is not None and hasattr(cb, "last_model_path"):
            return getattr(cb, "last_model_path", None)
        # Fallback: try to find last .ckpt file in checkpoint_dir
        if hasattr(self, "checkpoint_dir") and self.checkpoint_dir:
            ckpts = list(self.checkpoint_dir.glob("*.ckpt"))
            if ckpts:
                return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])
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
        monitor: str = "val_loss",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna directly in AstroTrainer.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            search_space: Custom hyperparameter search space
            monitor: Metric to optimize
            **kwargs: Additional arguments

        Returns:
            Dictionary with best parameters and results
        """
        # Optuna direction and pruner are valid for create_study, but use kwargs to avoid linter errors
        study_kwargs = dict(
            direction="minimize" if "loss" in monitor else "maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,  # Let more trials complete first
                n_warmup_steps=5,  # Wait longer before pruning
            ),
        )
        study = optuna.create_study(**study_kwargs)

        def objective(trial):
            # Suggest hyperparameters
            if search_space:
                # Use custom search space from config
                params = {}
                for name, config in search_space.items():
                    if config["type"] == "float" or config["type"] == "uniform":
                        params[name] = trial.suggest_float(
                            name, config["low"], config["high"]
                        )
                    elif config["type"] == "loguniform":
                        params[name] = trial.suggest_float(
                            name, config["low"], config["high"], log=True
                        )
                    elif config["type"] == "int":
                        params[name] = trial.suggest_int(
                            name, config["low"], config["high"]
                        )
                    elif config["type"] == "categorical":
                        params[name] = trial.suggest_categorical(
                            name, config["choices"]
                        )
                    else:
                        logger.warning(
                            f"Unknown parameter type '{config['type']}' for {name}, skipping"
                        )

                logger.debug(f"Trial {trial.number} params from config: {params}")
            else:
                # Default search space
                params = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-2, log=True
                    ),
                    "hidden_dim": trial.suggest_int("hidden_dim", 64, 512),
                    "num_layers": trial.suggest_int("num_layers", 2, 6),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 1e-6, 1e-3, log=True
                    ),
                }

                logger.debug(f"Trial {trial.number} params from default: {params}")

            # Create model with trial parameters
            model_config = self.astro_module.model_config
            if model_config:
                # Update existing config
                model_config.hidden_dim = params.get(
                    "hidden_dim", model_config.hidden_dim
                )
                model_config.num_layers = params.get(
                    "num_layers", model_config.num_layers
                )
                model_config.dropout = params.get("dropout", model_config.dropout)

                # Create new model
                model = self.astro_module._create_model_from_config(model_config)
            else:
                # Create from scratch using new import path
                from astro_lab.models.core.survey_gnn import AstroSurveyGNN

                # Defensive: ensure input_dim and output_dim are integers
                input_dim = getattr(self.astro_module.model, "input_dim", None)
                if (
                    input_dim is not None
                    and not isinstance(input_dim, int)
                    and hasattr(input_dim, "item")
                ):
                    input_dim = int(input_dim.item())
                output_dim = getattr(self.astro_module.model, "output_dim", 1)
                if (
                    output_dim is not None
                    and not isinstance(output_dim, int)
                    and hasattr(output_dim, "item")
                ):
                    output_dim = int(output_dim.item())
                model = AstroSurveyGNN(
                    input_dim=input_dim,
                    hidden_dim=params.get("hidden_dim", 128),
                    output_dim=output_dim,
                    num_layers=params.get("num_layers", 3),
                    dropout=params.get("dropout", 0.2),
                    conv_type="gcn",
                )

            # Create new Lightning module for this trial
            trial_module = AstroLightningModule(
                model=model,
                task_type=self.astro_module.task_type,
                learning_rate=params.get("learning_rate", 1e-3),
                weight_decay=params.get("weight_decay", 1e-4),
                num_classes=self.astro_module.num_classes,
            )

            # Create trial trainer with minimal callbacks
            trial_trainer = Trainer(
                max_epochs=50,  # Longer for better optimization
                accelerator=self.accelerator,
                devices=1,  # Single device for optimization
                enable_checkpointing=False,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(
                        monitor=monitor,
                        patience=8,
                        mode="min" if "loss" in monitor else "max",
                    ),
                    PyTorchLightningPruningCallback(trial, monitor=monitor),
                ],
                logger=False,  # Disable logging during optimization
            )

            # Train and evaluate
            try:
                trial_trainer.fit(trial_module, train_dataloader, val_dataloader)

                # Get validation metric
                val_metric = trial_trainer.callback_metrics.get(monitor, float("inf"))
                # Always return a float for Optuna compatibility
                if not isinstance(val_metric, (float, int)) and hasattr(
                    val_metric, "item"
                ):
                    return float(val_metric.item())
                return float(val_metric)

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float("inf") if "loss" in monitor else float("-inf")

        # Run optimization
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Store the study for later plotting
        self._optuna_study = study

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        logger.info("Optimization complete!")
        logger.info(f"Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Return results
        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "study": study,
        }

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
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {checkpoint.name}: {e}")

    def save_best_models_to_results(self, top_k: int = 3) -> Dict[str, Path]:
        """
        Save best models to results directory with organized structure.

        Args:
            top_k: Number of best models to save

        Returns:
            Dictionary mapping model names to saved paths
        """
        try:
            # Use the survey name extracted during fit()
            survey = getattr(self, "survey", "unknown")

            # Extract model name from training config
            model_name = "unknown"
            if self.training_config and hasattr(self.training_config, "model"):
                model_name = getattr(self.training_config.model, "name", "unknown")

            # Create organized results structure: results/survey/model/
            results_structure = data_config.ensure_results_directories(
                survey, model_name
            )

            models_dir = results_structure["models"]

            # Get all checkpoints
            # Use data_config to get the correct checkpoint directory
            data_config.ensure_experiment_directories(self.experiment_name)
            exp_paths = data_config.get_experiment_paths(self.experiment_name)
            checkpoint_dir = exp_paths["checkpoints"]

            if not checkpoint_dir.exists():
                logger.error(f"No checkpoint directory found: {checkpoint_dir}")
                return {}

            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                logger.error(f"No checkpoints found in: {checkpoint_dir}")
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
                except Exception as e:
                    logger.error(f"Failed to save model {i + 1}: {e}")

            logger.info(f"Saved {len(saved_models)} models to: {models_dir}")
            return saved_models

        except Exception as e:
            logger.error(f"Failed to save models to results: {e}")
            return {}

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

    @property
    def lightning_module(self) -> Optional[AstroLightningModule]:
        """Get the lightning module."""
        return self.astro_module


__all__ = ["AstroTrainer"]
