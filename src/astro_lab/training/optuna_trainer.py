"""
AstroLab Optuna Trainer - Hyperparameter Optimization
====================================================

Advanced hyperparameter optimization using Optuna with MLflow integration.
Supports both single-objective and multi-objective optimization.
"""

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import mlflow
import optuna
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

from astro_lab.training.trainer import AstroTrainer
from astro_lab.training.lightning_module import AstroLightningModule

# MLflow integration (optional)
try:
    from lightning.pytorch.loggers import MLFlowLogger

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFlowLogger = None
    MLFLOW_AVAILABLE = False


class OptunaTrainer:
    """Optimized Optuna-based hyperparameter optimization with optional MLflow integration."""

    def __init__(
        self,
        model_factory: Callable,
        train_dataloader: Any,
        val_dataloader: Any,
        mlflow_experiment: str = "optuna_hyperparameter_tuning",
        study_name: Optional[str] = None,
        log_plots: bool = True,
        survey: str = "gaia",
        experiment_name: str = "optuna_optimization",
    ):
        """
        Initialize Optuna trainer.

        Args:
            model_factory: Function that creates model instances
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            mlflow_experiment: MLflow experiment name
            study_name: Optuna study name (optional)
            log_plots: Whether to log Optuna visualization plots
        """
        self.model_factory = model_factory
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.mlflow_experiment = mlflow_experiment
        self.study_name = study_name
        self.log_plots = log_plots
        self.survey = survey
        self.experiment_name = experiment_name

        # Setup organized results structure
        from ..data.config import data_config

        self.results_structure = data_config.ensure_results_directories(
            survey, experiment_name
        )

        # Create Optuna study with modern optimizations
        self.study = optuna.create_study()
        # Set pruner and sampler separately
        self.study.pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
        self.study.sampler = optuna.samplers.TPESampler()

        # Set up MLflow experiment if available
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.set_experiment(self.mlflow_experiment)

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        run_context = None

        if MLFLOW_AVAILABLE and mlflow is not None:
            run_context = mlflow.start_run(nested=True)
            # Log trial number
            mlflow.log_param("trial_number", trial.number)

        try:
            # Create model with trial suggestions
            model = self.model_factory(trial)

            # Wrap in Lightning module if needed
            from .lightning_module import AstroLightningModule

            if not isinstance(model, AstroLightningModule):
                model = AstroLightningModule(model=model)

            # Create trainer with optimized settings
            trainer = Trainer(
                max_epochs=50,  # Reduced for faster optimization
                enable_checkpointing=False,
                logger=False,  # Use MLflow manually for better control
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=5, verbose=False),
                    PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                ],
                enable_progress_bar=False,
                accelerator="auto",
                devices=1,  # Single device for optimization
                precision="16-mixed",  # Fast mixed precision
            )

            # Train model
            trainer.fit(model, self.train_dataloader, self.val_dataloader)

            # Get final validation loss
            val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
            val_loss_float = (
                float(val_loss) if hasattr(val_loss, "item") else float(val_loss)
            )

            # Log metrics and parameters if MLflow available
            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.log_metric("val_loss", val_loss_float)
                # Log hyperparameters
                for key, value in trial.params.items():
                    mlflow.log_param(key, value)

            return val_loss_float

        except optuna.TrialPruned:
            # Log pruned trial if MLflow available
            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.log_param("pruned", True)
            raise
        finally:
            if run_context is not None:
                mlflow.end_run()

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)

        Returns:
            Completed Optuna study
        """
        run_context = None

        if MLFLOW_AVAILABLE and mlflow is not None:
            run_context = mlflow.start_run(
                run_name=f"optuna_study_{self.study_name or 'default'}"
            )
            # Log study configuration
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("timeout", timeout)
            mlflow.log_param("study_name", self.study_name)

        try:
            # Run optimization
            self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

            # Log best results if MLflow available
            if MLFLOW_AVAILABLE and mlflow is not None:
                best_trial = self.study.best_trial
                if best_trial.value is not None:
                    mlflow.log_metric("best_val_loss", best_trial.value)

                # Log best parameters
                for key, value in best_trial.params.items():
                    mlflow.log_param(f"best_{key}", value)

                # Log study summary
                self._log_study_summary()

                # Log Optuna plots
                if self.log_plots:
                    self._log_optuna_plots()
                    # Also save to organized results structure
                    self._save_optuna_plots_to_results()

        finally:
            if run_context is not None:
                mlflow.end_run()

        return self.study

    def _log_study_summary(self):
        """Log study summary to MLflow."""
        if not MLFLOW_AVAILABLE or mlflow is None:
            return

        study_summary = {
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "n_completed_trials": len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
            ),
            "n_pruned_trials": len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ]
            ),
            "n_failed_trials": len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.FAIL
                ]
            ),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(study_summary, f, indent=2)
            f.flush()  # Ensure data is written
            mlflow.log_artifact(f.name, "study_summary.json")

        # Clean up with Windows-safe approach
        try:
            Path(f.name).unlink()
        except (PermissionError, FileNotFoundError):
            # File might be locked on Windows, skip cleanup
            pass

    def _log_optuna_plots(self):
        """Log Optuna visualization plots using built-in functions."""
        if not self.log_plots or len(self.study.trials) < 2:
            return

        if not MLFLOW_AVAILABLE or mlflow is None:
            return

        try:
            # Optimization history plot
            if len(self.study.trials) > 1:
                try:
                    fig = optuna.visualization.plot_optimization_history(self.study)
                    mlflow.log_figure(fig, "optuna_optimization_history.html")
                except Exception as e:
                    print(f"Could not create optimization history plot: {e}")

            # Parameter importances plot
            if len(self.study.trials) > 5:
                try:
                    fig = optuna.visualization.plot_param_importances(self.study)
                    mlflow.log_figure(fig, "optuna_param_importances.html")
                except Exception as e:
                    print(f"Could not create parameter importances plot: {e}")

            # Parallel coordinate plot
            if len(self.study.trials) > 5:
                try:
                    fig = optuna.visualization.plot_parallel_coordinate(self.study)
                    mlflow.log_figure(fig, "optuna_parallel_coordinate.html")
                except Exception as e:
                    print(f"Could not create parallel coordinate plot: {e}")

            # Slice plot
            if len(self.study.trials) > 5:
                try:
                    fig = optuna.visualization.plot_slice(self.study)
                    mlflow.log_figure(fig, "optuna_slice_plot.html")
                except Exception as e:
                    print(f"Could not create slice plot: {e}")

            # Contour plot (if 2+ parameters)
            if len(self.study.trials) > 10 and len(self.study.best_params) >= 2:
                try:
                    fig = optuna.visualization.plot_contour(self.study)
                    mlflow.log_figure(fig, "optuna_contour_plot.html")
                except Exception as e:
                    print(f"Could not create contour plot: {e}")

        except Exception as e:
            print(f"Warning: Could not create Optuna plots: {e}")

    def _save_optuna_plots_to_results(self):
        """Save Optuna plots directly to organized results structure."""
        if not self.log_plots or len(self.study.trials) < 2:
            return

        plots_dir = self.results_structure["optuna_plots"]

        try:
            # Save plots as HTML files directly to results
            # Optimization history plot
            if len(self.study.trials) > 1:
                try:
                    fig = optuna.visualization.plot_optimization_history(self.study)
                    fig.write_html(plots_dir / "optuna_optimization_history.html")
                except Exception as e:
                    print(f"Could not save optimization history plot: {e}")

            # Parameter importances plot
            if len(self.study.trials) > 5:
                try:
                    fig = optuna.visualization.plot_param_importances(self.study)
                    fig.write_html(plots_dir / "optuna_param_importances.html")
                except Exception as e:
                    print(f"Could not save parameter importances plot: {e}")

            # Parallel coordinate plot
            if len(self.study.trials) > 5:
                try:
                    fig = optuna.visualization.plot_parallel_coordinate(self.study)
                    fig.write_html(plots_dir / "optuna_parallel_coordinate.html")
                except Exception as e:
                    print(f"Could not save parallel coordinate plot: {e}")

            # Slice plot
            if len(self.study.trials) > 5:
                try:
                    fig = optuna.visualization.plot_slice(self.study)
                    fig.write_html(plots_dir / "optuna_slice_plot.html")
                except Exception as e:
                    print(f"Could not save slice plot: {e}")

            # Contour plot (if 2+ parameters)
            if len(self.study.trials) > 10 and len(self.study.best_params) >= 2:
                try:
                    fig = optuna.visualization.plot_contour(self.study)
                    fig.write_html(plots_dir / "optuna_contour_plot.html")
                except Exception as e:
                    print(f"Could not save contour plot: {e}")

            # Save study summary as JSON
            study_summary = {
                "survey": self.survey,
                "experiment": self.experiment_name,
                "n_trials": len(self.study.trials),
                "best_value": self.study.best_value,
                "best_params": self.study.best_params,
                "completed_trials": len(
                    [
                        t
                        for t in self.study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                ),
                "pruned_trials": len(
                    [
                        t
                        for t in self.study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                ),
            }

            with open(plots_dir / "study_summary.json", "w") as f:
                json.dump(study_summary, f, indent=2)

            print(f"📊 Saved Optuna plots to: {plots_dir}")

        except Exception as e:
            print(f"Warning: Could not save Optuna plots to results: {e}")

    def get_best_model(self) -> AstroLightningModule:
        """Create and return the best model found during optimization."""
        if not self.study.best_trial:
            raise ValueError("No trials completed. Run optimize() first.")

        # Create trial with best parameters
        best_trial = self.study.best_trial

        # Create a mock trial for the model factory
        class MockTrial:
            def __init__(self, params):
                self.params = params

            def suggest_float(self, name, low, high, **kwargs):
                return self.params.get(name, (low + high) / 2)

            def suggest_int(self, name, low, high, **kwargs):
                return self.params.get(name, (low + high) // 2)

            def suggest_categorical(self, name, choices, **kwargs):
                return self.params.get(name, choices[0])

        mock_trial = MockTrial(best_trial.params)
        return self.model_factory(mock_trial)

    def save_study(self, filepath: str):
        """Save the study to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.study, f)

        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_artifact(filepath, "optuna_study.pkl")

    def load_study(filepath: str) -> optuna.Study:
        """Load a study from a file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


__all__ = ["OptunaTrainer"]
