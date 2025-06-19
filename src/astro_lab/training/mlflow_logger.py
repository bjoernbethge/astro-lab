"""
MLflow Integration for AstroLab Training

Optimized MLflow logging for astronomical models with modern best practices.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch
from lightning.pytorch.loggers import MLFlowLogger as LightningMLFlowLogger


class AstroMLflowLogger(LightningMLFlowLogger):
    """Optimized MLflow logger for astronomical models."""

    def __init__(
        self,
        experiment_name: str = "astro_experiments",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_location: Optional[str] = None,
        run_name: Optional[str] = None,
        **kwargs,
    ):
        # Set default tracking URI
        if tracking_uri is None:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

        super().__init__(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tags=tags,
            artifact_location=artifact_location,
            run_name=run_name,
            **kwargs,
        )

        # Set astronomical tags
        astro_tags = {
            "framework": "astrolab",
            "domain": "astronomy",
            "version": "0.3.0",
        }
        if tags:
            astro_tags.update(tags)

        # Log tags
        for key, value in astro_tags.items():
            mlflow.set_tag(key, value)

    def log_model_architecture(self, model: torch.nn.Module) -> None:
        """Log model architecture with optimized metrics."""
        try:
            # Calculate model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

            architecture_info = {
                "model_class": model.__class__.__name__,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": round(model_size_mb, 2),
                "num_layers": len(list(model.named_modules())),
            }

            # Log as metrics for easy comparison
            for key, value in architecture_info.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"model_{key}", value)
                else:
                    mlflow.log_param(f"model_{key}", str(value))

            # Save detailed architecture
            arch_file = "model_architecture.json"
            with open(arch_file, "w") as f:
                json.dump(architecture_info, f, indent=2)
            
            mlflow.log_artifact(arch_file)
            Path(arch_file).unlink()  # Cleanup

        except Exception as e:
            print(f"Warning: Could not log model architecture: {e}")

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters with structured organization."""
        # Organize parameters by category
        categories = {
            "model": ["hidden_dim", "num_layers", "dropout", "num_classes"],
            "training": ["learning_rate", "weight_decay", "scheduler", "batch_size"],
            "optimization": ["precision", "gradient_clip_val", "accumulate_grad_batches"],
            "hardware": ["accelerator", "devices", "enable_swa"],
        }

        # Log categorized parameters
        for category, param_names in categories.items():
            for param_name in param_names:
                if param_name in params:
                    mlflow.log_param(f"{category}_{param_name}", params[param_name])

        # Log remaining parameters
        logged_params = {param for param_list in categories.values() for param in param_list}
        for key, value in params.items():
            if key not in logged_params:
                mlflow.log_param(key, value)

    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """Log dataset information efficiently."""
        if not dataset_info:
            return

        # Log key metrics
        metrics_to_log = ["dataset_size", "num_features", "num_classes", "num_graphs"]
        for metric in metrics_to_log:
            if metric in dataset_info:
                value = dataset_info[metric]
                if isinstance(value, (int, float)) and value > 0:
                    mlflow.log_metric(metric, value)

        # Log dataset metadata
        metadata_file = "dataset_info.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        mlflow.log_artifact(metadata_file)
        Path(metadata_file).unlink()

    def log_survey_info(self, survey: str, bands: Optional[list] = None) -> None:
        """Log astronomical survey information."""
        mlflow.set_tag("survey", survey)
        
        if bands:
            mlflow.set_tag("bands", ",".join(map(str, bands)))
            mlflow.log_metric("num_bands", len(bands))

    def log_final_model(self, model: torch.nn.Module, model_name: str = "astro_model") -> Optional[Any]:
        """Log final trained model with metadata."""
        try:
            # Log model with PyTorch Lightning integration
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                conda_env=self._get_conda_env(),
                code_paths=["src/astro_lab/"],  # Include source code
            )

            # Register model if in production mode
            if os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true":
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=f"astrolab_{model.__class__.__name__.lower()}",
                    tags={"domain": "astronomy", "framework": "astrolab"},
                )
                mlflow.log_param("registered_model_version", model_version.version)

            return model_info

        except Exception as e:
            print(f"Warning: Could not log model: {e}")
            return None

    def _get_conda_env(self) -> Dict[str, Any]:
        """Get conda environment for model deployment."""
        return {
            "channels": ["conda-forge", "pytorch", "pyg"],
            "dependencies": [
                "python=3.11",
                "pytorch",
                "torch-geometric",
                "lightning",
                "astropy",
                {
                    "pip": [
                        "mlflow",
                        "optuna",
                        "astroquery",
                    ]
                },
            ],
        }

    def log_predictions(self, predictions_file: str, description: str = "Model predictions") -> None:
        """Log prediction results with description."""
        if Path(predictions_file).exists():
            mlflow.log_artifact(predictions_file, "predictions")
            mlflow.set_tag("predictions_logged", description)

    def log_visualization(self, plot_path: str, plot_type: str = "plot") -> None:
        """Log visualization plots."""
        if Path(plot_path).exists():
            mlflow.log_artifact(plot_path, f"plots/{plot_type}")

    def end_run(self) -> None:
        """End MLflow run with cleanup."""
        try:
            # Log final run status
            mlflow.set_tag("run_status", "completed")
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Error ending MLflow run: {e}")


def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """Setup MLflow experiment with error handling."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags={"domain": "astronomy", "framework": "astrolab"},
        )
        print(f"Created new experiment: {experiment_name}")
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name}")
        else:
            raise ValueError(f"Could not create or find experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)
    return experiment_id


__all__ = ["AstroMLflowLogger", "setup_mlflow_experiment"]
