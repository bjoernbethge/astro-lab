"""
MLflow Integration for AstroLab Training

Enhanced MLflow logging with astronomical model artifacts and metrics.
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
    """Enhanced MLflow logger for astronomical models."""

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

        # Set additional astronomical tags
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

    def log_model_architecture(self, model: torch.nn.Module):
        """Log model architecture details."""
        try:
            # Model summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            architecture_info = {
                "model_class": model.__class__.__name__,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            }

            # Log architecture metrics
            for key, value in architecture_info.items():
                mlflow.log_metric(f"model_{key}", value)

            # Save architecture as artifact
            architecture_path = "model_architecture.json"
            with open(architecture_path, "w") as f:
                json.dump(architecture_info, f, indent=2)
            mlflow.log_artifact(architecture_path)
            os.remove(architecture_path)

        except Exception as e:
            print(f"Warning: Could not log model architecture: {e}")

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters with astronomical context."""
        # Separate model and training hyperparameters
        model_params = {}
        training_params = {}
        astro_params = {}

        for key, value in params.items():
            if key in ["hidden_dim", "num_layers", "conv_type", "dropout"]:
                model_params[key] = value
            elif key in ["learning_rate", "weight_decay", "batch_size", "scheduler"]:
                training_params[key] = value
            elif key in ["survey", "task_type", "use_photometry", "use_astrometry"]:
                astro_params[key] = value
            else:
                mlflow.log_param(key, value)

        # Log categorized parameters
        for category, param_dict in [
            ("model", model_params),
            ("training", training_params),
            ("astro", astro_params),
        ]:
            for key, value in param_dict.items():
                mlflow.log_param(f"{category}_{key}", value)

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        dataset_metrics = {
            "dataset_size": dataset_info.get("size", 0),
            "num_features": dataset_info.get("num_features", 0),
            "num_classes": dataset_info.get("num_classes", 0),
        }

        for key, value in dataset_metrics.items():
            if value > 0:
                mlflow.log_metric(key, value)

        # Log dataset metadata as artifact
        if dataset_info:
            dataset_path = "dataset_info.json"
            with open(dataset_path, "w") as f:
                json.dump(dataset_info, f, indent=2)
            mlflow.log_artifact(dataset_path)
            os.remove(dataset_path)

    def log_survey_info(self, survey: str, bands: Optional[list] = None):
        """Log astronomical survey information."""
        mlflow.set_tag("survey", survey)

        if bands:
            mlflow.set_tag("bands", ",".join(bands))
            mlflow.log_metric("num_bands", len(bands))

    def log_final_model(self, model: torch.nn.Module, model_name: str = "astro_model"):
        """Log final trained model."""
        try:
            # Log model with custom metadata
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                extra_files=["model_architecture.json"]
                if os.path.exists("model_architecture.json")
                else None,
            )

            # Register model if in production mode
            if os.getenv("MLFLOW_REGISTER_MODEL", "false").lower() == "true":
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
                mlflow.register_model(
                    model_uri=model_uri,
                    name=f"astrolab_{model.__class__.__name__.lower()}",
                )

            return model_info

        except Exception as e:
            print(f"Warning: Could not log model: {e}")
            return None

    def log_predictions(self, predictions_file: str):
        """Log prediction results as artifacts."""
        if os.path.exists(predictions_file):
            mlflow.log_artifact(predictions_file, "predictions")

    def log_confusion_matrix(self, cm_path: str):
        """Log confusion matrix plot."""
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, "plots")

    def end_run(self):
        """End MLflow run with cleanup."""
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Error ending MLflow run: {e}")


def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """Setup MLflow experiment for AstroLab training."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


__all__ = ["AstroMLflowLogger", "setup_mlflow_experiment"]
