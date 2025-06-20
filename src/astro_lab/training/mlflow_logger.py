"""
MLflow Integration for AstroLab Training

Optimized MLflow logging for astronomical models with modern best practices.
Integrated with the new AstroLab configuration system and 2025 system metrics.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch
from lightning.pytorch.loggers import MLFlowLogger as LightningMLFlowLogger

from astro_lab.data.config import data_config

# Optional imports for enhanced system monitoring
try:
    import GPUtil
    import psutil

    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False
    print("Warning: psutil/GPUtil not available. System metrics will be limited.")


class AstroMLflowLogger(LightningMLFlowLogger):
    """Optimized MLflow logger for astronomical models with 2025 system metrics integration."""

    def __init__(
        self,
        experiment_name: str = "astro_experiments",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        artifact_location: Optional[str] = None,
        run_name: Optional[str] = None,
        enable_system_metrics: bool = True,
        system_metrics_interval: int = 30,  # seconds
        **kwargs,
    ):
        # Set default tracking URI
        if tracking_uri is None:
            # Try environment variable first, then use local file-based tracking
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri is None:
                # Fallback to file-based tracking using data_config
                data_config.ensure_experiment_directories(experiment_name)
                tracking_uri = f"file://{data_config.mlruns_dir.absolute()}"

        # Set environment variable for consistency
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

        # Setup MLflow experiment first
        setup_mlflow_experiment(experiment_name, tracking_uri, artifact_location)

        # Combine user tags with astro tags
        astro_tags = {
            "framework": "astro-lab",
            "domain": "astronomy",
            "version": "0.3.0",
            "data_config_version": "2.0",  # Track config system version
            "tracking_uri": tracking_uri,
            "system_metrics_enabled": str(
                enable_system_metrics and SYSTEM_MONITORING_AVAILABLE
            ),
        }
        if tags:
            astro_tags.update(tags)

        super().__init__(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tags=astro_tags,  # Pass tags to parent
            artifact_location=artifact_location,
            run_name=run_name,
            **kwargs,
        )

        # System metrics monitoring setup
        self.enable_system_metrics = (
            enable_system_metrics and SYSTEM_MONITORING_AVAILABLE
        )
        self.system_metrics_interval = system_metrics_interval
        self._system_metrics_thread = None
        self._stop_system_metrics = threading.Event()

    def start_system_metrics_logging(self) -> None:
        """Start automatic system metrics logging in background thread."""
        if not self.enable_system_metrics:
            return

        if self._system_metrics_thread and self._system_metrics_thread.is_alive():
            return  # Already running

        self._stop_system_metrics.clear()
        self._system_metrics_thread = threading.Thread(
            target=self._log_system_metrics_loop, daemon=True
        )
        self._system_metrics_thread.start()
        print(
            f"🖥️  Started system metrics logging (interval: {self.system_metrics_interval}s)"
        )

    def stop_system_metrics_logging(self) -> None:
        """Stop system metrics logging."""
        if self._system_metrics_thread and self._system_metrics_thread.is_alive():
            self._stop_system_metrics.set()
            self._system_metrics_thread.join(timeout=5)
            print("🖥️  Stopped system metrics logging")

    def _log_system_metrics_loop(self) -> None:
        """Background loop for logging system metrics."""
        while not self._stop_system_metrics.wait(self.system_metrics_interval):
            try:
                self._log_system_metrics_snapshot()
            except Exception as e:
                print(f"Warning: Failed to log system metrics: {e}")

    def _log_system_metrics_snapshot(self) -> None:
        """Log a single snapshot of system metrics using 2025 best practices."""
        try:
            timestamp = time.time()

            # CPU Metrics (using slash notation for grouping)
            cpu_percent = psutil.cpu_percent(interval=1)
            mlflow.log_metric(
                "system/cpu/utilization_percent",
                float(cpu_percent),
                step=int(timestamp),
            )

            # Memory Metrics
            memory = psutil.virtual_memory()
            mlflow.log_metric(
                "system/memory/used_gb",
                float(memory.used / (1024**3)),
                step=int(timestamp),
            )
            mlflow.log_metric(
                "system/memory/available_gb",
                float(memory.available / (1024**3)),
                step=int(timestamp),
            )
            mlflow.log_metric(
                "system/memory/utilization_percent",
                float(memory.percent),
                step=int(timestamp),
            )

            # Disk Metrics
            disk = psutil.disk_usage("/")
            mlflow.log_metric(
                "system/disk/used_gb", float(disk.used / (1024**3)), step=int(timestamp)
            )
            mlflow.log_metric(
                "system/disk/free_gb", float(disk.free / (1024**3)), step=int(timestamp)
            )
            mlflow.log_metric(
                "system/disk/utilization_percent",
                float((disk.used / disk.total) * 100),
                step=int(timestamp),
            )

            # Network I/O
            net_io = psutil.net_io_counters()
            if (
                net_io
                and hasattr(net_io, "bytes_sent")
                and hasattr(net_io, "bytes_recv")
            ):
                mlflow.log_metric(
                    "system/network/bytes_sent_mb",
                    float(getattr(net_io, "bytes_sent", 0) / (1024**2)),
                    step=int(timestamp),
                )
                mlflow.log_metric(
                    "system/network/bytes_recv_mb",
                    float(getattr(net_io, "bytes_recv", 0) / (1024**2)),
                    step=int(timestamp),
                )

            # GPU Metrics (if available)
            if torch.cuda.is_available():
                try:
                    # PyTorch GPU metrics
                    for i in range(torch.cuda.device_count()):
                        # Memory usage
                        memory_allocated = torch.cuda.memory_allocated(i) / (
                            1024**3
                        )  # GB
                        memory_reserved = torch.cuda.memory_reserved(i) / (
                            1024**3
                        )  # GB
                        max_memory = torch.cuda.max_memory_allocated(i) / (
                            1024**3
                        )  # GB

                        mlflow.log_metric(
                            f"system/gpu_{i}/memory_allocated_gb",
                            memory_allocated,
                            step=int(timestamp),
                        )
                        mlflow.log_metric(
                            f"system/gpu_{i}/memory_reserved_gb",
                            memory_reserved,
                            step=int(timestamp),
                        )
                        mlflow.log_metric(
                            f"system/gpu_{i}/max_memory_gb",
                            max_memory,
                            step=int(timestamp),
                        )

                        # GPU utilization (if GPUtil available)
                        if "GPUtil" in globals():
                            gpus = GPUtil.getGPUs()
                            if i < len(gpus):
                                gpu = gpus[i]
                                mlflow.log_metric(
                                    f"system/gpu_{i}/utilization_percent",
                                    gpu.load * 100,
                                    step=int(timestamp),
                                )
                                mlflow.log_metric(
                                    f"system/gpu_{i}/temperature_c",
                                    gpu.temperature,
                                    step=int(timestamp),
                                )
                                mlflow.log_metric(
                                    f"system/gpu_{i}/memory_utilization_percent",
                                    gpu.memoryUtil * 100,
                                    step=int(timestamp),
                                )
                except Exception as gpu_error:
                    print(f"Warning: GPU metrics logging failed: {gpu_error}")

        except Exception as e:
            print(f"Warning: System metrics snapshot failed: {e}")

    def log_training_environment(self) -> None:
        """Log comprehensive training environment information."""
        try:
            env_info = {
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "platform": psutil.os.name,
            }

            # GPU Information
            if torch.cuda.is_available():
                env_info.update(
                    {
                        "cuda_available": True,
                        "cuda_version": "unknown",
                        "gpu_count": torch.cuda.device_count(),
                        "gpu_names": [
                            torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())
                        ],
                    }
                )
            else:
                env_info["cuda_available"] = False

            # Log as parameters
            for key, value in env_info.items():
                if isinstance(value, list):
                    mlflow.log_param(f"env_{key}", ", ".join(map(str, value)))
                else:
                    mlflow.log_param(f"env_{key}", value)

            # Save detailed environment info
            env_file = "training_environment.json"
            with open(env_file, "w") as f:
                json.dump(env_info, f, indent=2, default=str)

            mlflow.log_artifact(env_file, "environment")
            Path(env_file).unlink()  # Cleanup

        except Exception as e:
            print(f"Warning: Could not log training environment: {e}")

    def log_model_architecture(self, model: torch.nn.Module) -> None:
        """Log model architecture with optimized metrics."""
        try:
            # Calculate model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
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
            "optimization": [
                "precision",
                "gradient_clip_val",
                "accumulate_grad_batches",
            ],
            "hardware": ["accelerator", "devices", "enable_swa"],
            "data": ["survey", "k_neighbors", "distance_threshold", "num_features"],
        }

        # Log categorized parameters
        for category, param_names in categories.items():
            for param_name in param_names:
                if param_name in params:
                    mlflow.log_param(f"{category}_{param_name}", params[param_name])

        # Log remaining parameters
        logged_params = {
            param for param_list in categories.values() for param in param_list
        }
        for key, value in params.items():
            if key not in logged_params:
                mlflow.log_param(key, value)

    def log_config_info(self, config: Dict[str, Any]) -> None:
        """Log configuration information from the new config system."""
        try:
            # Log data paths
            mlflow.log_param("data_base_dir", str(data_config.base_dir))
            mlflow.log_param("mlruns_dir", str(data_config.mlruns_dir))
            mlflow.log_param("checkpoints_dir", str(data_config.checkpoints_dir))

            # Log config sections
            if "training" in config:
                training_config = config["training"]
                mlflow.log_param("max_epochs", training_config.get("max_epochs"))
                mlflow.log_param("batch_size", training_config.get("batch_size"))
                mlflow.log_param("learning_rate", training_config.get("learning_rate"))
                mlflow.log_param("accelerator", training_config.get("accelerator"))
                mlflow.log_param("precision", training_config.get("precision"))

            if "model" in config:
                model_config = config["model"]
                mlflow.log_param("model_type", model_config.get("type"))
                mlflow.log_param("hidden_dim", model_config.get("hidden_dim"))
                mlflow.log_param("num_layers", model_config.get("num_layers"))
                mlflow.log_param("dropout", model_config.get("dropout"))

            # Save full config as artifact
            config_file = "experiment_config.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2, default=str)

            mlflow.log_artifact(config_file, "config")
            Path(config_file).unlink()  # Cleanup

        except Exception as e:
            print(f"Warning: Could not log config info: {e}")

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
            json.dump(dataset_info, f, indent=2, default=str)

        mlflow.log_artifact(metadata_file, "data")
        Path(metadata_file).unlink()

    def log_survey_info(self, survey: str, bands: Optional[list] = None) -> None:
        """Log astronomical survey information with data config integration."""
        mlflow.set_tag("survey", survey)

        # Log survey-specific paths
        survey_raw_dir = data_config.get_survey_raw_dir(survey)
        survey_processed_dir = data_config.get_survey_processed_dir(survey)

        mlflow.log_param("survey_raw_dir", str(survey_raw_dir))
        mlflow.log_param("survey_processed_dir", str(survey_processed_dir))

        if bands:
            mlflow.set_tag("bands", ",".join(map(str, bands)))
            mlflow.log_metric("num_bands", len(bands))

    def log_final_model(
        self, model: torch.nn.Module, model_name: str = "astro_model"
    ) -> Optional[Any]:
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
                "pyyaml",  # For config loading
                {
                    "pip": [
                        "mlflow",
                        "optuna",
                        "astroquery",
                    ]
                },
            ],
        }

    def log_predictions(
        self, predictions_file: str, description: str = "Model predictions"
    ) -> None:
        """Log prediction results with description."""
        if Path(predictions_file).exists():
            mlflow.log_artifact(predictions_file, "predictions")
            mlflow.set_tag("predictions_logged", description)

    def log_visualization(self, plot_path: str, plot_type: str = "plot") -> None:
        """Log visualization plots."""
        if Path(plot_path).exists():
            mlflow.log_artifact(plot_path, f"plots/{plot_type}")

    def log_checkpoint_info(self, checkpoint_dir: Path) -> None:
        """Log checkpoint directory information."""
        mlflow.log_param("checkpoint_dir", str(checkpoint_dir))

        # Log checkpoint files if they exist
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            mlflow.log_metric("num_checkpoints", len(checkpoints))

            if checkpoints:
                # Log latest checkpoint info
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                mlflow.log_param("latest_checkpoint", latest_checkpoint.name)
                mlflow.log_metric(
                    "checkpoint_size_mb",
                    latest_checkpoint.stat().st_size / (1024 * 1024),
                )

    def end_run(self) -> None:
        """End MLflow run with cleanup."""
        try:
            # Stop system metrics logging
            self.stop_system_metrics_logging()

            # Log final run status
            mlflow.set_tag("run_status", "completed")
            mlflow.set_tag("data_config_base_dir", str(data_config.base_dir))
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Error ending MLflow run: {e}")


def create_astro_mlflow_logger(
    config: Dict[str, Any], experiment_name: Optional[str] = None
) -> AstroMLflowLogger:
    """Create AstroMLflowLogger from configuration dictionary."""
    mlflow_config = config.get("mlflow", {})

    # Use experiment name from config or parameter
    exp_name = experiment_name or mlflow_config.get(
        "experiment_name", "astro_experiment"
    )

    # Handle artifact location with data config integration
    artifact_location = mlflow_config.get("artifact_location")
    if artifact_location and "${data.base_dir}" in str(artifact_location):
        # Replace placeholder with actual path
        artifact_location = artifact_location.replace(
            "${data.base_dir}", str(data_config.base_dir)
        )

    logger = AstroMLflowLogger(
        experiment_name=exp_name,
        tracking_uri=mlflow_config.get("tracking_uri"),
        tags=mlflow_config.get("tags"),
        artifact_location=artifact_location,
        run_name=mlflow_config.get("run_name"),
        enable_system_metrics=mlflow_config.get("enable_system_metrics", True),
        system_metrics_interval=mlflow_config.get("system_metrics_interval", 30),
    )

    # Log configuration info
    logger.log_config_info(config)

    return logger


def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """Setup MLflow experiment with data config integration."""
    # Use data_config system if no tracking URI provided
    if tracking_uri is None:
        data_config.ensure_experiment_directories(experiment_name)
        tracking_uri = str(data_config.mlruns_dir)

    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags={
                "domain": "astronomy",
                "framework": "astrolab",
                "data_config_version": "2.0",
                "tracking_uri": tracking_uri,
            },
        )
        print(f"🧪 Created new MLflow experiment: {experiment_name}")
        print(f"   Tracking URI: {tracking_uri}")
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"🧪 Using existing MLflow experiment: {experiment_name}")
        else:
            raise ValueError(f"Could not create or find experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def setup_mlflow_from_config(config: Dict[str, Any]) -> str:
    """Setup MLflow experiment from configuration dictionary."""
    mlflow_config = config.get("mlflow", {})

    return setup_mlflow_experiment(
        experiment_name=mlflow_config.get("experiment_name", "astro_experiment"),
        tracking_uri=mlflow_config.get("tracking_uri"),
        artifact_location=mlflow_config.get("artifact_location"),
    )


__all__ = [
    "AstroMLflowLogger",
    "setup_mlflow_experiment",
    "setup_mlflow_from_config",
    "create_astro_mlflow_logger",
]
