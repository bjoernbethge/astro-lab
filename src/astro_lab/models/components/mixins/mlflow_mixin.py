"""
MLflow Integration Mixin
========================

Provides comprehensive MLflow tracking capabilities for models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


class MLflowMixin:
    """
    Mixin for comprehensive MLflow integration.
    
    Features:
    - Automatic parameter logging
    - Metric tracking with windowing
    - Model checkpointing
    - Artifact management
    - Auto-logging of system metrics
    - Experiment organization
    """
    
    def __init__(self):
        """Initialize MLflow tracking."""
        self._mlflow_run_id = None
        self._mlflow_experiment_id = None
        self._metric_history = {}
        self._logged_params = False
        self._auto_log_enabled = True
        
    def setup_mlflow(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> None:
        """
        Setup MLflow tracking for the model.
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name of the run
            tags: Additional tags for the run
            nested: Whether this is a nested run
        """
        if not self._auto_log_enabled:
            return
            
        try:
            # Set experiment
            if experiment_name:
                mlflow.set_experiment(experiment_name)
                self._mlflow_experiment_id = mlflow.get_experiment_by_name(
                    experiment_name
                ).experiment_id
            
            # Start or get active run
            if mlflow.active_run() is None or nested:
                run = mlflow.start_run(run_name=run_name, nested=nested)
                self._mlflow_run_id = run.info.run_id
            else:
                self._mlflow_run_id = mlflow.active_run().info.run_id
            
            # Set tags
            if tags:
                mlflow.set_tags(tags)
            
            # Enable autologging
            mlflow.pytorch.autolog(
                log_models=True,
                log_model_signatures=True,
                log_input_examples=True,
                disable=False,
                exclusive=False,
                registered_model_name=None,
                extra_tags=tags,
            )
            
            # Log system information
            self._log_system_info()
            
            logger.info(f"MLflow tracking initialized: {self._mlflow_run_id}")
            
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self._auto_log_enabled = False
    
    def log_hyperparameters(self, prefix: str = "") -> None:
        """
        Automatically log all hyperparameters.
        
        Args:
            prefix: Prefix for parameter names
        """
        if not self._auto_log_enabled or self._logged_params:
            return
            
        try:
            # Get hyperparameters from LightningModule
            if hasattr(self, "hparams"):
                params = dict(self.hparams)
            else:
                # Fallback to manual extraction
                params = self._extract_hyperparameters()
            
            # Add prefix if provided
            if prefix:
                params = {f"{prefix}_{k}": v for k, v in params.items()}
            
            # Log to MLflow
            mlflow.log_params(params)
            self._logged_params = True
            
            # Log model architecture summary
            self._log_model_architecture()
            
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        window_size: int = 100,
    ) -> None:
        """
        Log metrics with windowing for large-scale training.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            window_size: Window size for averaging
        """
        if not self._auto_log_enabled:
            return
            
        try:
            # Update metric history
            for name, value in metrics.items():
                if name not in self._metric_history:
                    self._metric_history[name] = []
                self._metric_history[name].append(value)
                
                # Keep only window_size values
                if len(self._metric_history[name]) > window_size:
                    self._metric_history[name].pop(0)
                
                # Log current value
                mlflow.log_metric(name, value, step=step)
                
                # Log windowed average
                if len(self._metric_history[name]) >= window_size:
                    avg_value = sum(self._metric_history[name]) / len(
                        self._metric_history[name]
                    )
                    mlflow.log_metric(f"{name}_avg_{window_size}", avg_value, step=step)
            
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_model_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        metrics: Optional[Dict[str, float]] = None,
        artifact_path: str = "checkpoints",
    ) -> None:
        """
        Log model checkpoint as MLflow artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            metrics: Associated metrics
            artifact_path: Artifact path in MLflow
        """
        if not self._auto_log_enabled:
            return
            
        try:
            # Log checkpoint file
            mlflow.log_artifact(str(checkpoint_path), artifact_path)
            
            # Log associated metrics
            if metrics:
                for name, value in metrics.items():
                    mlflow.log_metric(f"checkpoint_{name}", value)
            
            # Log checkpoint metadata
            metadata = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_mb": Path(checkpoint_path).stat().st_size / 1e6,
            }
            mlflow.log_dict(metadata, f"{artifact_path}/checkpoint_metadata.json")
            
        except Exception as e:
            logger.warning(f"Failed to log checkpoint: {e}")
    
    def log_batch_statistics(
        self, batch: Union[Data, Batch], prefix: str = "batch"
    ) -> None:
        """
        Log statistics about the current batch.
        
        Args:
            batch: Current batch
            prefix: Prefix for metric names
        """
        if not self._auto_log_enabled:
            return
            
        try:
            stats = {}
            
            # Basic statistics
            if hasattr(batch, "num_nodes"):
                stats[f"{prefix}_num_nodes"] = batch.num_nodes
            if hasattr(batch, "num_edges"):
                stats[f"{prefix}_num_edges"] = batch.num_edges
            if hasattr(batch, "batch"):
                stats[f"{prefix}_num_graphs"] = batch.batch.max().item() + 1
            
            # Feature statistics
            if hasattr(batch, "x") and batch.x is not None:
                stats[f"{prefix}_feature_mean"] = batch.x.mean().item()
                stats[f"{prefix}_feature_std"] = batch.x.std().item()
            
            # Memory usage
            if torch.cuda.is_available():
                stats[f"{prefix}_gpu_memory_mb"] = (
                    torch.cuda.memory_allocated() / 1e6
                )
            
            # Log all statistics
            step = self.current_epoch if hasattr(self, "current_epoch") else None
            for name, value in stats.items():
                mlflow.log_metric(name, value, step=step)
            
        except Exception as e:
            logger.debug(f"Failed to log batch statistics: {e}")
    
    def _extract_hyperparameters(self) -> Dict[str, Any]:
        """Extract hyperparameters from model attributes."""
        params = {}
        
        # Common attributes to log
        param_attrs = [
            "learning_rate",
            "weight_decay",
            "hidden_dim",
            "num_layers",
            "dropout",
            "num_features",
            "num_classes",
            "task",
            "conv_type",
            "heads",
            "compile_model",
            "compile_mode",
        ]
        
        for attr in param_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                # Convert non-serializable types
                if isinstance(value, (torch.Tensor, nn.Module)):
                    continue
                params[attr] = value
        
        return params
    
    def _log_system_info(self) -> None:
        """Log system information."""
        try:
            import platform
            import psutil
            
            system_info = {
                "system_platform": platform.platform(),
                "system_python_version": platform.python_version(),
                "system_cpu_count": psutil.cpu_count(),
                "system_memory_gb": psutil.virtual_memory().total / 1e9,
            }
            
            if torch.cuda.is_available():
                system_info.update({
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                })
            
            mlflow.log_params(system_info)
            
        except Exception as e:
            logger.debug(f"Failed to log system info: {e}")
    
    def _log_model_architecture(self) -> None:
        """Log model architecture details."""
        try:
            if not isinstance(self, nn.Module):
                return
                
            # Count parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            
            mlflow.log_params({
                "model_total_params": total_params,
                "model_trainable_params": trainable_params,
                "model_class": self.__class__.__name__,
            })
            
            # Log model summary as text
            from io import StringIO
            buffer = StringIO()
            print(self, file=buffer)
            mlflow.log_text(buffer.getvalue(), "model_architecture.txt")
            
        except Exception as e:
            logger.debug(f"Failed to log model architecture: {e}")
    
    def log_predictions(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        sample_size: int = 100,
    ) -> None:
        """
        Log sample predictions for visualization.
        
        Args:
            predictions: Model predictions
            targets: True targets (optional)
            sample_size: Number of samples to log
        """
        if not self._auto_log_enabled:
            return
            
        try:
            import pandas as pd
            
            # Sample predictions
            n_samples = min(sample_size, len(predictions))
            indices = torch.randperm(len(predictions))[:n_samples]
            
            # Create dataframe
            data = {"prediction": predictions[indices].cpu().numpy()}
            if targets is not None:
                data["target"] = targets[indices].cpu().numpy()
            
            df = pd.DataFrame(data)
            
            # Log as artifact
            with mlflow.start_run(run_id=self._mlflow_run_id, nested=True):
                mlflow.log_table(df, "predictions/sample_predictions.json")
            
        except Exception as e:
            logger.debug(f"Failed to log predictions: {e}")
    
    def finalize_mlflow(self) -> None:
        """Finalize MLflow tracking."""
        if not self._auto_log_enabled:
            return
            
        try:
            # Log final model
            if isinstance(self, nn.Module):
                mlflow.pytorch.log_model(self, "model")
            
            # End run if we started it
            if self._mlflow_run_id and mlflow.active_run():
                if mlflow.active_run().info.run_id == self._mlflow_run_id:
                    mlflow.end_run()
            
        except Exception as e:
            logger.warning(f"Failed to finalize MLflow: {e}")
