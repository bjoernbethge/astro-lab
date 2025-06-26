"""
Lightning MLflow Logger for AstroLab
===================================

A streamlined MLflow logger specifically for PyTorch Lightning training
with astronomical models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import mlflow.pytorch
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only

logger = logging.getLogger(__name__)


class LightningMLflowLogger(MLFlowLogger):
    """
    Enhanced MLflow logger for PyTorch Lightning with AstroLab integration.
    
    Features:
    - Automatic experiment organization
    - Model architecture tracking
    - Configuration persistence
    - Enhanced artifact logging
    - Integration with AstroLab model factories
    """
    
    def __init__(
        self,
        experiment_name: str = "astrolab_experiment",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        save_dir: Optional[str] = "./mlruns",
        log_model: bool = True,
        prefix: str = "",
    ):
        """
        Initialize Lightning MLflow logger.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
            run_name: Name for this specific run
            tags: Additional tags for the run
            save_dir: Directory to save MLflow runs
            log_model: Whether to log models automatically
            prefix: Prefix for metric keys
        """
        # Set tracking URI with proper file:// prefix
        if tracking_uri is None:
            save_path = Path(save_dir).resolve()
            # Convert to proper file URI format
            tracking_uri = save_path.as_uri()
        
        # Prepare tags with defaults
        default_tags = {
            "framework": "lightning",
            "library": "astrolab",
            "created_at": datetime.now().isoformat(),
        }
        if tags:
            default_tags.update(tags)
        
        # Initialize parent MLFlowLogger
        super().__init__(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tags=default_tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            run_name=run_name,
        )
        
        self._log_model = log_model
        self._logged_model_summary = False
    
    @property
    def name(self) -> str:
        """Return logger name."""
        return "AstroLabMLflow"
    
    @property
    def version(self) -> str:
        """Return run id."""
        return self.run_id if self.run_id else "unknown"
    
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Any]) -> None:
        """
        Log hyperparameters with enhanced handling for AstroLab models.
        
        Args:
            params: Parameters to log (dict or namespace)
        """
        # Convert to dict if needed
        if hasattr(params, "hparams"):
            # Lightning module with hparams
            params_dict = dict(params.hparams)
        elif hasattr(params, "model_dump"):
            # Pydantic model
            params_dict = params.model_dump()
        elif hasattr(params, "__dict__"):
            # Object with attributes
            params_dict = {k: v for k, v in params.__dict__.items() 
                          if not k.startswith("_")}
        else:
            params_dict = dict(params) if isinstance(params, dict) else {}
        
        # Handle AstroLab specific parameters
        if "model_config" in params_dict:
            model_config = params_dict.pop("model_config")
            if hasattr(model_config, "to_dict"):
                params_dict["model"] = model_config.to_dict()
            else:
                params_dict["model"] = str(model_config)
        
        # Flatten nested parameters
        flat_params = self._flatten_dict(params_dict)
        
        # Filter out non-loggable values
        loggable_params = {
            k: v for k, v in flat_params.items() 
            if v is not None and self._is_loggable_value(v)
        }
        
        # Log via parent method
        super().log_hyperparams(loggable_params)
    
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics with validation.
        
        Args:
            metrics: Dictionary of metrics
            step: Global step number
        """
        # Filter out non-numeric values
        numeric_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_metrics[k] = float(v)
            elif hasattr(v, "item"):  # Tensor
                numeric_metrics[k] = float(v.item())
        
        # Log via parent method
        super().log_metrics(numeric_metrics, step)
    
    @rank_zero_only
    def log_model_summary(self, model: torch.nn.Module, max_depth: int = 2) -> None:
        """
        Log model architecture summary.
        
        Args:
            model: PyTorch model
            max_depth: Maximum depth for summary
        """
        if self._logged_model_summary:
            return
            
        try:
            from lightning.pytorch.utilities.model_summary import ModelSummary
            
            # Generate summary
            summary = ModelSummary(model, max_depth=max_depth)
            summary_str = str(summary)
            
            # Log as text artifact
            self.log_text(summary_str, "model_summary.txt")
            
            # Log key statistics as metrics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.experiment.log_metric(self.run_id, "model/total_params", total_params)
            self.experiment.log_metric(self.run_id, "model/trainable_params", trainable_params)
            self.experiment.log_metric(self.run_id, "model/size_mb", total_params * 4 / 1024 / 1024)
            
            # Log model type if it's an AstroLab model
            model_type = type(model).__name__
            self.experiment.set_tag(self.run_id, "model_type", model_type)
            
            self._logged_model_summary = True
            
        except Exception as e:
            logger.warning(f"Could not log model summary: {e}")
    
    @rank_zero_only
    def log_configuration(self, config: Any, filename: str = "config.yaml") -> None:
        """
        Log configuration as artifact.
        
        Args:
            config: Configuration object
            filename: Name for config file
        """
        # Convert to dict
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump()
        elif hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dict__"):
            config_dict = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith("_")}
        else:
            config_dict = config if isinstance(config, dict) else {"config": str(config)}
        
        # Handle nested objects
        config_dict = self._make_serializable(config_dict)
        
        # Save based on file extension
        config_path = Path(filename)
        
        try:
            if config_path.suffix in [".yaml", ".yml"]:
                import yaml
                with open(config_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
            
            # Log artifact
            self.experiment.log_artifact(self.run_id, str(config_path))
            config_path.unlink()
            
        except Exception as e:
            logger.warning(f"Could not log configuration: {e}")
    
    @rank_zero_only
    def log_text(self, text: str, filename: str) -> None:
        """
        Log text as artifact.
        
        Args:
            text: Text content
            filename: Filename for artifact
        """
        text_path = Path(filename)
        text_path.write_text(text)
        self.experiment.log_artifact(self.run_id, str(text_path))
        text_path.unlink()
    
    @rank_zero_only
    def log_astrolab_model_info(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """
        Log AstroLab specific model information.
        
        Args:
            model_name: Name of the model from factory
            model_config: Model configuration dictionary
        """
        # Log model factory name
        self.experiment.set_tag(self.run_id, "astrolab/model_name", model_name)
        
        # Log model configuration
        for key, value in model_config.items():
            if self._is_loggable_value(value):
                self.experiment.log_param(self.run_id, f"astrolab/{key}", value)
    
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """
        Finalize the experiment run.
        
        Args:
            status: Status of the run
        """
        try:
            if self.experiment:
                mlflow.set_tag("status", status)
                mlflow.set_tag("ended_at", datetime.now().isoformat())
        except Exception as e:
            logger.warning(f"Could not finalize run: {e}")
        
        try:
            # Call parent finalize but catch checkpoint scanning errors
            super().finalize(status)
        except Exception as e:
            logger.warning(f"Error during checkpoint scanning: {e}")
            # Continue without failing the entire training
    
    # Helper methods
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _is_loggable_value(self, value: Any) -> bool:
        """Check if value can be logged to MLflow."""
        if isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, (list, tuple)):
            return len(str(value)) < 500  # Avoid very long lists
        return False
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return str(obj)
        elif isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return str(obj)
        else:
            return obj


class MLflowModelCheckpoint(Callback):
    """
    Lightning callback to log best models to MLflow.
    
    This callback monitors a metric and saves the best model(s) to MLflow
    when improvements are detected.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        artifact_path: str = "best_model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Initialize MLflow model checkpoint callback.
        
        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Number of best models to save
            artifact_path: Path in MLflow artifacts
            registered_model_name: Name for model registry
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.artifact_path = artifact_path
        self.registered_model_name = registered_model_name
        
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_scores = []
    
    def on_validation_end(self, trainer, pl_module):
        """Check and save best model after validation."""
        if self.monitor not in trainer.callback_metrics:
            logger.warning(f"Metric '{self.monitor}' not found in callback metrics")
            return
        
        current = trainer.callback_metrics[self.monitor].item()
        
        # Check if this is a new best
        is_best = (
            (self.mode == "min" and current < self.best_score) or
            (self.mode == "max" and current > self.best_score)
        )
        
        if is_best:
            self.best_score = current
            
            # Log model to MLflow
            if hasattr(trainer.logger, "experiment"):
                try:
                    mlflow.pytorch.log_model(
                        pl_module,
                        artifact_path=self.artifact_path,
                        registered_model_name=self.registered_model_name
                    )
                    
                    # Log best score
                    trainer.logger.log_metrics({
                        f"best_{self.monitor}": current
                    })
                    
                    # Log checkpoint info
                    checkpoint_info = {
                        "epoch": trainer.current_epoch,
                        "global_step": trainer.global_step,
                        "metric": self.monitor,
                        "score": current,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    info_path = Path("best_model_info.json")
                    with open(info_path, "w") as f:
                        json.dump(checkpoint_info, f, indent=2)
                    
                    mlflow.log_artifact(str(info_path))
                    info_path.unlink()
                    
                    logger.info(f"Saved best model with {self.monitor}={current:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to log model to MLflow: {e}")
