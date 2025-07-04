"""
Central Configuration for AstroLab
=================================

Single source of truth for all AstroLab configuration.
Config YAMLs are loaded from the top-level 'configs' directory.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

# Project root logic (for rare use, e.g. find_project_root())


def find_project_root():
    """Find the project root directory (where pyproject.toml is located)."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent.parent


# Config YAML loading


def load_yaml(filename):
    path = Path(__file__).parent.parent.parent / "configs" / filename
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


SURVEY_CONFIGS = {}
for section in [
    "data",
    "model",
    "training",
    "mlflow",
    "tasks",
    "hpo",
    "albpy",
    "surveys",
]:
    loaded = load_yaml(f"{section}.yaml")
    if loaded is not None:
        if section == "surveys":
            SURVEY_CONFIGS[section] = loaded
        else:
            SURVEY_CONFIGS[section] = loaded[section]

# Config getter functions


def get_config() -> Dict[str, Any]:
    """Get the merged configuration from all YAMLs."""
    return SURVEY_CONFIGS


def get_data_config() -> dict:
    return SURVEY_CONFIGS["data"]


def get_model_config() -> dict:
    return SURVEY_CONFIGS["model"]


def get_training_config() -> dict:
    return SURVEY_CONFIGS["training"]


def get_mlflow_config() -> dict:
    return SURVEY_CONFIGS["mlflow"]


def get_task_config(task: str) -> dict:
    return SURVEY_CONFIGS["tasks"][task]


def get_hpo_config() -> dict:
    return SURVEY_CONFIGS["hpo"]


def get_albpy_config() -> dict:
    return SURVEY_CONFIGS["albpy"]


def get_survey_config(survey: str) -> dict:
    surveys = SURVEY_CONFIGS["surveys"]
    if survey not in surveys:
        raise ValueError(f"Survey '{survey}' not found in surveys config.")
    return surveys[survey]


def get_combined_config(survey: str, task: str) -> Dict[str, Any]:
    """Get combined configuration for survey and task."""
    data_config = get_survey_config(survey)
    model_config = get_task_config(task)
    training_config = get_training_config()
    mlflow_config = get_mlflow_config()
    return {
        **data_config,
        **model_config,
        **training_config,
        **mlflow_config,
    }


def get_checkpoint_filename(
    survey: str,
    task: str,
    epoch: int,
    metric_value: float,
    metric_name: str = "val_loss",
) -> str:
    """Generate checkpoint filename."""
    return f"{survey}_{task}_epoch{epoch}_{metric_name}{metric_value:.4f}"


def get_model_name(survey: str, task: str) -> str:
    """Generate model name."""
    return f"{survey}_{task}_model"


def get_run_name(survey: str, task: str, model_type: str = "astro_model") -> str:
    """Generate run name."""
    return f"{survey}_{task}_{model_type}"


def get_optimal_batch_size(gpu_memory_gb: Optional[float] = None) -> int:
    """Get optimal batch size based on GPU memory."""
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb:
        if gpu_memory_gb < 8:
            return 16
        elif gpu_memory_gb < 16:
            return 32
        else:
            return 64
    return 32  # Default


def get_data_paths() -> Dict[str, str]:
    """Get all relevant data paths from the YAML config."""
    data_cfg = get_data_config()
    mlflow_cfg = get_mlflow_config()
    return {
        "base_dir": data_cfg.get("base_dir", ""),
        "raw_dir": data_cfg.get("raw_dir", ""),
        "processed_dir": data_cfg.get("processed_dir", ""),
        "cache_dir": data_cfg.get("cache_dir", ""),
        "checkpoint_dir": data_cfg.get("checkpoint_dir", ""),
        "mlruns_dir": mlflow_cfg.get("tracking_uri", ""),
    }
