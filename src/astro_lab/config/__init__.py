"""
AstroLab Configuration Module
============================

Central configuration management for all AstroLab components.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .data import (
    DataConfig,
    data_config,
    get_data_config,
    list_survey_paths,
    setup_data_directories,
)
from .defaults import (
    DATA_DEFAULTS,
    HPO_SEARCH_SPACES,
    MODEL_DEFAULTS,
    SURVEY_TRAINING_DEFAULTS,
    TASK_DEFAULTS,
    TRAINING_DEFAULTS,
    get_hpo_search_space,
    get_training_config,
)
from .defaults import (
    get_data_config as get_data_defaults,
)
from .model import (
    AstroGraphGNNConfig,
    AstroNodeGNNConfig,
    AstroPointNetConfig,
    AstroTemporalGNNConfig,
    ModelConfig,
    create_config_from_dict,
    get_available_presets,
    get_model_config,
    get_model_presets,
    get_model_type_for_task,
    get_preset,
    list_presets,
)
from .registry import PREPROCESSOR_REGISTRY
from .surveys import (
    SurveyOptimizationConfig,
    get_available_surveys,
    get_survey_config,
    get_survey_coordinates,
    get_survey_features,
    get_survey_info,
    get_survey_magnitudes,
    get_survey_optimization,
    register_survey,
)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries

    Returns:
        Merged configuration
    """
    merged = {}

    for config in configs:
        if config:
            merged.update(config)

    return merged


def get_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Experiment configuration
    """
    config_path = get_data_config().configs_dir / f"{experiment_name}.yaml"

    if config_path.exists():
        config = load_config(config_path)
        return config if isinstance(config, dict) else TRAINING_DEFAULTS.copy()
    else:
        # Return default config if no experiment config exists
        return TRAINING_DEFAULTS.copy()


def save_experiment_config(config: Dict[str, Any], experiment_name: str):
    """
    Save configuration for a specific experiment.

    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment
    """
    config_path = get_data_config().configs_dir / f"{experiment_name}.yaml"
    save_config(config, config_path)


__all__ = [
    # Data configuration
    "DataConfig",
    "data_config",
    "get_data_config",
    "setup_data_directories",
    "list_survey_paths",
    # Defaults
    "TRAINING_DEFAULTS",
    "MODEL_DEFAULTS",
    "DATA_DEFAULTS",
    "TASK_DEFAULTS",
    "SURVEY_TRAINING_DEFAULTS",
    "HPO_SEARCH_SPACES",
    "get_training_config",
    "get_data_defaults",
    "get_hpo_search_space",
    # Model configurations
    "ModelConfig",
    "AstroNodeGNNConfig",
    "AstroGraphGNNConfig",
    "AstroTemporalGNNConfig",
    "AstroPointNetConfig",
    "get_preset",
    "list_presets",
    "get_available_presets",
    "create_config_from_dict",
    "get_model_type_for_task",
    "get_model_config",
    "get_model_presets",
    # Registry
    "PREPROCESSOR_REGISTRY",
    # Survey configurations
    "get_survey_config",
    "get_survey_coordinates",
    "get_survey_features",
    "get_survey_magnitudes",
    "get_survey_info",
    "get_available_surveys",
    "register_survey",
    "get_survey_optimization",
    "SurveyOptimizationConfig",
    # Config utilities
    "load_config",
    "save_config",
    "merge_configs",
    "get_experiment_config",
    "save_experiment_config",
]
