"""
AstroLab Configuration Management
================================

Centralized configuration management with automatic path setup and experiment organization.
"""

from .loader import (
    ConfigLoader,
    load_experiment_config,
    load_survey_config,
    setup_experiment_from_config,
)
from .params import (
    distribute_config_parameters,
    get_data_params,
    get_lightning_params,
    get_mlflow_params,
    get_optuna_params,
    get_trainer_params,
    print_parameter_distribution,
    validate_parameter_conflicts,
)
from .surveys import (
    get_survey_config,
    get_survey_coordinates,
    get_survey_features,
    get_survey_magnitudes,
    register_survey,
)

__all__ = [
    # Loader functions
    "ConfigLoader",
    "load_experiment_config", 
    "load_survey_config",
    "setup_experiment_from_config",
    # Parameter management
    "distribute_config_parameters",
    "get_trainer_params",
    "get_lightning_params", 
    "get_optuna_params",
    "get_mlflow_params",
    "get_data_params",
    "validate_parameter_conflicts",
    "print_parameter_distribution",
    # Survey configurations
    "get_survey_config",
    "get_survey_coordinates",
    "get_survey_features",
    "get_survey_magnitudes",
    "register_survey",
]
