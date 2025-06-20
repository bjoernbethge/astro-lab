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
    get_trainer_params,
    get_lightning_params,
    get_optuna_params,
    get_mlflow_params,
    get_data_params,
    validate_parameter_conflicts,
    print_parameter_distribution,
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
] 