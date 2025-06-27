"""
AstroLab Configuration Management
================================

Simple configuration management with automatic path setup.
"""

from .loader import (
    ConfigLoader,
    load_experiment_config,
    load_survey_config,
    setup_experiment_from_config,
)
from .params import (
    get_data_params,
    get_lightning_params,
    get_mlflow_params,
    get_trainer_params,
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
    "get_trainer_params",
    "get_lightning_params",
    "get_mlflow_params",
    "get_data_params",
    # Survey configurations
    "get_survey_config",
    "get_survey_coordinates",
    "get_survey_features",
    "get_survey_magnitudes",
    "register_survey",
]
