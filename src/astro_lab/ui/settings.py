"""
AstroLab UI Settings Integration
===============================

Clean marimo UI integration with AstroLab's configuration system.
Handles ConfigLoader and data_config integration properly.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import marimo as mo

from ..config.loader import ConfigLoader, load_survey_config
from ..config.surveys import get_survey_config as get_survey_config_from_surveys
from ..data.config import data_config, get_survey_paths
from ..models.config import CONFIGS as MODEL_CONFIGS
from ..models.config import get_predefined_config


class UIConfigManager:
    """UI interface for AstroLab's configuration system."""

    def __init__(self):
        """Initialize the UI config manager."""
        self.config_loader = ConfigLoader()
        self.current_config: Optional[Dict[str, Any]] = None
        self.available_configs = self._discover_configs()

    def _discover_configs(self) -> Dict[str, List[str]]:
        """Discover available configuration files."""
        configs_dir = Path("configs")

        available = {
            "experiments": [],
            "surveys": [],
            "models": list(MODEL_CONFIGS.keys()),
            "training": ["default"],  # Simplified training configs
        }

        if configs_dir.exists():
            # Find experiment configs
            for config_file in configs_dir.glob("*.yaml"):
                if config_file.name != "default.yaml":
                    available["experiments"].append(config_file.stem)

            # Find survey configs
            surveys_dir = configs_dir / "surveys"
            if surveys_dir.exists():
                for survey_file in surveys_dir.glob("*.yaml"):
                    available["surveys"].append(survey_file.stem)

        return available

    def load_config(
        self,
        config_path: str = "configs/default.yaml",
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load configuration using ConfigLoader."""
        self.config_loader = ConfigLoader(config_path)
        self.current_config = self.config_loader.load_config(experiment_name)
        return self.current_config

    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get currently loaded configuration."""
        return self.current_config

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration from data_config."""
        return {
            "base_dir": str(data_config.base_dir),
            "raw_dir": str(data_config.raw_dir),
            "processed_dir": str(data_config.processed_dir),
            "cache_dir": str(data_config.cache_dir),
            "results_dir": str(data_config.results_dir),
            "experiments_dir": str(data_config.experiments_dir),
        }

    def get_model_configs(self) -> Dict[str, Any]:
        """Get available model configurations."""
        return {name: config.to_dict() for name, config in MODEL_CONFIGS.items()}

    def get_training_configs(self) -> Dict[str, str]:
        """Get available training configurations."""
        return {
            "default": "Default training configuration",
            "fast": "Fast training for testing",
            "production": "Production training with full optimization",
        }

    def setup_experiment(self, experiment_name: str) -> Dict[str, Path]:
        """Setup experiment directories."""
        data_config.ensure_experiment_directories(experiment_name)
        return data_config.get_experiment_paths(experiment_name)

    def get_survey_info(self, survey: str) -> Dict[str, Any]:
        """Get survey configuration and paths."""
        try:
            survey_config = get_survey_config_from_surveys(survey)
            survey_paths = get_survey_paths(survey)
            return {
                "config": survey_config,
                "paths": {k: str(v) for k, v in survey_paths.items()},
            }
        except Exception as e:
            return {"error": str(e)}


# Global UI config manager
ui_config = UIConfigManager()


def ui_config_loader() -> mo.ui.dictionary:
    """Configuration loader UI component."""
    configs_dir = Path("configs")
    config_files = []

    if configs_dir.exists():
        config_files = [f.name for f in configs_dir.glob("*.yaml")]

    if not config_files:
        config_files = ["default.yaml"]

    return mo.ui.dictionary(
        {
            "config_file": mo.ui.dropdown(
                label="Configuration File",
                options=config_files,
                value="default.yaml",
            ),
            "experiment_name": mo.ui.text(
                label="Experiment Name",
                placeholder="e.g., gaia_stellar_classification",
            ),
            "load_button": mo.ui.button(label="📂 Load Configuration"),
            "create_experiment": mo.ui.button(label="🧪 Create Experiment"),
        },
        label="📂 Configuration Loader",
    )


def ui_data_paths() -> mo.ui.dictionary:
    """Data paths configuration UI component."""
    data_info = ui_config.get_data_config()

    return mo.ui.dictionary(
        {
            "base_dir": mo.ui.text(
                label="Base Data Directory",
                value=data_info["base_dir"],
                disabled=True,
            ),
            "raw_dir": mo.ui.text(
                label="Raw Data Directory",
                value=data_info["raw_dir"],
                disabled=True,
            ),
            "processed_dir": mo.ui.text(
                label="Processed Data Directory",
                value=data_info["processed_dir"],
                disabled=True,
            ),
            "cache_dir": mo.ui.text(
                label="Cache Directory",
                value=data_info["cache_dir"],
                disabled=True,
            ),
            "setup_dirs": mo.ui.button(label="📁 Setup Directories"),
        },
        label="📁 Data Paths",
    )


def ui_survey_selector() -> mo.ui.dictionary:
    """Survey selection and configuration UI component."""
    available_surveys = ui_config.available_configs["surveys"]

    if not available_surveys:
        available_surveys = ["gaia", "sdss", "nsa", "linear", "tng50"]

    return mo.ui.dictionary(
        {
            "survey": mo.ui.dropdown(
                label="Survey",
                options=available_surveys,
                value=available_surveys[0] if available_surveys else "gaia",
            ),
            "load_survey_config": mo.ui.button(label="📊 Load Survey Config"),
            "setup_survey_dirs": mo.ui.button(label="📁 Setup Survey Directories"),
        },
        label="📊 Survey Configuration",
    )


def ui_model_selector() -> mo.ui.dictionary:
    """Model selection UI component."""
    available_models = ui_config.available_configs["models"]

    return mo.ui.dictionary(
        {
            "model": mo.ui.dropdown(
                label="Model",
                options=available_models,
                value=available_models[0] if available_models else "gaia_classifier",
            ),
            "load_model_config": mo.ui.button(label="🤖 Load Model Config"),
            "show_model_details": mo.ui.button(label="📋 Show Model Details"),
        },
        label="🤖 Model Configuration",
    )


def ui_training_selector() -> mo.ui.dictionary:
    """Training configuration UI component."""
    available_training = ui_config.available_configs["training"]

    return mo.ui.dictionary(
        {
            "training_config": mo.ui.dropdown(
                label="Training Configuration",
                options=available_training,
                value=available_training[0] if available_training else "default",
            ),
            "load_training_config": mo.ui.button(label="🏃 Load Training Config"),
            "show_training_details": mo.ui.button(label="📋 Show Training Details"),
        },
        label="🏃 Training Configuration",
    )


def ui_experiment_manager() -> mo.ui.dictionary:
    """Experiment management UI component."""
    return mo.ui.dictionary(
        {
            "experiment_name": mo.ui.text(
                label="Experiment Name",
                placeholder="e.g., gaia_stellar_classification_v1",
            ),
            "create_experiment": mo.ui.button(label="🧪 Create Experiment"),
            "list_experiments": mo.ui.button(label="📋 List Experiments"),
            "load_experiment": mo.ui.button(label="📂 Load Experiment"),
        },
        label="🧪 Experiment Management",
    )


def ui_config_status() -> mo.ui.dictionary:
    """Configuration status display UI component."""
    current_config = ui_config.get_current_config()
    data_info = ui_config.get_data_config()

    status_info = {
        "config_loaded": "✅" if current_config else "❌",
        "data_dirs_setup": "✅" if Path(data_info["base_dir"]).exists() else "❌",
        "experiments_dir": "✅"
        if Path(data_info["experiments_dir"]).exists()
        else "❌",
        "mlflow_ready": "✅" if Path(data_info["experiments_dir"]).exists() else "❌",
    }

    return mo.ui.dictionary(
        {
            "config_status": mo.ui.text(
                label="Configuration Status",
                value=status_info["config_loaded"],
                disabled=True,
            ),
            "data_status": mo.ui.text(
                label="Data Directories",
                value=status_info["data_dirs_setup"],
                disabled=True,
            ),
            "experiments_status": mo.ui.text(
                label="Experiments Directory",
                value=status_info["experiments_dir"],
                disabled=True,
            ),
            "mlflow_status": mo.ui.text(
                label="MLflow Ready",
                value=status_info["mlflow_ready"],
                disabled=True,
            ),
            "refresh_status": mo.ui.button(label="🔄 Refresh Status"),
        },
        label="📊 Configuration Status",
    )


def handle_config_actions(
    components: Dict[str, mo.ui.dictionary],
) -> Optional[Dict[str, Any]]:
    """Handle configuration-related actions."""
    result = {}

    # Handle config loader actions
    if components.get("config_loader"):
        config_loader = components["config_loader"]
        if config_loader["load_button"].value:
            try:
                config = ui_config.load_config(
                    str(config_loader["config_file"].value),
                    str(config_loader["experiment_name"].value)
                    if config_loader["experiment_name"].value
                    else None,
                )
                result["config_loaded"] = config
            except Exception as e:
                result["error"] = f"Failed to load config: {e}"

    # Handle data paths actions
    if components.get("data_paths"):
        data_paths = components["data_paths"]
        if data_paths["setup_dirs"].value:
            try:
                data_config.setup_directories()
                result["directories_setup"] = "Data directories created successfully"
            except Exception as e:
                result["error"] = f"Failed to setup directories: {e}"

    # Handle survey actions
    if components.get("survey_selector"):
        survey_selector = components["survey_selector"]
        if survey_selector["load_survey_config"].value:
            try:
                survey_info = ui_config.get_survey_info(
                    str(survey_selector["survey"].value)
                )
                result["survey_info"] = survey_info
            except Exception as e:
                result["error"] = f"Failed to load survey config: {e}"

        if survey_selector["setup_survey_dirs"].value:
            try:
                survey = str(survey_selector["survey"].value)
                data_config.ensure_survey_directories(survey)
                result["survey_dirs_setup"] = f"Directories created for {survey}"
            except Exception as e:
                result["error"] = f"Failed to setup survey directories: {e}"

    # Handle model actions
    if components.get("model_selector"):
        model_selector = components["model_selector"]
        if model_selector["load_model_config"].value:
            try:
                model_config = ui_config.get_model_configs()[
                    str(model_selector["model"].value)
                ]
                result["model_config"] = model_config
            except Exception as e:
                result["error"] = f"Failed to load model config: {e}"

    # Handle training actions
    if components.get("training_selector"):
        training_selector = components["training_selector"]
        if training_selector["load_training_config"].value:
            try:
                training_configs = ui_config.get_training_configs()
                result["training_config"] = training_configs[
                    str(training_selector["training_config"].value)
                ]
            except Exception as e:
                result["error"] = f"Failed to load training config: {e}"

    # Handle experiment actions
    if components.get("experiment_manager"):
        experiment_manager = components["experiment_manager"]
        if experiment_manager["create_experiment"].value:
            try:
                experiment_name = (
                    str(experiment_manager["experiment_name"].value)
                    if experiment_manager["experiment_name"].value
                    else None
                )
                if experiment_name:
                    paths = ui_config.setup_experiment(experiment_name)
                    result["experiment_created"] = {
                        "name": experiment_name,
                        "paths": {k: str(v) for k, v in paths.items()},
                    }
                else:
                    result["error"] = "Experiment name is required"
            except Exception as e:
                result["error"] = f"Failed to create experiment: {e}"

    return result if result else None


def create_settings_ui() -> mo.ui.dictionary:
    """Create the main settings UI."""
    return mo.ui.dictionary(
        {
            "config_loader": ui_config_loader(),
            "data_paths": ui_data_paths(),
            "survey_selector": ui_survey_selector(),
            "model_selector": ui_model_selector(),
            "training_selector": ui_training_selector(),
            "experiment_manager": ui_experiment_manager(),
            "config_status": ui_config_status(),
        },
        label="⚙️ AstroLab Settings",
    )
