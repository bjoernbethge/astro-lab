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

from ..data.config import data_config, get_survey_paths
from ..models.config import CONFIGS as MODEL_CONFIGS
from ..models.config import get_predefined_config
from ..training.config import PREDEFINED_TRAINING_CONFIGS
from ..utils.config.loader import ConfigLoader, load_survey_config


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
            "training": list(PREDEFINED_TRAINING_CONFIGS.keys()),
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
            name: config.description or "No description"
            for name, config in PREDEFINED_TRAINING_CONFIGS.items()
        }

    def setup_experiment(self, experiment_name: str) -> Dict[str, Path]:
        """Setup experiment directories."""
        data_config.ensure_experiment_directories(experiment_name)
        return data_config.get_experiment_paths(experiment_name)

    def get_survey_info(self, survey: str) -> Dict[str, Any]:
        """Get survey configuration and paths."""
        try:
            survey_config = load_survey_config(survey)
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
            "model_name": mo.ui.dropdown(
                label="Model Configuration",
                options=available_models,
                value=available_models[0] if available_models else "gaia_classifier",
            ),
            "load_model_config": mo.ui.button(label="🤖 Load Model Config"),
            "show_model_details": mo.ui.button(label="ℹ️ Show Details"),
        },
        label="🤖 Model Configuration",
    )


def ui_training_selector() -> mo.ui.dictionary:
    """Training configuration selector UI component."""
    available_training = ui_config.available_configs["training"]

    return mo.ui.dictionary(
        {
            "training_config": mo.ui.dropdown(
                label="Training Configuration",
                options=available_training,
                value=available_training[0]
                if available_training
                else "gaia_stellar_training",
            ),
            "load_training_config": mo.ui.button(label="🏋️ Load Training Config"),
            "show_training_details": mo.ui.button(label="ℹ️ Show Details"),
        },
        label="🏋️ Training Configuration",
    )


def ui_experiment_manager() -> mo.ui.dictionary:
    """Experiment management UI component."""
    return mo.ui.dictionary(
        {
            "experiment_name": mo.ui.text(
                label="Experiment Name",
                placeholder="e.g., gaia_stellar_v2",
            ),
            "description": mo.ui.text_area(
                label="Description",
                placeholder="Experiment description...",
            ),
            "create_experiment": mo.ui.button(label="🧪 Create Experiment"),
            "list_experiments": mo.ui.button(label="📋 List Experiments"),
        },
        label="🧪 Experiment Management",
    )


def ui_config_status() -> mo.ui.dictionary:
    """Configuration status display."""
    current_config = ui_config.get_current_config()

    status_text = "No configuration loaded"
    if current_config:
        exp_name = current_config.get("mlflow", {}).get("experiment_name", "Unknown")
        status_text = f"Loaded: {exp_name}"

    return mo.ui.dictionary(
        {
            "status": mo.ui.text(
                label="Configuration Status",
                value=status_text,
                disabled=True,
            ),
            "config_details": mo.ui.text_area(
                label="Configuration Details",
                value=str(current_config) if current_config else "No config loaded",
                disabled=True,
            ),
            "refresh": mo.ui.button(label="🔄 Refresh Status"),
        },
        label="ℹ️ Configuration Status",
    )


def handle_config_actions(
    components: Dict[str, mo.ui.dictionary],
) -> Optional[Dict[str, Any]]:
    """Handle configuration-related actions from UI components."""

    # Handle config loading
    if "config_loader" in components:
        loader_values = components["config_loader"].value

        if loader_values.get("load_button"):
            config_file = f"configs/{loader_values.get('config_file', 'default.yaml')}"
            experiment_name = loader_values.get("experiment_name")
            experiment_name = str(experiment_name) if experiment_name else None

            try:
                config = ui_config.load_config(config_file, experiment_name)
                print(f"✅ Configuration loaded: {config_file}")
                if experiment_name:
                    print(f"🧪 Experiment: {experiment_name}")
                return config
            except Exception as e:
                print(f"❌ Failed to load configuration: {e}")
                return None

        if loader_values.get("create_experiment"):
            experiment_name = loader_values.get("experiment_name")
            if experiment_name:
                experiment_name = str(experiment_name)
                try:
                    paths = ui_config.setup_experiment(experiment_name)
                    print(f"✅ Experiment created: {experiment_name}")
                    print(f"📁 Paths: {paths}")
                except Exception as e:
                    print(f"❌ Failed to create experiment: {e}")

    # Handle survey configuration
    if "survey_selector" in components:
        survey_values = components["survey_selector"].value

        if survey_values.get("load_survey_config"):
            survey = survey_values.get("survey", "gaia")
            survey = str(survey)
            try:
                survey_info = ui_config.get_survey_info(survey)
                print(f"✅ Survey config loaded: {survey}")
                return survey_info
            except Exception as e:
                print(f"❌ Failed to load survey config: {e}")

        if survey_values.get("setup_survey_dirs"):
            survey = survey_values.get("survey", "gaia")
            survey = str(survey)
            try:
                data_config.ensure_survey_directories(survey)
                print(f"✅ Survey directories created: {survey}")
            except Exception as e:
                print(f"❌ Failed to create survey directories: {e}")

    # Handle model configuration
    if "model_selector" in components:
        model_values = components["model_selector"].value

        if model_values.get("load_model_config"):
            model_name = model_values.get("model_name", "gaia_classifier")
            model_name = str(model_name)
            try:
                model_config = get_predefined_config(model_name)
                print(f"✅ Model config loaded: {model_name}")
                print(f"🤖 Details: {model_config.to_dict()}")
                return model_config.to_dict()
            except Exception as e:
                print(f"❌ Failed to load model config: {e}")

    # Handle data paths
    if "data_paths" in components:
        paths_values = components["data_paths"].value

        if paths_values.get("setup_dirs"):
            try:
                data_config.setup_directories()
                print("✅ Data directories created")
            except Exception as e:
                print(f"❌ Failed to create data directories: {e}")

    return None


# Export main components
__all__ = [
    "UIConfigManager",
    "ui_config",
    "ui_config_loader",
    "ui_data_paths",
    "ui_survey_selector",
    "ui_model_selector",
    "ui_training_selector",
    "ui_experiment_manager",
    "ui_config_status",
    "handle_config_actions",
]
