#!/usr/bin/env python3
"""
AstroLab Configuration Loader
============================

Simple configuration loading with automatic path setup.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from astro_lab.data.config import data_config


class ConfigLoader:
    """Load and manage AstroLab configurations with automatic path setup."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_file: Path to config file. If None, uses default.yaml
        """
        self.config_file = config_file or "configs/default.yaml"
        self.config: Optional[Dict[str, Any]] = None

    def load_config(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration and set up experiment paths.

        Args:
            experiment_name: Override experiment name from config

        Returns:
            Loaded configuration dictionary
        """
        config_path = Path(self.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        if experiment_name and self.config:
            self.config["experiment_name"] = experiment_name

        if self.config:
            exp_name = self.config["experiment_name"]
            data_config.ensure_experiment_directories(exp_name)
            self._update_paths()

        return self.config or {}

    def _update_paths(self):
        """Update configuration paths using data_config."""
        if not self.config:
            return
        exp_name = self.config["experiment_name"]
        exp_paths = data_config.get_experiment_paths(exp_name)

        if "checkpoints" in self.config:
            self.config["checkpoints"]["dir"] = str(exp_paths["checkpoints"])
        if "data" in self.config:
            self.config["data"]["base_dir"] = str(data_config.base_dir)

    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save config. If None, saves to experiment config path.
        """
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")

        if output_path is None:
            exp_name = self.config["experiment_name"]
            output_file = data_config.configs_dir / f"{exp_name}.yaml"
        else:
            output_file = Path(output_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"💾 Config saved to: {output_file}")


def load_experiment_config(
    experiment_name: str, config_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load experiment configuration.

    Args:
        experiment_name: Name of the experiment
        config_file: Config file to load (default: configs/default.yaml)

    Returns:
        Loaded configuration dictionary
    """
    loader = ConfigLoader(config_file)
    return loader.load_config(experiment_name)


def load_survey_config(survey: str) -> Dict[str, Any]:
    """
    Convenience function to load survey configuration.

    Args:
        survey: Name of the survey (e.g., 'gaia', 'sdss')

    Returns:
        Survey configuration dictionary
    """
    from .surveys import get_survey_config

    return get_survey_config(survey)


def setup_experiment_from_config(config_path: str, experiment_name: str):
    """
    Set up experiment directories and environment from config file.

    Args:
        config_path: Path to configuration file
        experiment_name: Name of the experiment
    """
    loader = ConfigLoader(config_path)
    config = loader.load_config(experiment_name)

    print("🧪 Experiment setup complete:")
    print(f"   - Name: {experiment_name}")
    if config:
        print(f"   - MLflow: {config['mlflow']['tracking_uri']}")
        if "checkpoints" in config:
            print(f"   - Checkpoints: {config['checkpoints']['dir']}")
    print(f"   - Config saved: {data_config.configs_dir / f'{experiment_name}.yaml'}")

    return config
