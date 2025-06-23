#!/usr/bin/env python3
"""
AstroLab Configuration Loader
============================

Centralized configuration management with automatic path setup and experiment organization.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        # Load base config
        config_path = Path(self.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Override experiment name if provided
        if experiment_name and self.config:
            self.config["mlflow"]["experiment_name"] = experiment_name

        # Set up experiment directories
        if self.config:
            exp_name = self.config["mlflow"]["experiment_name"]
            data_config.ensure_experiment_directories(exp_name)

            # Update paths in config using data_config
            self._update_paths()

            # Set environment variables for MLflow
            self._setup_environment()

        return self.config or {}

    def _update_paths(self):
        """Update configuration paths using data_config."""
        if not self.config:
            return

        exp_name = self.config["mlflow"]["experiment_name"]
        exp_paths = data_config.get_experiment_paths(exp_name)

        # Always ensure MLflow tracking URI points to the correct experiment directory
        # This ensures consistency across all experiments
        if "mlflow" not in self.config:
            self.config["mlflow"] = {}
        
        # Update tracking URI to use experiment-specific path
        self.config["mlflow"]["tracking_uri"] = f"file://{exp_paths['mlruns'].absolute()}"
        
        # Also set artifact location for MLflow
        if "artifact_location" not in self.config["mlflow"]:
            self.config["mlflow"]["artifact_location"] = f"file://{exp_paths['artifacts'].absolute()}"

        # Update checkpoint directory (only if checkpoints section exists)
        if "checkpoints" in self.config:
            self.config["checkpoints"]["dir"] = str(exp_paths["checkpoints"])

        # Update data base directory
        self.config["data"]["base_dir"] = str(data_config.base_dir)

    def _setup_environment(self):
        """Set up environment variables for MLflow and Lightning."""
        if not self.config:
            return

        # MLflow environment
        os.environ["MLFLOW_TRACKING_URI"] = self.config["mlflow"]["tracking_uri"]

        # Lightning environment (optional)
        if "LIGHTNING_LOGS_DIR" not in os.environ and "checkpoints" in self.config:
            os.environ["LIGHTNING_LOGS_DIR"] = self.config["checkpoints"]["dir"]

    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow-specific configuration."""
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self.config["mlflow"]

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self.config["training"]

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self.config["model"]

    def get_survey_config(self, survey: str) -> Dict[str, Any]:
        """Get survey-specific configuration from configs/surveys/ directory."""
        survey_config_path = Path(f"configs/surveys/{survey}.yaml")

        if not survey_config_path.exists():
            raise ValueError(f"Survey config not found: {survey_config_path}")

        with open(survey_config_path, "r", encoding="utf-8") as f:
            survey_config = yaml.safe_load(f)

        return survey_config["survey"]

    def list_available_surveys(self) -> List[str]:
        """List all available survey configurations."""
        surveys_dir = Path("configs/surveys")
        if not surveys_dir.exists():
            return []

        survey_files = surveys_dir.glob("*.yaml")
        return [f.stem for f in survey_files]

    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save config. If None, saves to experiment config path.
        """
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")

        if output_path is None:
            exp_name = self.config["mlflow"]["experiment_name"]
            # Save to configs directory, not experiments
            output_path = data_config.configs_dir / f"{exp_name}.yaml"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

        print(f"💾 Config saved to: {output_path}")

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
    loader = ConfigLoader()
    return loader.get_survey_config(survey)

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

def validate_config_integration(config_path: str = "configs/default.yaml") -> bool:
    """
    Validate that config integration works correctly.
    
    Args:
        config_path: Path to config file to test
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        print("🔍 Validating config integration...")
        
        # Test 1: Load config
        loader = ConfigLoader(config_path)
        config = loader.load_config("test_experiment")
        
        # Test 2: Check MLflow URI
        mlflow_uri = config.get("mlflow", {}).get("tracking_uri", "")
        if not mlflow_uri.startswith("file://"):
            print(f"❌ Invalid MLflow URI format: {mlflow_uri}")
            return False
            
        # Test 3: Check if URI points to data/experiments
        if "data/experiments" not in mlflow_uri and "data\\experiments" not in mlflow_uri:
            print(f"❌ MLflow URI doesn't point to data/experiments: {mlflow_uri}")
            return False
            
        # Test 4: Check environment variable
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if env_uri != mlflow_uri:
            print(f"❌ Environment variable mismatch: {env_uri} != {mlflow_uri}")
            return False
            
        # Test 5: Check experiment directories exist
        exp_name = config.get("mlflow", {}).get("experiment_name", "")
        exp_paths = data_config.get_experiment_paths(exp_name)
        
        for path_name, path in exp_paths.items():
            if not path.exists():
                print(f"❌ Experiment directory missing: {path}")
                return False
                
        print("✅ Config integration validation passed!")
        print(f"   - MLflow URI: {mlflow_uri}")
        print(f"   - Experiment: {exp_name}")
        print(f"   - Directories: {list(exp_paths.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config integration validation failed: {e}")
        return False
