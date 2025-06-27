#!/usr/bin/env python3
"""
AstroLab Config CLI
==================

Simplified configuration management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from astro_lab.config.surveys import SURVEY_CONFIGS
from astro_lab.models.config import list_presets


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_and_prepare_training_config(
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load training configuration - simplified version.

    Priority: CLI overrides > config file > preset > defaults
    """
    config: Dict[str, Any] = {}

    # 1. Load from file if provided
    if config_path:
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)
            if loaded_config is not None:
                config = loaded_config

    # 2. Or use preset
    elif preset:
        config = {
            "preset": preset,
            "model_type": preset.split("_")[0],  # e.g., "node_classifier" -> "node"
        }

    # 3. Apply CLI overrides
    if cli_overrides:
        config.update(cli_overrides)

    # 4. Set minimal defaults (let models/trainer handle the rest)
    config.setdefault("dataset", "gaia")
    config.setdefault("max_epochs", 50)
    config.setdefault("batch_size", 32)

    # Simple model type inference if needed
    if "model" in config and "model_type" not in config:
        model_name = config["model"].lower()
        if "node" in model_name:
            config["model_type"] = "node"
        elif "graph" in model_name:
            config["model_type"] = "graph"
        elif "temporal" in model_name:
            config["model_type"] = "temporal"
        elif "point" in model_name:
            config["model_type"] = "point"

    return config


def main(args) -> int:
    """Handle config CLI commands."""
    logger = setup_logging()

    try:
        if args.config_command == "create":
            return _create_config(args)
        elif args.config_command == "surveys":
            return _show_surveys()
        elif args.config_command == "show":
            return _show_survey_config(args.survey)
        else:
            logger.error(f"Unknown config command: {args.config_command}")
            return 1

    except Exception as e:
        logger.error(f"Config command failed: {e}")
        return 1


def _create_config(args) -> int:
    """Create a new configuration file."""
    from astro_lab.config.surveys import get_survey_config

    try:
        survey_config = get_survey_config(args.template)

        config = {
            "experiment_name": f"{args.template}_experiment",
            "dataset": args.template,
            "model": "node",  # Default model
            "max_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "precision": "16-mixed",
            "mlflow": {
                "tracking_uri": "file://./mlruns",
                "experiment_name": f"{args.template}_experiment",
            },
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"✅ Created config: {args.output}")
        return 0

    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return 1


def _show_surveys() -> int:
    """Show available surveys."""
    print("📊 Available Surveys:")
    print("=" * 40)

    for survey, config in SURVEY_CONFIGS.items():
        print(f"\n🔭 {survey.upper()}: {config['name']}")
        print(f"   Coordinates: {', '.join(config['coord_cols'])}")
        print(f"   Magnitudes: {', '.join(config['mag_cols'])}")
        if config["extra_cols"]:
            print(f"   Extra: {', '.join(config['extra_cols'])}")

    return 0


def _show_survey_config(survey: str) -> int:
    """Show detailed survey configuration."""
    try:
        from astro_lab.config.surveys import get_survey_config

        config = get_survey_config(survey)

        print(f"📊 Survey: {survey.upper()}")
        print(f"Name: {config['name']}")
        print(f"Coordinates: {config['coord_cols']}")
        print(f"Magnitudes: {config['mag_cols']}")
        print(f"Extra columns: {config['extra_cols']}")
        if config["color_pairs"]:
            print(f"Color pairs: {config['color_pairs']}")

        return 0

    except Exception as e:
        print(f"❌ Failed to show survey config: {e}")
        return 1
