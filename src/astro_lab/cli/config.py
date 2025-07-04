#!/usr/bin/env python3
"""
AstroLab Config CLI
==================

Configuration management for AstroLab.
"""

import logging
import sys
from typing import Any, Dict, Optional

import yaml

from ..config import get_config, get_survey_config


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
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

    Priority: CLI overrides > config file > defaults
    """
    config: Dict[str, Any] = {}

    # 1. Load from file if provided
    if config_path:
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)
            if loaded_config is not None:
                config = loaded_config

    # 2. Or use preset (simplified preset system)
    elif preset:
        # Basic preset system
        preset_configs = {
            "graph_classifier_small": {
                "model_type": "graph",
                "hidden_dim": 64,
                "num_layers": 2,
                "learning_rate": 0.001,
            },
            "node_classifier_medium": {
                "model_type": "node",
                "hidden_dim": 128,
                "num_layers": 3,
                "learning_rate": 0.001,
            },
            "temporal_gnn": {
                "model_type": "temporal",
                "hidden_dim": 128,
                "num_layers": 3,
                "learning_rate": 0.0005,
            },
        }

        if preset in preset_configs:
            config = preset_configs[preset].copy()
        else:
            # Fallback: infer model type from preset name
            if "node" in preset.lower():
                config["model_type"] = "node"
            elif "graph" in preset.lower():
                config["model_type"] = "graph"
            elif "temporal" in preset.lower():
                config["model_type"] = "temporal"
            elif "point" in preset.lower():
                config["model_type"] = "point"

    # 3. Apply CLI overrides
    if cli_overrides:
        config.update(cli_overrides)

    # 4. Set minimal defaults
    config.setdefault("dataset", "gaia")
    config.setdefault("model_type", "graph")
    config.setdefault("max_epochs", 50)
    config.setdefault("batch_size", 32)
    config.setdefault("learning_rate", 0.001)

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
    try:
        get_survey_config(args.template)

        config = {
            "experiment_name": f"{args.template}_experiment",
            "dataset": args.template,
            "model_type": "graph",  # Default model type
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

    for survey, config in get_config()["surveys"].items():
        print(f"\n🔭 {survey.upper()}: {config['name']}")
        print(f"   Coordinates: {', '.join(config['coord_cols'])}")
        print(f"   Magnitudes: {', '.join(config['mag_cols'])}")
        if config["extra_cols"]:
            print(f"   Extra: {', '.join(config['extra_cols'])}")

    return 0


def _show_survey_config(survey: str) -> int:
    """Show detailed survey configuration."""
    try:
        config = get_survey_config(survey)

        print(f"📊 Survey: {survey.upper()}")
        print(f"Name: {config['name']}")
        print(f"Coordinates: {config['coord_cols']}")
        print(f"Magnitudes: {config['mag_cols']}")
        print(f"Extra columns: {config['extra_cols']}")
        if config.get("color_pairs"):
            print(f"Color pairs: {config['color_pairs']}")

        return 0

    except Exception as e:
        print(f"❌ Failed to show survey config: {e}")
        return 1
