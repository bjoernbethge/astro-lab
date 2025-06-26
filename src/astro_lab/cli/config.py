#!/usr/bin/env python3
"""
AstroLab Config CLI
==================

Configuration management for AstroLab.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from astro_lab.data.config import data_config
from astro_lab.config.loader import ConfigLoader
from astro_lab.config.params import distribute_config_parameters
from astro_lab.config.surveys import SURVEY_CONFIGS, get_survey_config


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def main(args) -> int:
    """Main entry point for config command."""
    logger = setup_logging()
    try:
        if args.config_command == "create":
            return _config_create(args, logger)
        elif args.config_command == "surveys":
            return _config_surveys(args, logger)
        elif args.config_command == "show":
            return _config_show(args, logger)
        else:
            logger.error(f"Unknown config command: {args.config_command}")
            return 1
    except Exception as e:
        logger.error(f"❌ Configuration operation failed: {e}")
        return 1


def _config_create(args, logger: logging.Logger) -> int:
    """Create configuration file."""
    logger.info(f"📝 Creating configuration: {args.output}")
    try:
        # Create minimal, user-friendly configuration
        survey_name = args.template

        config_dict = {
            "name": f"{survey_name}_experiment",
            "dataset": survey_name,
            "model": "gaia_classifier",  # Default model
            "max_epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "precision": "16-mixed",  # Mixed precision für RTX 4070
            "early_stopping_patience": 10,
            "num_workers": 4,
        }

        # Write to file
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"✅ Configuration created: {args.output}")
        logger.info(f"📋 Template: {args.template}")
        logger.info(f"📊 Model: {config_dict['model']}")
        logger.info(f"🔄 Max epochs: {config_dict['max_epochs']}")
        logger.info(f"📦 Batch size: {config_dict['batch_size']}")
        logger.info(f"💡 Learning rate: {config_dict['learning_rate']}")
        return 0
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        return 1


def _config_surveys(args, logger: logging.Logger) -> int:
    """Show available survey configurations."""
    logger.info("📋 Available survey configurations:")

    for survey_name in SURVEY_CONFIGS.keys():
        try:
            survey_config = get_survey_config(survey_name)
            description = survey_config["survey"]["description"]
            features_count = len(survey_config["survey"]["features"])

            logger.info(f"  🌌 {survey_name}")
            logger.info(f"     📝 {description}")
            logger.info(f"     📊 {features_count} features")
            logger.info("")

        except Exception as e:
            logger.warning(f"  ⚠️  {survey_name}: Could not load config ({e})")

    logger.info("✅ Survey configurations listed!")
    return 0


def _config_show(args, logger: logging.Logger) -> int:
    """Show configuration details."""
    logger.info(f"📋 Survey configuration: {args.survey}")

    try:
        survey_config = get_survey_config(args.survey)

        logger.info(f"📝 Name: {survey_config['survey']['name']}")
        logger.info(f"📝 Description: {survey_config['survey']['description']}")
        logger.info("")

        logger.info("📊 Features:")
        for feature in survey_config["survey"]["features"]:
            logger.info(f"  - {feature}")
        logger.info("")

        logger.info("🔧 Graph configuration:")
        graph_config = survey_config["survey"]["graph"]
        logger.info(f"  - K-neighbors: {graph_config['k_neighbors']}")
        logger.info(f"  - Distance metric: {graph_config['distance_metric']}")
        logger.info("")

        logger.info("⚙️ Processing configuration:")
        processing_config = survey_config["survey"]["processing"]
        logger.info(
            f"  - Normalize features: {processing_config['normalize_features']}"
        )
        logger.info("")

        # Check if processed data exists
        processed_path = data_config.processed_dir / args.survey / f"{args.survey}.pt"

        if processed_path.exists():
            file_size = processed_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ Processed data: {processed_path} ({file_size:.2f} MB)")
        else:
            logger.info("⚠️  No processed data found")
            logger.info(f"   Run: astro-lab process --surveys {args.survey}")

    except Exception as e:
        logger.error(f"Could not show survey configuration: {e}")
        return 1

    logger.info("✅ Configuration details shown!")
    return 0


def load_and_prepare_training_config(
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Lädt die Trainingskonfiguration, wendet CLI-Overrides an und bereitet die Config für das Training vor.
    
    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei
        preset: Name eines Presets (optional)
        cli_overrides: Dictionary mit CLI-Overrides (optional)
        
    Returns:
        Konsolidiertes Config-Dict für das Training
    """
    config = {}
    
    # Load base configuration
    if config_path:
        loader = ConfigLoader(config_path)
        config = loader.load_config()
    elif preset:
        # Use preset configuration
        from astro_lab.models.lightning import MODEL_PRESETS
        if preset in MODEL_PRESETS:
            config = MODEL_PRESETS[preset].copy()
            # Extract model name from preset config
            if 'model_name' in config:
                config['model'] = config.pop('model_name')
            config['preset'] = preset
        else:
            raise ValueError(f"Unknown preset: {preset}")
    else:
        # Default configuration
        config = {
            "model": "gaia_classifier",
            "dataset": "gaia",
            "max_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
    
    # Apply CLI overrides
    if cli_overrides:
        # Ensure batch_size is converted to int
        if 'batch_size' in cli_overrides:
            cli_overrides['batch_size'] = int(cli_overrides['batch_size'])
        config.update(cli_overrides)
    
    # Ensure required fields
    if 'dataset' not in config and 'survey' in config:
        config['dataset'] = config['survey']
    
    # Ensure batch_size is int
    if 'batch_size' in config:
        config['batch_size'] = int(config['batch_size'])
    
    # Add default values for important parameters
    config.setdefault('experiment_name', 'astrolab_experiment')
    config.setdefault('checkpoint_dir', Path('checkpoints'))
    config.setdefault('num_workers', 4)
    config.setdefault('precision', '16-mixed')
    config.setdefault('optimizer', 'adamw')
    config.setdefault('scheduler', 'cosine')
    config.setdefault('weight_decay', 0.01)
    config.setdefault('gradient_clip_val', 1.0)
    
    return config
