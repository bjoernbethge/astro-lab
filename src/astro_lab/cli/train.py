#!/usr/bin/env python3
"""
AstroLab Training CLI (Lightning Edition)
========================================

CLI for training astronomical ML models using Lightning.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from astro_lab.models.lightning import list_lightning_models, list_presets
from astro_lab.training import train_model

from .config import load_and_prepare_training_config


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for training CLI."""
    available_models = list(list_lightning_models().keys())
    available_presets = list(list_presets().keys())

    parser = argparse.ArgumentParser(
        description="Train astronomical ML models with AstroLab Lightning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  astro-lab train --config myexp.yaml
  astro-lab train --preset gaia_classifier --epochs 10 --learning-rate 0.001
  astro-lab train --model survey_gnn --epochs 2 --dataset gaia
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (overrides all other options except explicit CLI overrides)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=available_presets,
        help="Use preset configuration (can be overridden by CLI parameters)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=available_models,
        help="Model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gaia", "sdss", "nsa", "tng50", "exoplanet", "rrlyrae", "linear"],
        help="Survey/dataset to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser


def main(args=None) -> int:
    """Main CLI function - parses arguments and calls AstroTrainer."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    logger = setup_logging(args.verbose)

    try:
        # CLI-Overrides sammeln
        cli_overrides = {}
        for key in [
            "model",
            "dataset",
            "epochs",
            "batch_size",
            "learning_rate",
            "experiment_name",
            "checkpoint_dir",
            "resume",
        ]:
            arg_val = getattr(args, key, None)
            if arg_val is not None:
                cli_overrides[key] = arg_val
        # Config laden und vorbereiten
        trainer_config = load_and_prepare_training_config(
            config_path=args.config,
            preset=args.preset,
            cli_overrides=cli_overrides,
        )
        if not trainer_config.get("model") and not args.preset:
            logger.error(
                "Must specify either --model, --preset or --config with model defined."
            )
            return 1
        success = train_model(trainer_config)
        return 0 if success else 1

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
