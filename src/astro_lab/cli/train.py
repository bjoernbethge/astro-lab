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
  astro-lab train --preset gaia_fast --epochs 10
  astro-lab train --model survey_gnn --dataset gaia --epochs 2
  astro-lab train --model gaia_classifier --dataset gaia --batch-size 64 --learning-rate 0.001
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
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adam", "sgd", "rmsprop"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "step", "exponential", "onecycle", "none"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["16-mixed", "bf16-mixed", "32-true"],
        help="Training precision (16-mixed recommended for RTX 4070)",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        help="Number of batches to accumulate gradients",
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
        "--num-workers",
        type=int,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to use (for testing)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast development test (1 batch)",
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
        # CLI-Overrides sammeln - erweiterte Liste von Parametern
        cli_overrides = {}
        for key in [
            "model", "dataset", "epochs", "batch_size", "learning_rate",
            "weight_decay", "optimizer", "scheduler", "precision",
            "gradient_clip_val", "accumulate_grad_batches",
            "experiment_name", "checkpoint_dir", "resume",
            "num_workers", "max_samples", "early_stopping_patience",
            "fast_dev_run"
        ]:
            # Convert argument names (e.g., batch-size -> batch_size)
            arg_key = key.replace("_", "-")
            arg_val = getattr(args, key, None)
            
            if arg_val is not None:
                # Special handling for batch_size to ensure it's int
                if key == "batch_size":
                    cli_overrides[key] = int(arg_val)
                else:
                    cli_overrides[key] = arg_val
        
        # Map 'epochs' to 'max_epochs' for consistency
        if "epochs" in cli_overrides:
            cli_overrides["max_epochs"] = cli_overrides.pop("epochs")
        
        # Config laden und vorbereiten
        trainer_config = load_and_prepare_training_config(
            config_path=args.config,
            preset=args.preset,
            cli_overrides=cli_overrides,
        )
        
        # Validate configuration
        if not trainer_config.get("model") and not trainer_config.get("preset"):
            logger.error(
                "Must specify either --model, --preset or --config with model defined."
            )
            parser.print_help()
            return 1
        
        if not trainer_config.get("dataset"):
            logger.warning("No dataset specified, using default 'gaia'")
            trainer_config["dataset"] = "gaia"
        
        # Log final configuration
        logger.info("📋 Final training configuration:")
        for key, value in sorted(trainer_config.items()):
            if key not in ["checkpoint_dir"]:  # Skip paths in log
                logger.info(f"   {key}: {value}")
        
        # Start training
        success = train_model(trainer_config)
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
