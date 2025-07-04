#!/usr/bin/env python3
"""AstroLab Training CLI - Train astronomical GNN models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..config import (
    get_combined_config,
    # get_data_config,  # removed unused
    # get_data_paths,   # removed unused
    # get_model_config, # removed unused
    # get_training_config, # removed unused
)
from ..models import AstroModel
from ..training import AstroTrainer, train_model


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training CLI."""
    parser = argparse.ArgumentParser(
        description="Train a GNN model on astronomical survey data.\n\nMost configuration (model, data, training, callbacks, logging, etc.) is controlled via YAML config files. Only a few essential overrides are available as CLI arguments.",
        epilog="""
YAML-centric workflow example:
  astro-lab train gaia
  astro-lab train gaia --task node_classification
  astro-lab train gaia -c configs/training.yaml
  astro-lab train gaia --checkpoint last.ckpt

Arguments:
  survey                Survey to train on (e.g. gaia, sdss, nsa, ...)
  --task                Task type (overrides config, e.g. node_classification)
  -c, --config          Path to YAML configuration file
  --checkpoint          Checkpoint to resume from
  -v, --verbose         Enable verbose logging

All other settings (model architecture, hyperparameters, callbacks, logging, etc.) must be set in the YAML config files.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "survey",
        choices=[
            "gaia",
            "sdss",
            "nsa",
            "tng50",
            "exoplanet",
            "twomass",
            "wise",
            "panstarrs",
            "des",
            "euclid",
            "linear",
            "rrlyrae",
        ],
        help="Survey to train on",
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Task type (overrides config)"
    )
    parser.add_argument(
        "--model-type", type=str, default=None, help="Model type (e.g., gcn, gat, sage)"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Number of GNN layers"
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to YAML configuration file"
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume from")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser


def main(args=None) -> int:
    """Main training function."""
    parser = create_parser()
    if isinstance(args, argparse.Namespace):
        parsed_args = args
    else:
        parsed_args = parser.parse_args(args)
    logger = setup_logging(parsed_args.verbose)

    # Use get_combined_config to build a flat config
    config = get_combined_config(
        parsed_args.survey,
        parsed_args.task,
    )

    # Ensure survey is set in config
    config["survey"] = parsed_args.survey

    # Override config with CLI arguments
    if hasattr(parsed_args, "model") and parsed_args.model:
        config["conv_type"] = parsed_args.model
    if hasattr(parsed_args, "max_epochs") and parsed_args.max_epochs:
        config["max_epochs"] = parsed_args.max_epochs
    if hasattr(parsed_args, "batch_size") and parsed_args.batch_size:
        config["batch_size"] = parsed_args.batch_size
    if hasattr(parsed_args, "learning_rate") and parsed_args.learning_rate:
        config["learning_rate"] = parsed_args.learning_rate
    if hasattr(parsed_args, "hidden_dim") and parsed_args.hidden_dim:
        config["hidden_dim"] = parsed_args.hidden_dim
    if hasattr(parsed_args, "num_layers") and parsed_args.num_layers:
        config["num_layers"] = parsed_args.num_layers

    # Ensure required parameters
    if not config.get("survey"):
        logger.error(
            "No dataset/survey specified. Use --survey or provide in config file."
        )
        return 1

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting training...")
        results = train_model(
            survey=config["survey"],
            task=config.get("task") or "node_classification",
            config=config,
        )

        # Save results
        results_file = output_dir / "results.yaml"
        with open(results_file, "w") as f:
            yaml.dump(results, f, default_flow_style=False)

        logger.info(f"Training completed! Results saved to {results_file}")
        if results["test_results"]:
            logger.info(
                f"Test accuracy: {results['test_results'].get('test_metric', 0):.4f}"
            )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if parsed_args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
