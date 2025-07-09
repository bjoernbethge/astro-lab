#!/usr/bin/env python3
"""AstroLab Training CLI - Train astronomical GNN models."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from ..config import (
    get_combined_config,
    get_survey_config,
)
from ..training import train_model


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
        "--task",
        type=str,
        default="node_classification",  # Default task
        help="Task type (default: node_classification)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model type (e.g., gcn, gat, sage, gin, transformer, pointnet, temporal, auto)",
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

    # Ensure task has a default
    task = parsed_args.task or "node_classification"

    # Get combined config (survey + task)
    config = get_combined_config(
        parsed_args.survey,
        task,
    )

    # Ensure survey and task are set in config
    config["survey"] = parsed_args.survey
    config["task"] = task

    # Load custom config file if provided
    if parsed_args.config:
        try:
            with open(parsed_args.config, "r") as f:
                custom_config = yaml.safe_load(f)
                if custom_config:
                    config.update(custom_config)
                    logger.info(f"Loaded custom config from {parsed_args.config}")
        except Exception as e:
            logger.error(f"Failed to load config file {parsed_args.config}: {e}")
            return 1

    # Override config with CLI arguments (CLI has highest priority)
    if parsed_args.model:
        config["conv_type"] = parsed_args.model
    if parsed_args.max_epochs is not None:
        config["max_epochs"] = parsed_args.max_epochs
    if parsed_args.batch_size is not None:
        config["batch_size"] = parsed_args.batch_size
    if parsed_args.learning_rate is not None:
        config["learning_rate"] = parsed_args.learning_rate
    if parsed_args.hidden_dim is not None:
        config["hidden_dim"] = parsed_args.hidden_dim
    if parsed_args.num_layers is not None:
        config["num_layers"] = parsed_args.num_layers

    # Ensure required parameters
    if not config.get("survey"):
        logger.error(
            "No dataset/survey specified. Use --survey or provide in config file."
        )
        return 1

    # Log configuration summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"Survey: {config['survey']}")
    logger.info(f"Task: {config['task']}")

    # Get survey-specific information
    try:
        survey_config = get_survey_config(config["survey"])
        logger.info(f"Survey Name: {survey_config.get('name', 'N/A')}")
        if "recommended_model" in survey_config:
            rec_model = survey_config["recommended_model"]
            logger.info(f"Recommended Model: {rec_model.get('conv_type', 'N/A')}")
    except Exception:
        pass

    logger.info(f"Model Type: {config.get('conv_type', 'auto')}")
    logger.info(f"Batch Size: {config.get('batch_size', 'default')}")
    logger.info(f"Learning Rate: {config.get('learning_rate', 'default')}")
    logger.info(f"Max Epochs: {config.get('max_epochs', 'default')}")
    logger.info("=" * 60 + "\n")

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Starting training...")

        # Pass model_type from config if not explicitly set
        model_type = config.get("conv_type") or config.get("model_type")

        results = train_model(
            survey=config["survey"],
            task=config["task"],
            model_type=model_type,
            config=config,
        )

        # Save results
        results_file = output_dir / f"{config['survey']}_{config['task']}_results.yaml"

        # Prepare results for YAML serialization
        save_results = {
            "survey": config["survey"],
            "task": config["task"],
            "model_type": model_type,
            "config": config,
        }

        if results.get("test_results"):
            test_results = results["test_results"]
            # Extract metrics safely
            save_results["test_results"] = {
                "test_loss": float(test_results.get("test_loss", 0.0)),
                "test_acc": float(test_results.get("test_acc", 0.0)),
                "test_f1": float(test_results.get("test_f1", 0.0)),
            }

        with open(results_file, "w") as f:
            yaml.dump(save_results, f, default_flow_style=False)

        logger.info(f"Training completed! Results saved to {results_file}")

        if results.get("test_results"):
            test_acc = results["test_results"].get("test_acc", 0.0)
            logger.info(f"Test accuracy: {test_acc:.4f}")

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
