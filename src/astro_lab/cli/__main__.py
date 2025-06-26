#!/usr/bin/env python3
"""
AstroLab CLI
===========

Main command-line interface for AstroLab.
"""

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="AstroLab - Astronomical Machine Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Data processing
  astro-lab process --surveys gaia nsa --max-samples 10000
  astro-lab preprocess catalog data/gaia_catalog.parquet --config gaia

  # Training
  astro-lab train --dataset gaia --model gaia_classifier --epochs 50
  astro-lab train -c my_experiment.yaml

  # Optimization
  astro-lab optimize my_experiment.yaml --trials 50

  # Configuration
  astro-lab config create -o my_experiment.yaml
  astro-lab config surveys
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process astronomical data",
        description="Process and prepare astronomical datasets",
    )
    process_parser.add_argument(
        "--surveys",
        nargs="+",
        choices=["gaia", "sdss", "nsa", "tng50", "exoplanet", "rrlyrae", "linear"],
        default=["gaia"],
        help="Surveys to process",
    )
    process_parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="Number of neighbors for graph construction",
    )
    process_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process",
    )

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess data files",
        description="Preprocess specific data files",
    )
    preprocess_subparsers = preprocess_parser.add_subparsers(
        dest="preprocess_command",
        help="Preprocessing operations",
    )

    # Preprocess catalog
    catalog_parser = preprocess_subparsers.add_parser(
        "catalog",
        help="Preprocess catalog data",
    )
    catalog_parser.add_argument(
        "file",
        type=Path,
        help="Input catalog file",
    )
    catalog_parser.add_argument(
        "--config",
        type=str,
        help="Configuration to use",
    )
    catalog_parser.add_argument(
        "--splits",
        action="store_true",
        help="Create train/val/test splits",
    )

    # Preprocess stats
    stats_parser = preprocess_subparsers.add_parser(
        "stats",
        help="Show data statistics",
    )
    stats_parser.add_argument(
        "file",
        type=Path,
        help="Input data file",
    )

    # Preprocess browse
    browse_parser = preprocess_subparsers.add_parser(
        "browse",
        help="Browse survey data",
    )
    browse_parser.add_argument(
        "--survey",
        choices=["gaia", "sdss", "nsa", "tng50", "exoplanet", "rrlyrae", "linear"],
        required=True,
        help="Survey to browse",
    )
    browse_parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed information",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train ML models",
        description="Train astronomical ML models",
    )
    train_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    train_parser.add_argument(
        "--dataset",
        choices=["gaia", "sdss", "nsa", "tng50", "exoplanet", "rrlyrae", "linear"],
        help="Dataset/Survey name",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g. gaia_classifier, sdss_galaxy)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    train_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples for debugging",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to resume from",
    )
    train_parser.add_argument(
        "--devices",
        type=int,
        help="Number of GPUs to use",
    )
    train_parser.add_argument(
        "--precision",
        choices=["32", "16", "bf16", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    train_parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a quick development test",
    )
    train_parser.add_argument(
        "--overfit-batches",
        type=float,
        help="Overfit on a few batches for testing",
    )

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize hyperparameters",
        description="Optimize hyperparameters for ML models",
    )
    optimize_parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    optimize_parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of optimization trials",
    )
    optimize_parser.add_argument(
        "--timeout",
        type=int,
        help="Optimization timeout in seconds",
    )
    optimize_parser.add_argument(
        "--algorithm",
        choices=["optuna", "ray", "grid", "random"],
        default="optuna",
        help="Optimization algorithm",
    )
    optimize_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples for debugging",
    )
    optimize_parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a quick development test",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        description="Manage AstroLab configurations",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        help="Configuration operations",
    )

    # Config create
    create_parser = config_subparsers.add_parser(
        "create",
        help="Create configuration file",
    )
    create_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output configuration file",
    )
    create_parser.add_argument(
        "--template",
        choices=["gaia", "sdss", "nsa", "tng50", "exoplanet", "rrlyrae", "linear"],
        default="gaia",
        help="Configuration template",
    )

    # Config surveys
    surveys_parser = config_subparsers.add_parser(
        "surveys",
        help="Show available survey configurations",
    )

    # Config show
    show_parser = config_subparsers.add_parser(
        "show",
        help="Show configuration details",
    )
    show_parser.add_argument(
        "survey",
        choices=["gaia", "sdss", "nsa", "tng50", "exoplanet", "rrlyrae", "linear"],
        help="Survey to show",
    )

    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "process":
            from .process import main as process_main

            return process_main(args)
        elif args.command == "preprocess":
            from .preprocess import main as preprocess_main

            return preprocess_main(args)
        elif args.command == "train":
            from .train import main as train_main

            return train_main(args)
        elif args.command == "optimize":
            from .optimize import main as optimize_main

            return optimize_main(args)
        elif args.command == "config":
            from .config import main as config_main

            return config_main(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
