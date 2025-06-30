#!/usr/bin/env python3
"""
AstroLab CLI
===========

Main command-line interface for AstroLab.
"""

import argparse
import sys
from pathlib import Path

from .preprocess import add_preprocess_arguments


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="AstroLab - Astronomical Machine Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        description="Process and prepare astronomical datasets for training",
    )
    process_parser.add_argument(
        "--surveys",
        nargs="+",
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
        ],
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

    # Preprocess command - simplified for raw data preprocessing
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess raw data files",
        description="Preprocess raw astronomical data files into training-ready format",
    )
    add_preprocess_arguments(preprocess_parser)

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
        ],
        help="Dataset/Survey name",
    )
    train_parser.add_argument(
        "--model-type",
        choices=["node", "graph", "temporal", "point"],
        help="Model type (node, graph, temporal, point)",
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
        "--overfit-batches",
        type=float,
        help="Overfit on a few batches for testing",
    )
    train_parser.add_argument(
        "--num-features",
        type=int,
        help="Number of input features for the model (overrides dataset default)",
    )
    train_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    # HPO command
    hpo_parser = subparsers.add_parser(
        "hpo",
        help="Hyperparameter optimization",
        description="Run hyperparameter optimization for ML models",
    )
    hpo_parser.add_argument(
        "--dataset",
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
        ],
        required=True,
        help="Dataset to use",
    )
    hpo_parser.add_argument(
        "--model-type",
        choices=["node", "graph", "temporal", "point"],
        required=True,
        help="Model type to optimize",
    )
    hpo_parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    hpo_parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum epochs per trial",
    )
    hpo_parser.add_argument(
        "--timeout",
        type=int,
        help="Optimization timeout in seconds",
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
        ],
        default="gaia",
        help="Configuration template",
    )

    # Config surveys
    config_subparsers.add_parser(
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
        ],
        help="Survey to show",
    )

    # Cosmic Web command
    cosmic_web_parser = subparsers.add_parser(
        "cosmic-web",
        help="Analyze cosmic web structure",
        description="Analyze cosmic web structure in astronomical surveys",
    )
    cosmic_web_parser.add_argument(
        "survey",
        choices=["gaia", "nsa", "exoplanet", "sdss", "tng50"],
        help="Survey to analyze",
    )
    cosmic_web_parser.add_argument(
        "--catalog-path",
        type=Path,
        help="Path to catalog file (uses default if not specified)",
    )
    cosmic_web_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of objects to analyze",
    )
    cosmic_web_parser.add_argument(
        "--clustering-scales",
        nargs="+",
        type=float,
        help="Clustering scales (parsecs for Gaia/exoplanet, Mpc for NSA)",
    )
    cosmic_web_parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples for DBSCAN clustering (default: 5)",
    )
    cosmic_web_parser.add_argument(
        "--magnitude-limit",
        type=float,
        default=12.0,
        help="Magnitude limit for Gaia (default: 12.0)",
    )
    cosmic_web_parser.add_argument(
        "--redshift-limit",
        type=float,
        default=0.15,
        help="Redshift limit for NSA (default: 0.15)",
    )
    cosmic_web_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )
    cosmic_web_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations",
    )
    cosmic_web_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
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
        elif args.command == "hpo":
            from .hpo import main as hpo_main

            return hpo_main(args)
        elif args.command == "config":
            from .config import main as config_main

            return config_main(args)
        elif args.command == "cosmic-web":
            from .cosmic_web import main as cosmic_web_main

            return cosmic_web_main(args)
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
