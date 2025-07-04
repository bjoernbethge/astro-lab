#!/usr/bin/env python3
"""
AstroLab CLI
===========

Unified command-line interface for AstroLab.

Command structure:
    astro-lab <command> <survey> [options]

Examples:
    astro-lab preprocess gaia
    astro-lab train gaia --epochs 100
    astro-lab info gaia
    astro-lab cosmic-web gaia --visualize
"""

import argparse
import sys
from pathlib import Path

# List of available surveys
AVAILABLE_SURVEYS = [
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
]


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="AstroLab - Astronomical Machine Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    astro-lab preprocess gaia              # Preprocess Gaia data
    astro-lab train gaia --epochs 100      # Train model on Gaia
    astro-lab info gaia                    # Show Gaia data info
    astro-lab cosmic-web gaia --visualize  # Analyze cosmic web
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # ==================== PREPROCESS ====================
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess survey data",
        description="Preprocess raw astronomical survey data into training-ready format",
    )
    preprocess_parser.add_argument(
        "survey",
        choices=AVAILABLE_SURVEYS,
        help="Survey to preprocess",
    )
    preprocess_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if processed data exists",
    )
    preprocess_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process (for testing)",
    )
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory",
    )
    preprocess_parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="knn",
        help="Sampling strategy for ML dataset (knn, radius, fps, cluster, graphsaint, etc.)",
    )
    preprocess_parser.add_argument(
        "--type",
        type=str,
        default="spatial",
        help="Dataset type (spatial, pointcloud, graph, etc.)",
    )
    preprocess_parser.add_argument(
        "--k",
        type=int,
        help="Number of neighbors for KNN sampling",
    )
    preprocess_parser.add_argument(
        "--radius",
        type=float,
        help="Radius for radius-based sampling",
    )
    preprocess_parser.add_argument(
        "--num-subgraphs",
        type=int,
        help="Number of subgraphs for point cloud sampling",
    )
    preprocess_parser.add_argument(
        "--points-per-subgraph",
        type=int,
        help="Points per subgraph for point cloud sampling",
    )
    preprocess_parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for dataset processing (default: 10000)",
    )
    preprocess_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for processing (cpu or cuda, default: cuda)",
    )

    # ==================== TRAIN ====================
    train_parser = subparsers.add_parser(
        "train",
        help="Train model on survey",
        description="Train a Graph Neural Network model on astronomical survey data",
    )
    train_parser.add_argument(
        "survey",
        choices=AVAILABLE_SURVEYS,
        help="Survey to train on",
    )
    train_parser.add_argument(
        "--task",
        choices=[
            "node_classification",
            "graph_classification",
            "node_regression",
            "graph_regression",
        ],
        default="node_classification",
        help="Task type (default: node_classification)",
    )
    train_parser.add_argument(
        "--model",
        choices=["gcn", "gat", "sage", "gin"],
        default="gcn",
        help="Model architecture (default: gcn)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    train_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension (default: 128)",
    )
    train_parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers (default: 3)",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
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
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    train_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to YAML configuration file (overrides other options)",
    )

    # ==================== INFO ====================
    info_parser = subparsers.add_parser(
        "info",
        help="Show survey information",
        description="Display detailed information about survey data",
    )

    # Special case: 'astro-lab info' shows all surveys
    info_parser.add_argument(
        "survey",
        nargs="?",
        choices=AVAILABLE_SURVEYS + ["all"],
        default="all",
        help="Survey to inspect (default: all)",
    )
    info_parser.add_argument(
        "--columns",
        action="store_true",
        help="Show detailed column information",
    )
    info_parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Number of sample rows to show",
    )
    info_parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data validation checks",
    )

    # ==================== COSMIC-WEB ====================
    cosmic_parser = subparsers.add_parser(
        "cosmic-web",
        help="Analyze cosmic web structure",
        description="Analyze cosmic web structure in astronomical surveys",
    )
    cosmic_parser.add_argument(
        "survey",
        choices=["gaia", "nsa", "exoplanet", "sdss", "tng50"],
        help="Survey to analyze",
    )
    cosmic_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of objects to analyze",
    )
    cosmic_parser.add_argument(
        "--clustering-scales",
        nargs="+",
        type=float,
        help="Clustering scales (parsecs for Gaia/exoplanet, Mpc for NSA)",
    )
    cosmic_parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples for DBSCAN clustering (default: 5)",
    )
    cosmic_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations",
    )
    cosmic_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )

    # ==================== HPO ====================
    hpo_parser = subparsers.add_parser(
        "hpo",
        help="Hyperparameter optimization",
        description="Run hyperparameter optimization for a survey",
    )
    hpo_parser.add_argument(
        "survey",
        choices=AVAILABLE_SURVEYS,
        help="Survey to optimize",
    )
    hpo_parser.add_argument(
        "--task",
        choices=[
            "node_classification",
            "graph_classification",
            "node_regression",
            "graph_regression",
        ],
        default="node_classification",
        help="Task type (default: node_classification)",
    )
    hpo_parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    hpo_parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum epochs per trial (default: 20)",
    )
    hpo_parser.add_argument(
        "--timeout",
        type=int,
        help="Optimization timeout in seconds",
    )

    # ==================== CONFIG ====================
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        description="Show or create configuration files",
    )
    config_parser.add_argument(
        "action",
        choices=["show", "create"],
        help="Action to perform",
    )
    config_parser.add_argument(
        "survey",
        nargs="?",
        choices=AVAILABLE_SURVEYS,
        help="Survey configuration",
    )
    config_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file for 'create' action",
    )

    # ==================== DOWNLOAD ====================
    download_parser = subparsers.add_parser(
        "download",
        help="Download survey data",
        description="Download raw survey data from official sources",
    )
    download_parser.add_argument(
        "survey",
        choices=AVAILABLE_SURVEYS,
        help="Survey to download",
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory",
    )

    # ==================== BUILD-DATASET ====================
    build_dataset_parser = subparsers.add_parser(
        "build-dataset",
        help="Build ML-ready dataset from harmonized data",
        description="Convert harmonized .parquet to ML-ready .pt dataset (PyG/TensorDict)",
    )
    build_dataset_parser.add_argument(
        "survey",
        choices=AVAILABLE_SURVEYS,
        help="Survey to build dataset for",
    )
    build_dataset_parser.add_argument(
        "--type",
        choices=["spatial", "photometric", "temporal"],
        default="spatial",
        help="Dataset type (default: spatial)",
    )
    build_dataset_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if .pt exists",
    )

    # Global options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 for Windows compatibility)",
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
        # Route to appropriate command handler
        if args.command == "preprocess":
            from .preprocess import main

            return main(args)

        elif args.command == "train":
            from .train import main

            return main(args)

        elif args.command == "info":
            from .info import main

            return main(args)

        elif args.command == "cosmic-web":
            from .cosmic_web import main

            return main(args)

        elif args.command == "hpo":
            from .hpo import main

            return main(args)

        elif args.command == "config":
            from .config import main

            return main(args)

        elif args.command == "download":
            from .download import main

            return main(args)

        elif args.command == "build-dataset":
            # Build ML-ready dataset from harmonized .parquet
            import os

            from astro_lab.data.dataset.astrolab import create_dataset

            survey = args.survey
            dtype = args.type
            force = getattr(args, "force", False)
            dataset = create_dataset(survey_name=survey, data_type=dtype)
            pt_path = dataset.processed_paths[0]
            if not os.path.exists(pt_path) or force:
                print(f"[AstroLab] Building ML dataset for {survey} ({dtype})...")
                dataset.process()
                print(f"[AstroLab] Dataset built and saved to {pt_path}")
            else:
                print(
                    f"[AstroLab] ML dataset already exists at {pt_path}. Use --force to overwrite."
                )
            return 0

        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        return 1
    except Exception as e:
        if args.verbose:
            import traceback

            traceback.print_exc()
        print(f"❌ Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
