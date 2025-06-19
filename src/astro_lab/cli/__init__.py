"""
AstroLab CLI Module
==================

Command-line interfaces for AstroLab:
- Download: Download astronomical datasets
- Preprocessing: Data processing and train/val/test splits
- Training: ML model training with Lightning + MLflow
"""

# Suppress NumPy warnings before any other imports
import os
import warnings
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*") 
warnings.filterwarnings("ignore", message=".*numpy.core.multiarray.*")
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:numpy'

import argparse
import sys

from astro_lab.cli.download import main as download_main
from astro_lab.cli.preprocessing import main as preprocessing_main
from astro_lab.cli.train import main as train_main

__all__ = [
    "download_main",
    "preprocessing_main", 
    "train_main",
    "main",
]


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="AstroLab - Astronomical Machine Learning Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 Available Commands:

astro-lab download    Download astronomical datasets
astro-lab preprocess  Data preprocessing and train/val/test splits  
astro-lab train       Train ML models with Lightning + MLflow

📖 Examples:

# Download Gaia DR3 bright stars
astro-lab download gaia --magnitude-limit 12.0

# List available datasets
astro-lab download list

# Show preprocessing functions
astro-lab preprocess --show-functions

# Process TNG50 simulation data
astro-lab preprocess tng50 data/raw/TNG50-4/output/snapdir_099/snap_099.0.hdf5 --output data/processed/tng50/

# List TNG50 snapshots
astro-lab preprocess tng50-list --inspect

# Train a model
astro-lab train create-config --output config.yaml
astro-lab train train --config config.yaml

# Quick training without config
astro-lab train train --dataset gaia --model gaia_classifier --epochs 50

💡 Use 'astro-lab <command> --help' for detailed options!
        """,
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="AstroLab 0.1.0"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download subcommand
    download_parser = subparsers.add_parser("download", help="Download astronomical datasets")
    download_subparsers = download_parser.add_subparsers(dest="download_action", help="Download actions")
    
    # Download gaia
    gaia_parser = download_subparsers.add_parser("gaia", help="Download Gaia DR3 data")
    gaia_parser.add_argument(
        "--magnitude-limit",
        type=float,
        default=12.0,
        help="Magnitude limit for Gaia stars (default: 12.0)",
    )
    
    # Download list
    download_subparsers.add_parser("list", help="List available datasets")
    
    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", help="Data preprocessing")
    preprocess_parser.add_argument(
        "--show-functions",
        action="store_true",
        help="Show available data module functions"
    )
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_subparsers = train_parser.add_subparsers(dest="train_action", help="Training actions")
    
    # Train create-config
    config_parser = train_subparsers.add_parser("create-config", help="Create default configuration file")
    config_parser.add_argument("--output", "-o", default="config.yaml", help="Output configuration file")
    
    # Train train
    train_train_parser = train_subparsers.add_parser("train", help="Train model")
    train_train_parser.add_argument("--config", "-c", help="Configuration file path")
    train_train_parser.add_argument("--dataset", choices=["gaia", "sdss", "nsa"], help="Dataset to use")
    train_train_parser.add_argument("--model", choices=["gaia_classifier", "sdss_galaxy_classifier", "lsst_transient_detector"], help="Model type")
    train_train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    train_train_parser.add_argument("--experiment-name", default="quick_train", help="Experiment name")
    
    # Train optimize
    optimize_parser = train_subparsers.add_parser("optimize", help="Optimize hyperparameters")
    optimize_parser.add_argument("--config", "-c", required=True, help="Configuration file path")

    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return

    # Route to appropriate subcommand
    if args.command == "download":
        if not args.download_action:
            download_parser.print_help()
            return
            
        # Import download functions here to avoid circular imports
        from astro_lab.data import download_bright_all_sky, list_catalogs
        
        if args.download_action == "gaia":
            print(f"🌟 Downloading Gaia DR3 data (mag < {args.magnitude_limit})")
            try:
                result = download_bright_all_sky(magnitude_limit=args.magnitude_limit)
                print(f"✅ Success: {result}")
            except Exception as e:
                print(f"❌ Error: {e}")
                sys.exit(1)
                
        elif args.download_action == "list":
            print("📋 Available datasets:")
            try:
                catalogs = list_catalogs()
                print(catalogs)
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.command == "preprocess":
        if args.show_functions:
            print("📦 Available astro_lab.data functions:")
            print()
            print("🔧 Preprocessing:")
            print("  • preprocess_catalog(df)")
            print("  • create_training_splits(df)")
            print("  • save_splits_to_parquet(train, val, test, path, name)")
            print("  • load_splits_from_parquet(path, name)")
            print("  • get_data_statistics(df)")
            print()
            print("🌌 TNG50 Simulation:")
            print("  • TNG50Loader.load_snapshot()")
            print("  • process_tng50_command()")
            print("  • list_tng50_snapshots_command()")
            print()
            print("📊 Datasets (PyTorch Geometric):")
            print("  • create_exoplanet_dataloader()")
            print("  • create_gaia_dataloader()")
            print("  • create_nsa_dataloader()")
            print("  • create_linear_lightcurve_dataloader()")
            print("  • create_rrlyrae_dataloader()")
            print("  • create_tng50_dataloader()")
            print()
            print("📁 Data Management:")
            print("  • AstroDataManager()")
            print("  • download_bright_all_sky()")
            print("  • list_catalogs()")
            print()
            print("💡 See astro_lab.data.__init__.py for all functions!")
            print()
            print("🚀 TNG50 CLI Examples:")
            print("  astro-lab preprocess tng50 data/raw/TNG50-4/output/snapdir_099/snap_099.0.hdf5 --output data/processed/")
            print("  astro-lab preprocess tng50-list --inspect")
            print("  astro-lab preprocess tng50 snap.hdf5 --particle-types PartType4,PartType5 --max-particles 5000")
        else:
            # Route to preprocessing CLI
            preprocessing_main()
    
    elif args.command == "train":
        if not args.train_action:
            train_parser.print_help()
            return
            
        # Import train functions
        from .train import create_default_config, train_from_config, optimize_from_config
        from pathlib import Path
        import yaml
        
        if args.train_action == "create-config":
            create_default_config(args.output)
            
        elif args.train_action == "train":
            if args.config:
                # Train from config file
                train_from_config(args.config)
            else:
                # Quick train mode
                if not args.dataset or not args.model:
                    print("❌ For quick train, --dataset and --model are required")
                    return

                # Create temporary config
                quick_config = {
                    "model": {"type": args.model, "params": {}},
                    "data": {
                        "dataset": args.dataset,
                        "batch_size": args.batch_size,
                        "num_workers": 4,
                    },
                    "training": {
                        "max_epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "experiment_name": args.experiment_name,
                    },
                }

                # Save temporary config
                temp_config_path = "temp_config.yaml"
                with open(temp_config_path, "w") as f:
                    yaml.dump(quick_config, f)

                try:
                    train_from_config(temp_config_path)
                finally:
                    Path(temp_config_path).unlink(missing_ok=True)
                    
        elif args.train_action == "optimize":
            optimize_from_config(args.config)


if __name__ == "__main__":
    main()
