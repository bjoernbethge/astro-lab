"""
AstroLab CLI - Unified Command Line Interface
============================================

Modern CLI for astronomical data processing and ML training.
Supports multiple survey types with unified preprocessing and training pipelines.
"""

import argparse
import datetime
import json
import os
import sys
import traceback
import warnings
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click

from astro_lab.data import (
    AstroDataset,
    create_astro_datamodule,
    data_config,
    load_gaia_data,
    load_nsa_data,
    load_sdss_data,
    load_tng50_data,
)
from astro_lab.models.factory import ModelFactory
from astro_lab.training.trainer import AstroTrainer
from astro_lab.utils.config.loader import ConfigLoader

__all__ = [
    "main",
]


def main():
    """Main CLI entry point with streamlined interface."""
    # Welcome message
    print("⭐ Welcome to AstroLab - Astronomical Machine Learning Laboratory!")
    print()
    
    parser = argparse.ArgumentParser(
        description="AstroLab - Astronomical Machine Learning Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 Available Commands:

astro-lab download       Download astronomical datasets
astro-lab preprocess     Datenvorverarbeitung und Graph-Erstellung
astro-lab train          ML-Model Training mit Lightning + MLflow
astro-lab optimize       Hyperparameter-Optimierung mit Optuna
astro-lab config         Konfigurationsverwaltung

📖 Beispiele:

# Download Gaia DR3 bright stars
astro-lab download gaia --magnitude-limit 12.0

# Preprocess catalog with survey config
astro-lab preprocess catalog data/gaia_catalog.parquet --config gaia --splits --output data/processed/

# Show catalog statistics
astro-lab preprocess stats data/gaia_catalog.parquet

# Process TNG50 simulation (uses full CLI)
astro-lab preprocess tng50 data/snap_099.0.hdf5 --particle-types PartType4 --output data/tng50/

# Create and edit default config
astro-lab config create --output my_config.yaml
# Then edit and train:
astro-lab train --config my_config.yaml

# Quick training without config
astro-lab train --dataset gaia --model gaia_classifier --epochs 50

# Survey-spezifische Configs anzeigen
astro-lab config surveys

💡 Use 'astro-lab <command> --help' for detailed options!
        """,
    )

    parser.add_argument("--version", action="version", version="AstroLab 0.1.0")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download subcommand
    download_parser = subparsers.add_parser(
        "download", help="Download astronomical datasets"
    )
    download_subparsers = download_parser.add_subparsers(
        dest="download_action", help="Download actions"
    )

    gaia_parser = download_subparsers.add_parser("gaia", help="Download Gaia DR3 data")
    gaia_parser.add_argument(
        "--magnitude-limit",
        type=float,
        default=12.0,
        help="Magnitude limit for Gaia stars (default: 12.0)",
    )

    download_subparsers.add_parser("list", help="List available datasets")

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Data preprocessing and graph creation"
    )
    preprocess_subparsers = preprocess_parser.add_subparsers(
        dest="preprocess_action", help="Preprocessing actions"
    )

    # Quick catalog processing
    catalog_parser = preprocess_subparsers.add_parser(
        "catalog", help="Process astronomical catalog"
    )
    catalog_parser.add_argument("input", help="Input catalog file")
    catalog_parser.add_argument("--output", "-o", help="Output directory")
    catalog_parser.add_argument(
        "--config", "-c", help="Survey configuration (gaia, sdss, etc.)"
    )
    catalog_parser.add_argument(
        "--splits", action="store_true", help="Create train/val/test splits"
    )

    # TNG50 processing
    tng50_parser = preprocess_subparsers.add_parser(
        "tng50", help="Process TNG50 simulation"
    )
    tng50_parser.add_argument("input", help="TNG50 snapshot file")
    tng50_parser.add_argument("--output", "-o", help="Output directory")
    tng50_parser.add_argument(
        "--particle-types", default="PartType4", help="Particle types to process"
    )

    # Stats
    stats_parser = preprocess_subparsers.add_parser(
        "stats", help="Show catalog statistics"
    )
    stats_parser.add_argument("input", help="Input catalog file")

    # Browse raw data
    browse_parser = preprocess_subparsers.add_parser(
        "browse", help="Browse raw data directory"
    )
    browse_parser.add_argument(
        "--path", default="data/raw", help="Path to browse (default: data/raw)"
    )
    browse_parser.add_argument(
        "--survey", help="Browse specific survey (gaia, sdss, etc.)"
    )
    browse_parser.add_argument(
        "--details", action="store_true", help="Show detailed file information"
    )

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument("--config", "-c", help="Configuration file path")

    # Optimize subcommand (dedicated)
    optimize_parser = subparsers.add_parser(
        "optimize", help="Run hyperparameter optimization"
    )
    optimize_parser.add_argument("config", help="Configuration file path")
    optimize_parser.add_argument(
        "--trials", type=int, default=10, help="Number of optimization trials"
    )
    optimize_parser.add_argument("--experiment-name", help="Override experiment name")
    train_parser.add_argument(
        "--dataset", choices=["gaia", "sdss", "nsa"], help="Dataset to use"
    )
    train_parser.add_argument(
        "--model",
        choices=[
            "gaia_classifier",
            "sdss_galaxy_classifier",
            "lsst_transient_detector",
        ],
        help="Model type",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    train_parser.add_argument(
        "--experiment-name", default="quick_train", help="Experiment name"
    )

    # Config subcommand
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(
        dest="config_action", help="Config actions"
    )

    create_parser = config_subparsers.add_parser(
        "create", help="Create default configuration file"
    )
    create_parser.add_argument(
        "--output", "-o", default="config.yaml", help="Output configuration file"
    )

    config_subparsers.add_parser("surveys", help="List available survey configurations")

    show_parser = config_subparsers.add_parser("show", help="Show survey configuration")
    show_parser.add_argument("survey", help="Survey name (e.g., gaia, sdss)")

    # 🌟 NEW: Model config subcommand
    model_parser = config_subparsers.add_parser("model", help="Model configuration management")
    model_parser.add_argument("action", choices=["list", "show", "create"], help="Model config action")
    model_parser.add_argument("--name", help="Model config name")
    model_parser.add_argument("--output", "-o", help="Output file for create action")

    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate command handler
    if args.command == "download":
        handle_download(args)
    elif args.command == "preprocess":
        handle_preprocess(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "optimize":
        handle_optimize(args)
    elif args.command == "config":
        handle_config(args)


def handle_download(args):
    """Handle download command."""
    if not args.download_action:
        print("❌ Download action required. Use --help for options.")
        return

    try:
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
    except Exception as e:
        print(f"❌ Error in download command: {e}")
        sys.exit(1)


def handle_preprocess(args):
    """Handle preprocess command with config-based approach."""
    if not args.preprocess_action:
        print("❌ Preprocessing action required. Use --help for options.")
        return

    try:
        from pathlib import Path

        from astro_lab.data import (
            create_graph_datasets_from_splits,
            create_training_splits,
            get_data_statistics,
            load_catalog,
            preprocess_catalog,
            save_splits_to_parquet,
        )
        from astro_lab.utils.config import ConfigLoader

        if args.preprocess_action == "catalog":
            print(f"📂 Processing catalog: {args.input}")

            # Load catalog
            try:
                df = load_catalog(args.input)
                print(f"📊 Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"❌ Error loading catalog: {e}")
                return

            # Load survey config if specified
            config = {}
            if args.config:
                try:
                    loader = ConfigLoader()
                    survey_config = loader.get_survey_config(args.config)
                    config = survey_config.get("processing", {})
                    print(f"✅ Using {args.config} survey configuration")
                except Exception as e:
                    print(f"⚠️  Could not load survey config: {e}")

            # Preprocess with config
            df_clean = preprocess_catalog(df, **config)
            print(f"✅ Processed: {df_clean.shape[0]:,} rows retained")

            # Create splits if requested
            if args.splits:
                print("🔄 Creating training splits...")
                train, val, test = create_training_splits(df_clean)

                if args.output:
                    output_path = Path(args.output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    dataset_name = Path(args.input).stem

                    save_splits_to_parquet(train, val, test, output_path, dataset_name)
                    create_graph_datasets_from_splits(
                        train, val, test, output_path, dataset_name
                    )
                    print(f"💾 Splits and graphs saved to: {output_path}")

            elif args.output:
                # Save processed catalog
                output_path = Path(args.output)
                if output_path.is_dir():
                    output_file = output_path / f"{Path(args.input).stem}_processed.parquet"
                else:
                    output_file = output_path

                df_clean.write_parquet(output_file)
                print(f"💾 Processed catalog saved to: {output_file}")

        elif args.preprocess_action == "stats":
            print(f"📊 Analyzing: {args.input}")
            try:
                df = load_catalog(args.input)
                stats = get_data_statistics(df)
                print("📈 Dataset Statistics:")
                print(f"  • Rows: {stats['n_rows']:,}")
                print(f"  • Columns: {stats['n_columns']}")
                print(f"  • Memory: {stats['memory_usage_mb']:.1f} MB")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif args.preprocess_action == "browse":
            base_path = Path(args.path)
            if not base_path.exists():
                print(f"❌ Path not found: {base_path}")
                return

            if args.survey:
                # Browse specific survey
                survey_path = base_path / args.survey
                if not survey_path.exists():
                    print(f"❌ Survey path not found: {survey_path}")
                    return
                browse_path = survey_path
            else:
                browse_path = base_path

            print(f"📂 Browsing: {browse_path}")
            print("=" * 50)

            try:
                items = list(browse_path.iterdir())
                dirs = [item for item in items if item.is_dir()]
                files = [item for item in items if item.is_file()]

                # Show directories first
                if dirs:
                    print("📁 Directories:")
                    for dir_item in sorted(dirs):
                        try:
                            subdir_count = len(
                                [x for x in dir_item.iterdir() if x.is_dir()]
                            )
                            file_count = len([x for x in dir_item.iterdir() if x.is_file()])
                            print(
                                f"   📁 {dir_item.name}/ ({file_count} files, {subdir_count} subdirs)"
                            )
                        except PermissionError:
                            print(f"   📁 {dir_item.name}/ (access denied)")

                # Show files
                if files:
                    print("\n📄 Files:")
                    total_size = 0
                    file_info_list = []

                    # Collect all file information first
                    for file_item in sorted(files):
                        try:
                            size_mb = file_item.stat().st_size / (1024 * 1024)
                            total_size += size_mb

                            if args.details:
                                modified = file_item.stat().st_mtime
                                mod_time = datetime.datetime.fromtimestamp(
                                    modified
                                ).strftime("%Y-%m-%d %H:%M")
                                file_info_list.append(
                                    f"   📄 {file_item.name} ({size_mb:.1f} MB, {mod_time})"
                                )
                            else:
                                file_info_list.append(
                                    f"   📄 {file_item.name} ({size_mb:.1f} MB)"
                                )
                        except (OSError, PermissionError) as e:
                            file_info_list.append(
                                f"   📄 {file_item.name} (error reading file: {e})"
                            )

                    # Print all file information at once
                    for file_info in file_info_list:
                        print(file_info)

                    print(f"\n📊 Total: {len(files)} files, {total_size:.1f} MB")

                if not dirs and not files:
                    print("   (empty directory)")

            except Exception as e:
                print(f"❌ Error browsing directory: {e}")

        elif args.preprocess_action == "tng50":
            print(f"🌌 Processing TNG50: {args.input}")
            print("💡 TNG50 processing requires the full preprocessing CLI:")
            print(f"   python -m astro_lab.cli.preprocessing tng50 {args.input}")
            if args.output:
                print(f"   --output {args.output}")
            if args.particle_types != "PartType4":
                print(f"   --particle-types {args.particle_types}")

        else:
            print(f"❌ Unknown preprocessing action: {args.preprocess_action}")
    except Exception as e:
        print(f"❌ Error in preprocess command: {e}")
        sys.exit(1)


def handle_train(args):
    """Handle train command with new unified architecture."""
    from .train import create_default_config, optimize_from_config, train_from_config

    if args.config:
        # Config-based training (preferred method)
        if not Path(args.config).exists():
            print(f"❌ Configuration file not found: {args.config}")
            return

        try:
            print(f"🚀 Starting training with: {args.config}")
            train_from_config(args.config)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            sys.exit(1)

    else:
        # Quick train mode (requires dataset and model)
        if not args.dataset or not args.model:
            print("❌ For quick train, --dataset and --model are required")
            print("💡 Or use --config for full configuration support")
            return

        # Create temporary config for quick training using new architecture
        from astro_lab.models.config import ModelConfig, EncoderConfig, GraphConfig, OutputConfig
        
        # Create model config using new structure
        model_config = ModelConfig(
            name=f"{args.model}_quick",
            description=f"Quick training config for {args.model}",
            encoder=EncoderConfig(
                use_photometry=True,
                use_astrometry=True,
                use_spectroscopy=False,
            ),
            graph=GraphConfig(
                hidden_dim=128,
                num_layers=3,
                dropout=0.1,
            ),
            output=OutputConfig(
                task="node_classification",
                output_dim=1,
            ),
        )

        quick_config = {
            "model": {
                "type": args.model,
                "config": model_config.dict(),
                "use_tensors": True,
            },
            "data": {
                "dataset": args.dataset,
                "batch_size": args.batch_size,
                "max_samples": 5000,
                "return_tensor": True,
            },
            "training": {
                "max_epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "experiment_name": args.experiment_name,
            },
            "mlflow": {
                "experiment_name": args.experiment_name,
                "tracking_uri": "file:./mlruns",
            },
            "checkpoints": {"dir": "./checkpoints"},
        }

        # Save temporary config
        temp_config_path = "temp_quick_config.yaml"
        try:
            with open(temp_config_path, "w") as f:
                yaml.safe_dump(quick_config, f, default_flow_style=False, indent=2)
        except (AttributeError, ImportError):
            # Fallback if yaml.safe_dump is not available
            with open(temp_config_path, "w") as f:
                json.dump(quick_config, f, indent=2)

        try:
            print(f"🚀 Quick training: {args.dataset} + {args.model}")
            train_from_config(temp_config_path)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            sys.exit(1)
        finally:
            Path(temp_config_path).unlink(missing_ok=True)


def handle_optimize(args):
    """Handle optimize command for hyperparameter optimization."""
    from .train import optimize_from_config

    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return

    try:
        print(f"🎯 Starting hyperparameter optimization with: {args.config}")
        print(f"🔄 Running {args.trials} trials...")

        # Override experiment name if provided
        if args.experiment_name:
            print(f"📝 Using experiment name: {args.experiment_name}")

        optimize_from_config(
            config_path=args.config,
            n_trials=args.trials,
            experiment_name=args.experiment_name,
        )

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def handle_config(args):
    """Handle config command."""
    if not args.config_action:
        print("❌ Config action required. Use --help for options.")
        return

    from .train import create_default_config

    if args.config_action == "create":
        print(f"📝 Creating default configuration: {args.output}")
        create_default_config(args.output)

    elif args.config_action == "surveys":
        print("🌌 Available survey configurations:")
        try:
            loader = ConfigLoader()
            surveys = loader.list_available_surveys()

            if not surveys:
                print("   No survey configurations found in configs/surveys/")
            else:
                for survey in surveys:
                    print(f"   • {survey}")

            print("\n💡 Use 'astro-lab config show <survey>' to see details")
        except Exception as e:
            print(f"❌ Error listing surveys: {e}")
            traceback.print_exc()

    elif args.config_action == "show":
        print(f"📊 Survey configuration: {args.survey}")
        try:
            loader = ConfigLoader()
            survey_config = loader.get_survey_config(args.survey)

            print(f"   Survey: {survey_config['name']}")
            print(f"   Description: {survey_config['description']}")
            print(f"   Features: {len(survey_config['features'])}")

            if "graph" in survey_config:
                graph_config = survey_config["graph"]
                print(f"   K-neighbors: {graph_config.get('k_neighbors', 'N/A')}")
                print(
                    f"   Distance metric: {graph_config.get('distance_metric', 'N/A')}"
                )

            if "processing" in survey_config:
                proc_config = survey_config["processing"]
                print(
                    f"   Normalize features: {proc_config.get('normalize_features', 'N/A')}"
                )

            print("\n📋 Available features:")
            for feature in survey_config["features"]:
                print(f"   • {feature}")

        except Exception as e:
            print(f"❌ Error showing survey config: {e}")

    elif args.config_action == "model":
        handle_model_config(args)


def handle_model_config(args):
    """Handle model configuration management."""
    from astro_lab.models.config import list_predefined_configs, get_predefined_config, ModelConfig
    
    if args.action == "list":
        print("🏗️ Available model configurations:")
        configs = list_predefined_configs()
        for config_name in configs:
            print(f"   • {config_name}")
        print("\n💡 Use 'astro-lab config model show <name>' to see details")
        
    elif args.action == "show":
        if not args.name:
            print("❌ Model name required for show action")
            return
            
        try:
            config = get_predefined_config(args.name)
            print(f"🏗️ Model configuration: {args.name}")
            print(f"   Name: {config.name}")
            print(f"   Description: {config.description}")
            print(f"   Version: {config.version}")
            
            print(f"\n📊 Encoder settings:")
            print(f"   • Photometry: {config.encoder.use_photometry}")
            print(f"   • Astrometry: {config.encoder.use_astrometry}")
            print(f"   • Spectroscopy: {config.encoder.use_spectroscopy}")
            
            print(f"\n🕸️ Graph settings:")
            print(f"   • Conv type: {config.graph.conv_type}")
            print(f"   • Hidden dim: {config.graph.hidden_dim}")
            print(f"   • Num layers: {config.graph.num_layers}")
            print(f"   • Dropout: {config.graph.dropout}")
            
            print(f"\n🎯 Output settings:")
            print(f"   • Task: {config.output.task}")
            print(f"   • Output dim: {config.output.output_dim}")
            print(f"   • Pooling: {config.output.pooling}")
            
        except KeyError:
            print(f"❌ Model configuration '{args.name}' not found")
            print("💡 Use 'astro-lab config model list' to see available configs")
            
    elif args.action == "create":
        if not args.name:
            print("❌ Model name required for create action")
            return
            
        try:
            config = get_predefined_config(args.name)
            output_file = args.output or f"{args.name}_config.yaml"
            
            # Convert to dict and save
            config_dict = config.dict()
            with open(output_file, "w") as f:
                try:
                    yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
                except (AttributeError, ImportError):
                    # Fallback to json if yaml is not available
                    json.dump(config_dict, f, indent=2)
                
            print(f"✅ Model configuration saved to: {output_file}")
            print(f"💡 You can now use this config in your training:")
            print(f"   astro-lab train --config {output_file}")
            
        except KeyError:
            print(f"❌ Model configuration '{args.name}' not found")
            print("💡 Use 'astro-lab config model list' to see available configs")


if __name__ == "__main__":
    main()

