"""
AstroLab CLI Module
==================

Moderne Command-line Interface für AstroLab mit integrierter Config-Verwaltung:
- Download: Download astronomischer Datensätze
- Train: ML-Model Training mit Lightning + MLflow + Config-Integration
- Config: Konfigurationsverwaltung
"""

# Suppress NumPy warnings before any other imports
import os
import warnings

warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*numpy.core.multiarray.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:numpy"

import argparse
import sys
from pathlib import Path

from astro_lab.utils.config_loader import ConfigLoader

__all__ = [
    "main",
]


def main():
    """Main CLI entry point with streamlined interface."""
    parser = argparse.ArgumentParser(
        description="AstroLab - Astronomical Machine Learning Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 Verfügbare Commands:

astro-lab download       Download astronomischer Datensätze
astro-lab preprocess     Datenvorverarbeitung und Graph-Erstellung
astro-lab train          ML-Model Training mit Lightning + MLflow
astro-lab config         Konfigurationsverwaltung

📖 Beispiele:

# Download Gaia DR3 bright stars
astro-lab download gaia --magnitude-limit 12.0

# Catalog mit Survey-Config vorverarbeiten
astro-lab preprocess catalog data/gaia_catalog.parquet --config gaia --splits --output data/processed/

# Catalog-Statistiken anzeigen
astro-lab preprocess stats data/gaia_catalog.parquet

# TNG50 Simulation verarbeiten (verwendet vollständiges CLI)
astro-lab preprocess tng50 data/snap_099.0.hdf5 --particle-types PartType4 --output data/tng50/

# Default-Config erstellen und bearbeiten
astro-lab config create --output my_config.yaml
# Dann bearbeiten und trainieren:
astro-lab train --config my_config.yaml

# Quick Training ohne Config
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
    train_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization instead of training",
    )
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

    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return

    # Route to appropriate command handler
    if args.command == "download":
        handle_download(args)
    elif args.command == "preprocess":
        handle_preprocess(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "config":
        handle_config(args)


def handle_download(args):
    """Handle download command."""
    if not args.download_action:
        print("❌ Download action required. Use --help for options.")
        return

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


def handle_preprocess(args):
    """Handle preprocess command with config-based approach."""
    if not args.preprocess_action:
        print("❌ Preprocessing action required. Use --help for options.")
        return

    from pathlib import Path

    from astro_lab.data import (
        create_graph_datasets_from_splits,
        create_training_splits,
        get_data_statistics,
        load_catalog,
        preprocess_catalog,
        save_splits_to_parquet,
    )
    from astro_lab.utils.config_loader import ConfigLoader

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
        import datetime
        import os

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


def handle_train(args):
    """Handle train command."""
    import yaml

    from .train import create_default_config, optimize_from_config, train_from_config

    if args.config:
        # Config-based training (preferred method)
        if not Path(args.config).exists():
            print(f"❌ Configuration file not found: {args.config}")
            return

        try:
            if args.optimize:
                print(f"🎯 Starting hyperparameter optimization with: {args.config}")
                optimize_from_config(args.config)
            else:
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

        # Create temporary config for quick training
        quick_config = {
            "model": {
                "type": args.model,
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.1,
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
        with open(temp_config_path, "w") as f:
            yaml.dump(quick_config, f, default_flow_style=False, indent=2)

        try:
            print(f"🚀 Quick training: {args.dataset} + {args.model}")
            train_from_config(temp_config_path)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            sys.exit(1)
        finally:
            Path(temp_config_path).unlink(missing_ok=True)


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
            import traceback

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


if __name__ == "__main__":
    main()
