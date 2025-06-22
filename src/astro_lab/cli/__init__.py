"""
AstroLab CLI - Unified Command Line Interface
============================================

Modern CLI for astronomical data processing and ML training.
Supports multiple survey types with unified preprocessing and training pipelines.
"""

import argparse
import datetime

# Removed memory.py - using simple gc instead
import gc
import json
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import yaml

from astro_lab.data import (
    AstroDataset,
    create_astro_datamodule,
    create_training_splits,
    data_config,
    get_data_statistics,
    load_catalog,
    load_gaia_data,
    load_nsa_data,
    load_sdss_data,
    load_tng50_data,
    save_splits_to_parquet,
)
from astro_lab.data.preprocessing import preprocess_catalog, preprocess_catalog_lazy
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

astro-lab preprocess         Easy preprocessing - process all surveys or specific files
astro-lab download          Download astronomical datasets
astro-lab train             ML-Model Training with Lightning + MLflow
astro-lab optimize          Hyperparameter optimization with Optuna
astro-lab config            Configuration management

📖 Examples:

# Easy: Process all surveys with defaults
astro-lab preprocess

# Process specific surveys
astro-lab preprocess --surveys gaia nsa

# Process specific file with survey config
astro-lab preprocess data/gaia_catalog.parquet --config gaia

# With detailed parameters
astro-lab preprocess --surveys gaia --k-neighbors 8 --max-samples 10000 --splits

# Process catalog (statistics are shown automatically)
astro-lab preprocess data/catalog.parquet

# Process TNG50 simulation
astro-lab preprocess data/snap_099.0.hdf5 --tng50 --particle-types PartType4

# Download Gaia DR3 bright stars
astro-lab download gaia --magnitude-limit 12.0

# Training
astro-lab train --config my_config.yaml
astro-lab train --dataset gaia --model gaia_classifier --epochs 50

💡 Use 'astro-lab <command> --help' for detailed options!
        """,
    )

    parser.add_argument("--version", action="version", version="AstroLab 0.1.0")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 🌟 UNIFIED: Preprocess command (easy to use, detailed when needed)
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Data preprocessing and graph creation (easy to use, detailed when needed)",
    )

    # Optional input file (if not provided, processes all surveys)
    preprocess_parser.add_argument(
        "input",
        nargs="?",
        help="Input file to process (optional - if not provided, processes all surveys)",
    )

    # Survey selection (for batch processing or single file)
    preprocess_parser.add_argument(
        "--surveys",
        nargs="+",
        choices=["gaia", "nsa", "sdss", "linear", "exoplanet", "tng50"],
        help="Process specific surveys (default: all available surveys)",
    )

    # Survey configuration for single files
    preprocess_parser.add_argument(
        "--config",
        "-c",
        help="Survey configuration for single file processing (gaia, sdss, nsa, etc.)",
    )

    # Output options
    preprocess_parser.add_argument(
        "--output", "-o", help="Output directory or file path"
    )
    preprocess_parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for batch processing (default: data/processed)",
    )

    # Processing parameters
    preprocess_parser.add_argument(
        "--k-neighbors",
        "-k",
        type=int,
        default=8,
        help="Number of K-neighbors for graph creation (default: 8)",
    )
    preprocess_parser.add_argument(
        "--distance-threshold",
        "-d",
        type=float,
        default=50.0,
        help="Distance threshold for edges (default: 50.0)",
    )
    preprocess_parser.add_argument(
        "--max-samples", "-n", type=int, help="Maximum number of samples (default: all)"
    )

    # Special modes
    preprocess_parser.add_argument(
        "--splits", action="store_true", help="Create train/val/test splits"
    )

    preprocess_parser.add_argument(
        "--tng50", action="store_true", help="Process as TNG50 simulation data"
    )
    preprocess_parser.add_argument(
        "--particle-types",
        default="PartType4",
        help="Particle types for TNG50 processing (default: PartType4)",
    )

    # Control options
    preprocess_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing data"
    )
    preprocess_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Detailed output"
    )

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

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument("--config", "-c", help="Configuration file path")
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

    # Optimize subcommand
    optimize_parser = subparsers.add_parser(
        "optimize", help="Run hyperparameter optimization"
    )
    optimize_parser.add_argument("config", help="Configuration file path")
    optimize_parser.add_argument(
        "--trials", type=int, default=10, help="Number of optimization trials"
    )
    optimize_parser.add_argument("--experiment-name", help="Override experiment name")

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

    # Model config subcommand
    model_parser = config_subparsers.add_parser(
        "model", help="Model configuration management"
    )
    model_parser.add_argument(
        "action", choices=["list", "show", "create"], help="Model config action"
    )
    model_parser.add_argument("--name", help="Model config name")
    model_parser.add_argument("--output", "-o", help="Output file for create action")

    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate handler
    try:
        if args.command == "preprocess":
            handle_preprocess(args)
        elif args.command == "download":
            handle_download(args)
        elif args.command == "train":
            handle_train(args)
        elif args.command == "optimize":
            handle_optimize(args)
        elif args.command == "config":
            handle_config(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up memory to prevent leaks
        _cleanup_memory()

        # Note: Memory leak warning is expected due to heavy ML/astronomy imports
        # This is a known issue with PyTorch, astropy, and visualization libraries
        # The functionality is not affected.


def _cleanup_memory():
    """Clean up memory to prevent leaks from heavy imports."""
    try:
        # Clean up PyTorch CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Clean up matplotlib if imported
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:
            pass

    except Exception:
        # Memory cleanup shouldn't fail the entire program
        pass


def handle_preprocess(args):
    """Handle unified preprocess command - easy to use, detailed when needed."""
    print("🚀 AstroLab - Data Preprocessing")
    print("=" * 50)

    try:
        import subprocess
        import sys
        from pathlib import Path

        # Determine processing mode

        if args.tng50 and args.input:
            # TNG50 processing mode
            _process_tng50(args)
            return

        if args.input:
            # Single file processing mode
            _process_single_file(args)
            return

        # Batch processing mode (no input file provided)
        _process_all_surveys(args)

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def _show_catalog_stats(input_path: str):
    """Show statistics for a catalog file."""
    print(f"📊 Analyzing: {input_path}")
    try:
        from astro_lab.data import get_data_statistics, load_catalog

        df = load_catalog(input_path)
        stats = get_data_statistics(df)

        print("📈 Dataset Statistics:")
        print(f"  • Rows: {stats['n_rows']:,}")
        print(f"  • Columns: {stats['n_cols']}")
        print(f"  • Memory: {stats.get('memory_mb', 'N/A')} MB")

        if "columns" in stats:
            print("  • Column Types:")
            for col_type, count in stats["columns"].items():
                print(f"    - {col_type}: {count}")

    except Exception as e:
        print(f"❌ Error analyzing catalog: {e}")
        raise


def _process_tng50(args):
    """Process TNG50 simulation data."""
    print(f"🌌 Processing TNG50 simulation: {args.input}")

    try:
        from astro_lab.data.preprocessing import (
            preprocess_catalog,
            preprocess_catalog_lazy,
        )

        output_dir = args.output or "data/processed/tng50/"

        # Use lazy preprocessing for TNG50 data
        lf = preprocess_catalog_lazy(
            input_path=args.input,
            survey_type="tng50",
            max_samples=args.max_samples,
            output_dir=output_dir,
            use_streaming=True,
        )

        # Collect the lazy frame to get the actual DataFrame
        df = lf.collect()

        print(f"✅ TNG50 processing completed: {len(df)} particles processed")
        print(f"📁 Output: {output_dir}")

    except Exception as e:
        print(f"❌ TNG50 processing failed: {e}")
        raise


def _process_single_file(args):
    """Process a single catalog file."""
    print(f"📂 Processing single file: {args.input}")

    try:
        from astro_lab.data import (
            create_training_splits,
            load_catalog,
            save_splits_to_parquet,
        )
        from astro_lab.data.preprocessing import (
            create_graph_from_dataframe,
            preprocess_catalog_lazy,
        )
        from astro_lab.utils.config.loader import ConfigLoader

        # OPTIMIZED: Use astronomy processing context
        with astro_processing_context(f"Processing {Path(args.input).name}") as ctx:
            # Load catalog
            df = load_catalog(args.input)
            print(f"📊 Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

            # Determine survey type
            survey_type = args.config or "generic"

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

            # OPTIMIZED: Use lazy preprocessing for large files
            use_streaming = df.shape[0] > 50_000  # Use streaming for large datasets

            if use_streaming:
                print("🚀 Using streaming mode for large dataset")
                lf_clean = preprocess_catalog_lazy(
                    args.input,
                    survey_type=survey_type,
                    max_samples=args.max_samples,
                    use_streaming=True,
                )
                # Collect only if needed for further processing
                df_clean = lf_clean.collect()
            else:
                # Use lazy preprocessing for smaller files too (still more efficient)
                lf_clean = preprocess_catalog_lazy(
                    args.input,
                    survey_type=survey_type,
                    max_samples=args.max_samples,
                    use_streaming=False,
                )
                df_clean = lf_clean.collect()

            print(f"✅ Processed: {df_clean.shape[0]:,} rows retained")

            # Handle output
            output_path = (
                Path(args.output)
                if args.output
                else Path(f"data/processed/{survey_type}/")
            )
            output_path.mkdir(parents=True, exist_ok=True)

            # Create splits if requested
            if args.splits:
                print("🔄 Creating training splits...")
                train, val, test = create_training_splits(df_clean)

                dataset_name = Path(args.input).stem
                save_splits_to_parquet(train, val, test, output_path, dataset_name)
                print(f"💾 Splits and graphs saved to: {output_path}")
            else:
                # Save processed catalog and create graph
                output_file = output_path / f"{Path(args.input).stem}_processed.parquet"
                df_clean.write_parquet(output_file)
                print(f"💾 Processed catalog saved to: {output_file}")

                # Create graph
                graph_file = (
                    output_path / f"{Path(args.input).stem}_k{args.k_neighbors}.pt"
                )
                graph_data = create_graph_from_dataframe(
                    df=df_clean,
                    survey_type=survey_type,
                    k_neighbors=args.k_neighbors,
                    distance_threshold=args.distance_threshold,
                    output_path=graph_file,
                )

                if graph_data:
                    print(
                        f"📊 Graph created: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges"
                    )

    except Exception as e:
        print(f"❌ Single file processing failed: {e}")
        raise


def _process_all_surveys(args):
    """Process all surveys (batch mode)."""
    print("🌟 Batch processing mode - processing all surveys")

    try:
        import subprocess
        import sys
        from pathlib import Path

        # Available surveys
        all_surveys = ["gaia", "nsa", "sdss", "linear", "exoplanet", "tng50"]

        # Determine surveys to process
        surveys_to_process = args.surveys if args.surveys else all_surveys

        print(f"📊 Processing surveys: {', '.join(surveys_to_process)}")
        print(
            f"🔧 Parameters: k={args.k_neighbors}, max_samples={args.max_samples or 'all'}"
        )
        print(f"📁 Output: {args.output_dir}")
        print()

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each survey
        successful = 0
        for survey in surveys_to_process:
            print(f"🔄 Processing {survey.upper()}...")

            try:
                # Use the preprocessing module directly
                from astro_lab.data.preprocessing import (
                    create_graph_from_dataframe,
                    preprocess_catalog,
                )

                # Find data files for this survey
                survey_data_dir = Path(f"data/raw/{survey}")
                if not survey_data_dir.exists():
                    print(f"⚠️  No data directory found for {survey}: {survey_data_dir}")
                    continue

                data_files = list(survey_data_dir.glob("*.parquet")) + list(
                    survey_data_dir.glob("*.csv")
                )
                if not data_files:
                    print(f"⚠️  No data files found for {survey}")
                    continue

                survey_output_dir = output_dir / survey
                survey_output_dir.mkdir(parents=True, exist_ok=True)

                for data_file in data_files:
                    if args.verbose:
                        print(f"  📄 Processing {data_file.name}")

                    # Preprocess
                    lf = preprocess_catalog_lazy(
                        input_path=str(data_file),
                        survey_type=survey,
                        max_samples=args.max_samples,
                        use_streaming=True,
                    )
                    df = lf.collect()

                    # Save processed data
                    processed_file = (
                        survey_output_dir / f"{data_file.stem}_processed.parquet"
                    )
                    df.write_parquet(processed_file)

                    # Create graph
                    graph_file = (
                        survey_output_dir / f"{data_file.stem}_k{args.k_neighbors}.pt"
                    )
                    graph_data = create_graph_from_dataframe(
                        df=df,
                        survey_type=survey,
                        k_neighbors=args.k_neighbors,
                        distance_threshold=args.distance_threshold,
                        output_path=graph_file,
                    )

                    if args.verbose and graph_data:
                        print(
                            f"    📊 Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges"
                        )

                print(f"✅ {survey.upper()} processed successfully")
                successful += 1

            except Exception as e:
                print(f"❌ Error processing {survey.upper()}: {e}")
                if args.verbose:
                    traceback.print_exc()
                continue

        print()
        print(
            f"🎉 Batch processing completed! ({successful}/{len(surveys_to_process)} surveys successful)"
        )
        print(f"📁 Processed data in: {output_dir}")

        # Show summary
        print("\n📊 Summary:")
        for survey in surveys_to_process:
            survey_dir = output_dir / survey
            if survey_dir.exists():
                pt_files = list(survey_dir.glob("*.pt"))
                parquet_files = list(survey_dir.glob("*_processed.parquet"))
                print(
                    f"   {survey.upper()}: {len(parquet_files)} processed files, {len(pt_files)} graphs"
                )
            else:
                print(f"   {survey.upper()}: No processed data found")

    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        raise


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
        from astro_lab.models.config import (
            EncoderConfig,
            GraphConfig,
            ModelConfig,
            OutputConfig,
        )

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
    from astro_lab.models.config import (
        ModelConfig,
        get_predefined_config,
        list_predefined_configs,
    )

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

            print("\n📊 Encoder settings:")
            print(f"   • Photometry: {config.encoder.use_photometry}")
            print(f"   • Astrometry: {config.encoder.use_astrometry}")
            print(f"   • Spectroscopy: {config.encoder.use_spectroscopy}")

            print("\n🕸️ Graph settings:")
            print(f"   • Conv type: {config.graph.conv_type}")
            print(f"   • Hidden dim: {config.graph.hidden_dim}")
            print(f"   • Num layers: {config.graph.num_layers}")
            print(f"   • Dropout: {config.graph.dropout}")

            print("\n🎯 Output settings:")
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
            print("💡 You can now use this config in your training:")
            print(f"   astro-lab train --config {output_file}")

        except KeyError:
            print(f"❌ Model configuration '{args.name}' not found")
            print("💡 Use 'astro-lab config model list' to see available configs")


if __name__ == "__main__":
    main()
