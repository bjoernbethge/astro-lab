"""
AstroLab CLI - Command Line Interface
====================================

Unified command-line interface for AstroLab astronomical machine learning.
Supports data preprocessing, model training, and hyperparameter optimization.
"""

import argparse
import datetime
import json
import logging
import platform
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import yaml

# Configure clean logging ONLY ONCE at module level
# This prevents duplicate messages
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only configure if not already configured
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger


# Windows emoji handling 🖥️
def safe_print(message: str) -> str:
    """Remove emojis on Windows to avoid encoding issues."""
    if platform.system() == "Windows":
        # Replace common emojis with ASCII equivalents
        replacements = {
            "⭐": "*",
            "🚀": "[>]",
            "📖": "[Docs]",
            "💡": "[Tip]",
            "🔧": "[Setup]",
            "📋": "[Info]",
            "⚡": "[Fast]",
            "🌟": "**",
            "📊": "[Data]",
            "📈": "[Stats]",
            "✅": "[OK]",
            "❌": "[ERROR]",
            "⚠️": "[WARN]",
            "🔄": "[...]",
            "💾": "[Save]",
            "📁": "[Dir]",
            "📄": "[File]",
            "🎉": "[Done!]",
            "🎯": "[>]",
            "📝": "[Config]",
            "🏗️": "[Build]",
            "🕸️": "[Graph]",
            "🧠": "[AI]",
            "⏹️": "[Stop]",
            "🌌": "[Space]",
            "🔬": "[Science]",
            "🌠": "[*]",
            "💫": "[~]",
            "✨": "[*]",
            "🪐": "[o]",
            "🛸": "[UFO]",
            "🌍": "[Earth]",
            "🎓": "[Learn]",
            "🔍": "[Search]",
            "📐": "[Math]",
            "🧪": "[Test]",
            "⏱️": "[Time]",
            "🔴": "[o]",
            "🟢": "[o]",
            "🔵": "[o]",
            "🟡": "[o]",
            "🟣": "[o]",
        }
        for emoji, replacement in replacements.items():
            message = message.replace(emoji, replacement)
    return message


# Wrap logger methods for emoji safety
def create_safe_logger(base_logger):
    """Create a logger with safe emoji handling."""

    class SafeLogger:
        def __init__(self, logger):
            self._logger = logger

        def info(self, msg, *args, **kwargs):
            self._logger.info(safe_print(str(msg)), *args, **kwargs)

        def error(self, msg, *args, **kwargs):
            self._logger.error(safe_print(str(msg)), *args, **kwargs)

        def warning(self, msg, *args, **kwargs):
            self._logger.warning(safe_print(str(msg)), *args, **kwargs)

        def debug(self, msg, *args, **kwargs):
            self._logger.debug(safe_print(str(msg)), *args, **kwargs)

    return SafeLogger(base_logger)


# Use the safe logger
logger = create_safe_logger(logger)

# Removed memory.py - using simple gc instead
import gc
import os

from astro_lab.data import (
    AstroDataManager,
    AstroDataset,
    create_astro_datamodule,
    create_training_splits,
    data_config,
    data_manager,
    get_data_statistics,
    load_catalog,
    save_splits_to_parquet,
)
from astro_lab.data.preprocessing import (
    create_graph_from_dataframe,
    find_or_create_catalog_file,
    get_survey_input_file,
    preprocess_catalog,
    preprocess_catalog_lazy,
)
from astro_lab.models.factories import (
    create_asteroid_period_detector,
    create_gaia_classifier,
    create_galaxy_modeler,
    create_lightcurve_classifier,
    create_lsst_transient_detector,
    create_sdss_galaxy_model,
    create_temporal_graph_model,
)
from astro_lab.training.trainer import AstroTrainer
from astro_lab.utils.config.loader import ConfigLoader

__all__ = [
    "main",
]


def main():
    """Main CLI entry point with streamlined interface."""
    # Check if this is a train command to skip welcome message
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Skip welcome for train command 🚂
        pass
    else:
        # Welcome message for other commands 🌟
        logger.info(
            "⭐ Welcome to AstroLab - Astronomical Machine Learning Laboratory! 🔬"
        )
        logger.info("")

    parser = argparse.ArgumentParser(
        description="AstroLab - Astronomical Machine Learning Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=safe_print("""
Available Commands:

astro-lab preprocess     Easy preprocessing - process all surveys or specific files
astro-lab download      Download astronomical datasets  
astro-lab train         ML-Model Training with Lightning + MLflow
astro-lab optimize      Hyperparameter optimization with Optuna
astro-lab config        Configuration management

Examples:

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

Use 'astro-lab <command> --help' for detailed options!
        """),
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

    # Large Gaia processing
    preprocess_parser.add_argument(
        "--gaia-large",
        action="store_true",
        help="Process all 3M Gaia stars with GPU acceleration",
    )

    # SurveyTensor processing
    preprocess_parser.add_argument(
        "--gaia-tensor",
        action="store_true",
        help="Create complete SurveyTensor system from all 3M Gaia stars",
    )

    # Control options
    preprocess_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing data"
    )
    preprocess_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Detailed output"
    )

    # New option
    preprocess_parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Do not generate graph/PT file during preprocessing",
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
        "--dataset",
        choices=["gaia", "gaia_large", "sdss", "nsa"],
        help="Dataset to use",
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
        "--max-samples", type=int, default=1000, help="Maximum number of samples"
    )
    train_parser.add_argument(
        "--learning-rate", "--lr", type=float, default=1e-3, help="Learning rate"
    )
    train_parser.add_argument(
        "--devices", type=int, default=1, help="Number of GPUs to use"
    )
    train_parser.add_argument(
        "--strategy",
        choices=["auto", "ddp", "fsdp"],
        default="auto",
        help="Training strategy",
    )
    train_parser.add_argument(
        "--precision",
        choices=["32", "16-mixed", "bf16-mixed"],
        default="16-mixed",
        help="Training precision",
    )
    train_parser.add_argument(
        "--accumulate", type=int, default=1, help="Gradient accumulation steps"
    )
    train_parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile for optimization"
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

    # New config action
    config_subparsers.add_parser("validate", help="Validate config integration")

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
            logger.error(f"❌ Unknown command: {args.command}")
            parser.print_help()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Operation cancelled by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
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
    logger.info("🚀 AstroLab - Data Preprocessing")
    logger.info("=" * 50)

    try:
        import subprocess
        import sys
        from pathlib import Path

        # Determine processing mode

        if args.gaia_tensor:
            # SurveyTensor processing mode
            _process_gaia_tensor(args)
            return

        if args.gaia_large:
            # Large Gaia processing mode
            _process_gaia_large(args)
            return

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
        logger.error(f"❌ Error during preprocessing: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def _show_catalog_stats(input_path: str):
    """Show statistics for a catalog file."""
    logger.info(f"📊 Analyzing: {input_path}")
    try:
        from astro_lab.data import get_data_statistics, load_catalog

        df = load_catalog(input_path)
        stats = get_data_statistics(df)

        logger.info("📈 Dataset Statistics:")
        logger.info(f"  • Rows: {stats['n_rows']:,}")
        logger.info(f"  • Columns: {stats['n_cols']}")
        logger.info(f"  • Memory: {stats.get('memory_mb', 'N/A')} MB")

        if "columns" in stats:
            logger.info("  • Column Types:")
            for col_type, count in stats["columns"].items():
                logger.info(f"    - {col_type}: {count}")

    except Exception as e:
        logger.error(f"❌ Error analyzing catalog: {e}")
        raise


def _process_gaia_tensor(args):
    """Create complete SurveyTensor system from all 3M Gaia stars."""
    logger.info("🌟 Creating complete Gaia SurveyTensor system (3M stars)")

    try:
        from astro_lab.data.preprocessing import create_gaia_survey_tensor

        # Run the tensor system creation
        result = create_gaia_survey_tensor()

        if result:
            logger.info("✅ Gaia tensor system created successfully!")
            logger.info(f"📊 SurveyTensor: {result['files']['survey']}")
            logger.info(f"🌍 Spatial3DTensor: {result['files']['spatial']}")
            logger.info(f"📸 PhotometricTensor: {result['files']['photometric']}")
            logger.info(f"📋 Metadata: {result['files']['metadata']}")
            logger.info(
                "🎯 Ready for training: uv run astro-lab train --dataset gaia --model gaia_classifier"
            )
        else:
            logger.error("❌ Gaia tensor system creation failed")

    except Exception as e:
        logger.error(f"❌ Gaia tensor system creation failed: {e}")
        raise


def _process_gaia_large(args):
    """Process all 3M Gaia stars with GPU acceleration."""
    logger.info("🚀 Processing large Gaia dataset (3M stars) with GPU acceleration")

    try:
        from astro_lab.data.preprocessing import process_large_gaia_dataset

        # Run the GPU-accelerated processing
        result = process_large_gaia_dataset()

        if result:
            logger.info("✅ Large Gaia processing completed successfully!")
            logger.info(f"📁 Dataset saved to: {result}")
            logger.info(
                "🎯 You can now use: astro-lab train --dataset gaia_large --model gaia_classifier"
            )
        else:
            logger.error("❌ Large Gaia processing failed")

    except Exception as e:
        logger.error(f"❌ Large Gaia processing failed: {e}")
        raise


def _process_tng50(args):
    """Process TNG50 simulation data."""
    logger.info(f"🌌 Processing TNG50 simulation: {args.input}")

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

        logger.info(f"✅ TNG50 processing completed: {len(df)} particles processed")
        logger.info(f"📁 Output: {output_dir}")

    except Exception as e:
        logger.error(f"❌ TNG50 processing failed: {e}")
        raise


def _process_single_file(args):
    """Process a single catalog file."""
    logger.info(f"📂 Processing single file: {args.input}")
    try:
        from astro_lab.data import load_catalog
        from astro_lab.data.preprocessing import preprocess_catalog_lazy
        from astro_lab.utils.config.loader import ConfigLoader

        df = load_catalog(args.input)
        logger.info(f"📊 Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        survey_type = args.config or "generic"
        config = {}
        if args.config:
            try:
                loader = ConfigLoader()
                survey_config = loader.get_survey_config(args.config)
                config = survey_config.get("processing", {})
                logger.info(f"✅ Using {args.config} survey configuration")
            except Exception as e:
                logger.warning(f"⚠️  Could not load survey config: {e}")
        use_streaming = df.shape[0] > 50_000
        if use_streaming:
            logger.info("🚀 Using streaming mode for large dataset")
            lf_clean = preprocess_catalog_lazy(
                args.input,
                survey_type=survey_type,
                max_samples=args.max_samples,
                use_streaming=True,
            )
            df_clean = lf_clean.collect()
        else:
            lf_clean = preprocess_catalog_lazy(
                args.input,
                survey_type=survey_type,
                max_samples=args.max_samples,
                use_streaming=False,
            )
            df_clean = lf_clean.collect()
        # Zeige survey-basierten Output an
        from pathlib import Path

        output_dir = Path(args.output or "data/processed") / survey_type
        parquet_file = output_dir / f"{survey_type}.parquet"
        pt_file = output_dir / f"{survey_type}.pt"
        if parquet_file.exists():
            logger.info(f"✅ Parquet: {parquet_file.relative_to(output_dir.parent)}")
        else:
            logger.warning(
                f"❌ Parquet file missing: {parquet_file.relative_to(output_dir.parent)}"
            )
        if pt_file.exists():
            logger.info(f"✅ Graph:   {pt_file.relative_to(output_dir.parent)}")
        elif not args.no_graph:
            logger.warning(
                f"❌ Graph file missing: {pt_file.relative_to(output_dir.parent)}"
            )
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}")
        import traceback

        traceback.print_exc()


def _process_all_surveys(args):
    """Process all surveys (batch mode)."""
    logger.info("🌟 Batch processing mode - processing all surveys")
    try:
        import sys
        from pathlib import Path

        from astro_lab.data.preprocessing import (
            get_survey_input_file,
            preprocess_catalog_lazy,
        )

        all_surveys = ["gaia", "nsa", "sdss", "linear", "exoplanet", "tng50"]
        surveys_to_process = args.surveys if args.surveys else all_surveys
        max_samples_display = {}
        for survey in surveys_to_process:
            if survey.lower() in ["tng50", "tng50-4"]:
                max_samples_display[survey] = (
                    args.max_samples if args.max_samples is not None else 3_000_000
                )
            else:
                max_samples_display[survey] = args.max_samples or "all"
        logger.info(f"📊 Processing surveys: {', '.join(surveys_to_process)}")
        logger.info(
            "🔧 Parameters: "
            + ", ".join(
                [
                    f"{s}: k={args.k_neighbors}, max_samples={max_samples_display[s]}"
                    for s in surveys_to_process
                ]
            )
        )
        logger.info(f"📁 Output: {args.output_dir}")
        logger.info("")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        successful = 0
        for survey in surveys_to_process:
            logger.info(f"🔄 Processing {survey.upper()}...")
            try:
                data_file = get_survey_input_file(survey, data_manager)
            except FileNotFoundError as e:
                logger.warning(f"⚠️  {e}")
                continue
            survey_output_dir = output_dir / survey
            survey_output_dir.mkdir(parents=True, exist_ok=True)
            if args.verbose:
                logger.info(f"  📄 Processing {data_file.name}")
            ms = max_samples_display[survey]
            if isinstance(ms, str):
                if ms.lower() == "all":
                    ms = None
                else:
                    try:
                        ms = int(ms)
                    except Exception:
                        ms = None
            lf = preprocess_catalog_lazy(
                input_path=str(data_file),
                survey_type=survey,
                max_samples=ms,
                use_streaming=True,
                output_dir=survey_output_dir,
                write_graph=not args.no_graph,
                k_neighbors=args.k_neighbors,
                distance_threshold=args.distance_threshold,
            )
            df = lf.collect()
            parquet_file = survey_output_dir / f"{survey}.parquet"
            pt_file = survey_output_dir / f"{survey}.pt"
            logger.info(f"✅ {survey.upper()} processed successfully")
            if parquet_file.exists():
                logger.info(f"     Parquet: {parquet_file.relative_to(output_dir)}")
            else:
                logger.warning(
                    f"     Parquet file missing: {parquet_file.relative_to(output_dir)}"
                )
            if pt_file.exists():
                logger.info(f"     Graph:   {pt_file.relative_to(output_dir)}")
            elif not args.no_graph:
                logger.warning(
                    f"     Graph file missing: {pt_file.relative_to(output_dir)}"
                )
            successful += 1
        logger.info("")
        logger.info(
            f"🎉 Batch processing completed! ({successful}/{len(surveys_to_process)} surveys successful)"
        )
        logger.info(f"📁 Processed data in: {output_dir}")
    except Exception as e:
        logger.error(f"❌ Error during batch processing: {e}")
        raise


def handle_download(args):
    """Handle download command."""
    if not args.download_action:
        logger.error("❌ Download action required. Use --help for options.")
        return

    try:
        from astro_lab.data import download_bright_all_sky, list_catalogs

        if args.download_action == "gaia":
            logger.info(f"🌟 Downloading Gaia DR3 data (mag < {args.magnitude_limit})")
            try:
                result = download_bright_all_sky(magnitude_limit=args.magnitude_limit)
                logger.info(f"✅ Success: {result}")
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                sys.exit(1)

        elif args.download_action == "list":
            logger.info("📋 Available datasets:")
            try:
                catalogs = list_catalogs()
                logger.info(catalogs)
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    except Exception as e:
        logger.error(f"❌ Error in download command: {e}")
        sys.exit(1)


def handle_train(args):
    """Handle train command by delegating to train module."""
    # Simply pass args to train module
    from .train import train_from_config, train_quick

    if args.config:
        # Config-based training
        if not Path(args.config).exists():
            print(f"Configuration file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        train_from_config(args.config)
    elif args.dataset and args.model:
        # Quick training with all parameters
        train_quick(
            dataset=args.dataset,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=getattr(args, "max_samples", 1000),
            learning_rate=getattr(args, "learning_rate", 0.001),
            devices=getattr(args, "devices", 1),
            strategy=getattr(args, "strategy", "auto"),
            precision=getattr(args, "precision", "16-mixed"),
            accumulate=getattr(args, "accumulate", 1),
        )
    else:
        # Show help if insufficient arguments
        print("Usage:", file=sys.stderr)
        print("  astro-lab train --config <config.yaml>", file=sys.stderr)
        print(
            "  astro-lab train --dataset <dataset> --model <model> [--epochs N] [--batch-size N] [--learning-rate LR] [--devices N] [--strategy STRATEGY] [--precision PRECISION] [--accumulate N]",
            file=sys.stderr,
        )
        sys.exit(1)


def handle_optimize(args):
    """Handle optimize command for hyperparameter optimization."""
    from .optimize import optimize_from_config

    if not Path(args.config).exists():
        logger.error(f"❌ Configuration file not found: {args.config}")
        return

    try:
        logger.info(f"🎯 Starting hyperparameter optimization with: {args.config}")
        logger.info(f"🔄 Running {args.trials} trials...")

        # Override experiment name if provided
        if args.experiment_name:
            logger.info(f"📝 Using experiment name: {args.experiment_name}")

        optimize_from_config(
            config_path=args.config,
            n_trials=args.trials,
            experiment_name=args.experiment_name,
        )

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def handle_config(args):
    """Handle config command."""
    if not args.config_action:
        logger.error("❌ Config action required. Use --help for options.")
        return

    if args.config_action == "create":
        logger.info(f"📝 Creating configuration: {args.output}")
        try:
            # Use ConfigLoader for robust config creation
            from astro_lab.utils.config.loader import ConfigLoader

            # Create a temporary loader to get default config
            loader = ConfigLoader("configs/default.yaml")
            config = loader.load_config()

            # Save the processed config
            loader.save_config(args.output)

            logger.info(f"✅ Configuration created successfully: {args.output}")
            logger.info(
                "💡 The config has been processed with correct paths and MLflow settings"
            )

        except Exception as e:
            logger.error(f"❌ Failed to create config: {e}")
            # Fallback to simple config creation
            try:
                from .optimize import create_default_config

                create_default_config(args.output)
                logger.info(f"✅ Fallback config created: {args.output}")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                return

    elif args.config_action == "surveys":
        logger.info("🌌 Available survey configurations:")
        try:
            loader = ConfigLoader()
            surveys = loader.list_available_surveys()

            if not surveys:
                logger.info("   No survey configurations found in configs/surveys/")
            else:
                for survey in surveys:
                    logger.info(f"   • {survey}")

            logger.info("\n💡 Use 'astro-lab config show <survey>' to see details")
        except Exception as e:
            logger.error(f"❌ Error listing surveys: {e}")
            traceback.print_exc()

    elif args.config_action == "show":
        logger.info(f"📊 Survey configuration: {args.survey}")
        try:
            loader = ConfigLoader()
            survey_config = loader.get_survey_config(args.survey)

            logger.info(f"   Survey: {survey_config['name']}")
            logger.info(f"   Description: {survey_config['description']}")
            logger.info(f"   Features: {len(survey_config['features'])}")
            if "columns" in survey_config:
                logger.info("   • Column Types:")
                for col_type, count in survey_config["columns"].items():
                    logger.info(f"     - {col_type}: {count}")

        except Exception as e:
            logger.error(f"❌ Error showing survey config: {e}")

    elif args.config_action == "model":
        handle_model_config(args)

    elif args.config_action == "validate":
        logger.info("🔍 Validating config integration...")
        try:
            from astro_lab.utils.config.loader import validate_config_integration

            success = validate_config_integration(
                args.config if hasattr(args, "config") else "configs/default.yaml"
            )
            if success:
                logger.info("✅ Config integration is working correctly!")
            else:
                logger.error("❌ Config integration validation failed!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            sys.exit(1)


def handle_model_config(args):
    """Handle model configuration management."""
    from astro_lab.models.config import (
        ModelConfig,
        get_predefined_config,
        list_predefined_configs,
    )

    if args.action == "list":
        logger.info("🏗️ Available model configurations:")
        configs = list_predefined_configs()
        for config_name in configs:
            logger.info(f"   • {config_name}")
        logger.info("\n💡 Use 'astro-lab config model show <name>' to see details")

    elif args.action == "show":
        if not args.name:
            logger.error("❌ Model name required for show action")
            return

        try:
            config = get_predefined_config(args.name)
            logger.info(f"🏗️ Model configuration: {args.name}")
            logger.info(f"   Name: {config.name}")
            logger.info(f"   Description: {config.description}")
            logger.info(f"   Version: {config.version}")

            logger.info("\n📊 Encoder settings:")
            logger.info(f"   • Photometry: {config.encoder.use_photometry}")
            logger.info(f"   • Astrometry: {config.encoder.use_astrometry}")
            logger.info(f"   • Spectroscopy: {config.encoder.use_spectroscopy}")

            logger.info("\n🕸️ Graph settings:")
            logger.info(f"   • Conv type: {config.graph.conv_type}")
            logger.info(f"   • Hidden dim: {config.graph.hidden_dim}")
            logger.info(f"   • Num layers: {config.graph.num_layers}")
            logger.info(f"   • Dropout: {config.graph.dropout}")

            logger.info("\n🎯 Output settings:")
            logger.info(f"   • Task: {config.output.task}")
            logger.info(f"   • Output dim: {config.output.output_dim}")
            logger.info(f"   • Pooling: {config.output.pooling}")

        except KeyError:
            logger.error(f"❌ Model configuration '{args.name}' not found")
            logger.info("💡 Use 'astro-lab config model list' to see available configs")

    elif args.action == "create":
        if not args.name:
            logger.error("❌ Model name required for create action")
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

            logger.info(f"✅ Model configuration saved to: {output_file}")
            logger.info("💡 You can now use this config in your training:")
            logger.info(f"   astro-lab train --config {output_file}")

        except KeyError:
            logger.error(f"❌ Model configuration '{args.name}' not found")
            logger.info("💡 Use 'astro-lab config model list' to see available configs")


if __name__ == "__main__":
    main()
