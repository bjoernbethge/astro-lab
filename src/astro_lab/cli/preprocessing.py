#!/usr/bin/env python3
"""
AstroLab Preprocessing CLI - Clean wrapper around astro_lab.data

Uses existing astro_lab.data functions directly without redundancy.
Automatically creates PyTorch Geometric Graphs (.pt) for all surveys - standard in GNNs!
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from astro_lab.data import (
    create_graph_datasets_from_splits,
    create_graph_from_dataframe,
    create_training_splits,
    detect_survey_type,
    get_data_statistics,
    list_catalogs,
    load_catalog,
    load_splits_from_parquet,
    preprocess_catalog,
    save_splits_to_parquet,
)
from astro_lab.data.core import create_cosmic_web_loader


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("astro_lab_cli")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


# Available surveys with their descriptions
AVAILABLE_SURVEYS = {
    "gaia": {
        "name": "Gaia DR3",
        "description": "Stellar catalog with proper motions and photometry",
        "type": "stellar",
        "default_scales": [1.0, 2.0, 5.0, 10.0],
        "color": "#ffd700",  # Gold
    },
    "sdss": {
        "name": "SDSS DR17",
        "description": "Galaxy photometry and spectroscopy",
        "type": "galaxy",
        "default_scales": [5.0, 10.0, 20.0, 50.0],
        "color": "#4a90e2",  # Blue
    },
    "nsa": {
        "name": "NASA Sloan Atlas",
        "description": "Galaxy catalog with distance measurements",
        "type": "galaxy",
        "default_scales": [5.0, 10.0, 20.0, 50.0],
        "color": "#e24a4a",  # Red
    },
    "linear": {
        "name": "LINEAR",
        "description": "Asteroid light curves and variable stars",
        "type": "solar_system",
        "default_scales": [5.0, 10.0, 20.0, 50.0],
        "color": "#ff8800",  # Orange
    },
    "tng": {
        "name": "TNG50 Simulation",
        "description": "Cosmological simulation particles",
        "type": "simulation",
        "default_scales": [5.0, 10.0, 20.0, 50.0],
        "color": "#00ff00",  # Green
    },
    "exoplanet": {
        "name": "NASA Exoplanet Archive",
        "description": "Confirmed exoplanet systems",
        "type": "planetary",
        "default_scales": [10.0, 25.0, 50.0, 100.0],
        "color": "#ff00ff",  # Magenta
    },
}


def print_stats(
    stats: Dict, verbose: bool = False, logger: Optional[logging.Logger] = None
) -> None:
    """Pretty print data statistics."""
    if logger is None:
        logger = logging.getLogger("astro_lab_cli")

    logger.info("📈 Dataset Statistics:")
    logger.info(f"  • Rows: {stats['n_rows']:,}")
    logger.info(f"  • Columns: {stats['n_columns']}")
    logger.info(f"  • Memory usage: {stats['memory_usage_mb']:.1f} MB")
    logger.info(f"  • Numeric columns: {len(stats['numeric_columns'])}")

    if verbose and len(stats["columns"]) > 0:
        logger.info("📋 Column details:")
        for col in stats["columns"][:20]:  # Show first 20 columns
            dtype = stats["dtypes"][col]
            logger.info(f"  • {col}: {dtype}")

        if len(stats["columns"]) > 20:
            logger.info(f"  ... and {len(stats['columns']) - 20} more columns")

    if stats["missing_data"]:
        logger.info("🚨 Missing data:")
        for col, info in list(stats["missing_data"].items())[:10]:  # Show first 10
            logger.info(
                f"  • {col}: {info['null_count']:,} ({info['null_percentage']:.1f}%)"
            )

        if len(stats["missing_data"]) > 10:
            logger.info(
                f"  ... and {len(stats['missing_data']) - 10} more columns with missing data"
            )


def extract_preprocess_args(args) -> Dict:
    """Extract preprocessing arguments from CLI args."""
    return {
        "clean_null_columns": args.clean_nulls,
        "min_observations": args.min_observations,
        "magnitude_columns": args.magnitude_columns.split(",")
        if args.magnitude_columns
        else None,
        "coordinate_columns": args.coordinate_columns.split(",")
        if args.coordinate_columns
        else None,
    }


def extract_split_args(args) -> Dict:
    """Extract split arguments from CLI args."""
    return {
        "test_size": args.test_size,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "shuffle": args.shuffle,
    }


def process_catalog_command(args):
    """Process a catalog file - clean wrapper with automatic graph creation."""
    logger = setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"📂 Loading catalog: {input_path}")

    # Load using data module
    try:
        df = load_catalog(input_path)
        logger.info(f"📊 Original data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"❌ Error loading catalog: {e}")
        sys.exit(1)

    # Show original statistics if requested
    if args.stats:
        logger.info("📈 Original data statistics:")
        stats = get_data_statistics(df)
        print_stats(stats, args.verbose, logger)

    # Preprocess using data module function
    df_clean = preprocess_catalog(df, **extract_preprocess_args(args))
    logger.info(
        f"📈 Processed data: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns"
    )

    # Handle splits or save processed data
    if args.create_splits:
        logger.info("🔄 Creating training splits...")
        train, val, test = create_training_splits(df_clean, **extract_split_args(args))

        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            dataset_name = input_path.stem

            # Force mode: delete existing files
            if args.force:
                for split_name in ["train", "val", "test"]:
                    split_file = output_path / f"{dataset_name}_{split_name}.parquet"
                    if split_file.exists():
                        split_file.unlink()
                        logger.info(f"🗑️  Deleted existing: {split_file.name}")

            save_splits_to_parquet(train, val, test, output_path, dataset_name)
            logger.info(f"✅ Splits saved to: {output_path}")

            # AUTOMATIC Graph creation - Standard in GNNs!
            create_graph_datasets_from_splits(
                train,
                val,
                test,
                output_path,
                dataset_name,
                k_neighbors=getattr(args, "k_neighbors", 8),
                distance_threshold=getattr(args, "distance_threshold", 50.0),
            )
        else:
            logger.info("💡 Use --output to save splits to disk")

    elif args.output:
        # Save processed catalog
        output_path = Path(args.output)
        if output_path.is_dir():
            output_file = output_path / f"{input_path.stem}_processed.parquet"
        else:
            output_file = output_path

        if args.force and output_file.exists():
            output_file.unlink()
            logger.info(f"🗑️  Deleted existing: {output_file.name}")

        df_clean.write_parquet(output_file)
        logger.info(f"💾 Processed catalog saved to: {output_file}")

        # AUTOMATIC single graph creation - Standard in GNNs!
        survey_type = detect_survey_type(output_file.stem, df_clean)
        create_graph_from_dataframe(
            df_clean,
            survey_type,
            k_neighbors=getattr(args, "k_neighbors", 8),
            distance_threshold=getattr(args, "distance_threshold", 50.0),
            output_path=output_file.parent,
        )

    else:
        # Auto-save to survey-specific processed directory
        from astro_lab.data.config import data_config

        # Detect survey type from filename or data
        survey_type = detect_survey_type(input_path.stem, df_clean)

        # Create survey-specific processed directory
        data_config.ensure_survey_directories(survey_type)
        processed_dir = data_config.get_survey_processed_dir(survey_type)

        # Save processed catalog
        output_file = processed_dir / f"{input_path.stem}_processed.parquet"

        if args.force and output_file.exists():
            output_file.unlink()
            logger.info(f"🗑️  Deleted existing: {output_file.name}")

        df_clean.write_parquet(output_file)
        logger.info(f"💾 Processed catalog saved to: {output_file}")

        # AUTOMATIC single graph creation - Standard in GNNs!
        create_graph_from_dataframe(
            df_clean,
            survey_type,
            k_neighbors=getattr(args, "k_neighbors", 8),
            distance_threshold=getattr(args, "distance_threshold", 50.0),
            output_path=processed_dir,
        )

    logger.info(f"✅ Processing complete: {df_clean.shape[0]:,} rows retained")


def stats_command(args):
    """Show statistics for a catalog - direct wrapper."""
    logger = setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"📂 Analyzing: {input_path}")
    try:
        df = load_catalog(input_path)
        stats = get_data_statistics(df)
        print_stats(stats, args.verbose, logger)
    except Exception as e:
        logger.error(f"❌ Error analyzing catalog: {e}")
        sys.exit(1)


def load_splits_command(args):
    """Load and display splits - direct wrapper."""
    logger = setup_logging(args.verbose)

    base_path = Path(args.path)
    if not base_path.exists():
        logger.error(f"❌ Path not found: {base_path}")
        sys.exit(1)

    try:
        train, val, test = load_splits_from_parquet(base_path, args.dataset)
        logger.info(f"📊 Loaded splits from: {base_path}")
        logger.info(f"  • Train: {len(train):,} rows")
        logger.info(f"  • Val: {len(val):,} rows")
        logger.info(f"  • Test: {len(test):,} rows")
    except Exception as e:
        logger.error(f"❌ Error loading splits: {e}")
        sys.exit(1)


def list_catalogs_command(args):
    """List available catalogs - direct wrapper."""
    logger = setup_logging(args.verbose)

    try:
        catalogs = list_catalogs()
        logger.info("📚 Available catalogs:")
        for catalog in catalogs:
            logger.info(f"  • {catalog}")
    except Exception as e:
        logger.error(f"❌ Error listing catalogs: {e}")
        sys.exit(1)


def cosmic_web_command(args):
    """Perform cosmic web analysis using integrated Data module functions."""
    logger = setup_logging(args.verbose)

    logger.info(f"🌌 Cosmic Web Analysis for survey: {args.survey}")

    try:
        # Use integrated cosmic web analysis
        cosmic_web_results = create_cosmic_web_loader(
            survey=args.survey,
            max_samples=args.max_samples,
            scales_mpc=args.scales,
        )

        logger.info(
            f"✅ Analysis complete: {cosmic_web_results['n_objects']:,} objects processed"
        )

        # Save results if output specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save coordinates
            coords_tensor = torch.tensor(cosmic_web_results["coordinates"])
            torch.save(coords_tensor, output_dir / f"{args.survey}_coords_3d_mpc.pt")

            # Save summary
            with open(output_dir / f"{args.survey}_cosmic_web_summary.txt", "w") as f:
                f.write(
                    f"{args.survey.upper()} Cosmic Web Analysis (Integrated Data Module)\n"
                )
                f.write("=" * 50 + "\n\n")
                f.write(f"Survey: {cosmic_web_results['survey_name']}\n")
                f.write(f"Total objects: {cosmic_web_results['n_objects']:,}\n")
                f.write(
                    f"Total volume: {cosmic_web_results['total_volume']:.0f} Mpc³\n"
                )
                f.write(
                    f"Global density: {cosmic_web_results['global_density']:.2e} obj/Mpc³\n\n"
                )

                f.write("Multi-scale clustering results:\n")
                for scale, result in cosmic_web_results["results_by_scale"].items():
                    f.write(f"  {scale} Mpc:\n")
                    f.write(f"    Groups: {result['n_clusters']}\n")
                    f.write(f"    Grouped: {result['grouped_fraction'] * 100:.1f}%\n")
                    f.write(f"    Time: {result['time_s']:.1f}s\n")
                    f.write(
                        f"    Local density: {result['mean_local_density']:.2e} ± {result['density_variation']:.2e} obj/pc³\n"
                    )
                    f.write(
                        f"    Density stats: min={result['local_density_stats']['min']:.2e}, "
                    )
                    f.write(f"median={result['local_density_stats']['median']:.2e}, ")
                    f.write(f"max={result['local_density_stats']['max']:.2e}\n\n")

            logger.info(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Error in cosmic web analysis: {e}")
        sys.exit(1)


def process_all_surveys_command(args):
    """Process all available surveys with cosmic web analysis."""
    logger = setup_logging(args.verbose)

    logger.info("🌌 Processing All Available Surveys")
    logger.info("=" * 60)

    all_results = {}
    total_start_time = time.time()

    for survey in AVAILABLE_SURVEYS.keys():
        try:
            logger.info(
                f"📊 Processing {AVAILABLE_SURVEYS[survey]['name']} ({survey})..."
            )

            results = create_cosmic_web_loader(
                survey=survey,
                max_samples=args.max_samples,
                scales_mpc=args.scales,
            )
            all_results[survey] = results
            logger.info(f"✅ {results['n_objects']:,} objects processed")

        except Exception as e:
            logger.error(f"❌ Failed to process {survey}: {e}")
            continue

    total_time = time.time() - total_start_time

    # Print summary
    logger.info("📊 ALL SURVEYS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"⏱️ Total processing time: {total_time:.1f}s")
    logger.info(
        f"✅ Successfully processed: {len(all_results)}/{len(AVAILABLE_SURVEYS)} surveys"
    )

    if all_results:
        total_objects = sum(r["n_objects"] for r in all_results.values())
        total_volume = sum(r["total_volume"] for r in all_results.values())

        logger.info(f"📊 Total objects: {total_objects:,}")
        logger.info(f"📏 Total volume: {total_volume:.0f} Mpc³")

        logger.info("🌟 Survey breakdown:")
        for survey, results in all_results.items():
            survey_info = AVAILABLE_SURVEYS[survey]
            percentage = results["n_objects"] / total_objects * 100
            logger.info(
                f"  {survey_info['name']:20s}: {results['n_objects']:8,} objects ({percentage:5.1f}%)"
            )

    # Save results if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for survey, results in all_results.items():
            # Save coordinates
            coords_tensor = torch.tensor(results["coordinates"])
            torch.save(coords_tensor, output_dir / f"{survey}_coords_3d_mpc.pt")

            # Save summary
            survey_info = AVAILABLE_SURVEYS[survey]
            summary_file = output_dir / f"{survey}_cosmic_web_summary.txt"
            with open(summary_file, "w") as f:
                f.write(f"{survey_info['name']} Cosmic Web Analysis\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Survey: {survey}\n")
                f.write(f"Type: {survey_info['type']}\n")
                f.write(f"Total objects: {results['n_objects']:,}\n")
                f.write(f"Total volume: {results['total_volume']:.0f} Mpc³\n")
                f.write(f"Global density: {results['global_density']:.2e} obj/Mpc³\n\n")

                f.write("Multi-scale clustering results:\n")
                for scale, result in results["results_by_scale"].items():
                    f.write(f"  {scale} Mpc:\n")
                    f.write(f"    Groups: {result['n_clusters']}\n")
                    f.write(f"    Grouped: {result['grouped_fraction'] * 100:.1f}%\n")
                    f.write(f"    Time: {result['time_s']:.1f}s\n")

        logger.info(f"💾 All results saved to: {output_dir}")


def list_surveys_command(args):
    """List available surveys and their descriptions."""
    logger = setup_logging(args.verbose)

    logger.info("🌌 Available Surveys:")
    logger.info("=" * 60)
    for survey, info in AVAILABLE_SURVEYS.items():
        logger.info(f"  {survey:12s}: {info['name']}")
        logger.info(f"              {info['description']}")
        logger.info(
            f"              Type: {info['type']}, Default scales: {info['default_scales']} Mpc"
        )
        logger.info("")


def show_functions_command():
    """Show available functions - direct wrapper."""
    logger = setup_logging()

    logger.info("🔧 Available Data Module Functions:")
    logger.info("  • load_catalog() - Load any catalog file")
    logger.info("  • preprocess_catalog() - Clean and prepare data")
    logger.info("  • create_training_splits() - Split data for training")
    logger.info("  • create_graph_from_dataframe() - Create PyTorch Geometric graphs")
    logger.info("  • create_cosmic_web_loader() - Perform cosmic web analysis")
    logger.info("  • get_data_statistics() - Get detailed statistics")
    logger.info("  • list_catalogs() - List available catalogs")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AstroLab Data Preprocessing CLI - Clean wrapper around astro_lab.data"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process catalog command
    process_parser = subparsers.add_parser("process", help="Process a catalog file")
    process_parser.add_argument("input", help="Input catalog file")
    process_parser.add_argument("--output", "-o", help="Output file or directory")
    process_parser.add_argument("--stats", action="store_true", help="Show statistics")
    process_parser.add_argument(
        "--create-splits", action="store_true", help="Create train/val/test splits"
    )
    process_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )
    process_parser.add_argument(
        "--clean-nulls", action="store_true", help="Remove null columns"
    )
    process_parser.add_argument(
        "--min-observations", type=int, default=100, help="Minimum observations"
    )
    process_parser.add_argument(
        "--magnitude-columns", help="Comma-separated magnitude columns"
    )
    process_parser.add_argument(
        "--coordinate-columns", help="Comma-separated coordinate columns"
    )
    process_parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split size"
    )
    process_parser.add_argument(
        "--val-size", type=float, default=0.1, help="Validation split size"
    )
    process_parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed"
    )
    process_parser.add_argument("--shuffle", action="store_true", help="Shuffle data")
    process_parser.add_argument(
        "--k-neighbors", type=int, default=8, help="Number of neighbors for graph"
    )
    process_parser.add_argument(
        "--distance-threshold",
        type=float,
        default=50.0,
        help="Distance threshold for graph",
    )
    process_parser.set_defaults(func=process_catalog_command)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show catalog statistics")
    stats_parser.add_argument("input", help="Input catalog file")
    stats_parser.set_defaults(func=stats_command)

    # Load splits command
    splits_parser = subparsers.add_parser("splits", help="Load and display splits")
    splits_parser.add_argument("path", help="Path to splits directory")
    splits_parser.add_argument("dataset", help="Dataset name")
    splits_parser.set_defaults(func=load_splits_command)

    # List catalogs command
    list_parser = subparsers.add_parser("list", help="List available catalogs")
    list_parser.set_defaults(func=list_catalogs_command)

    # Cosmic web command
    cosmic_parser = subparsers.add_parser(
        "cosmic-web", help="Perform cosmic web analysis"
    )
    cosmic_parser.add_argument(
        "survey", help="Survey name (gaia, sdss, nsa, linear, tng, exoplanet)"
    )
    cosmic_parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples"
    )
    cosmic_parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0, 50.0],
        help="Analysis scales in Mpc",
    )
    cosmic_parser.add_argument("--output", "-o", help="Output directory for results")
    cosmic_parser.set_defaults(func=cosmic_web_command)

    # Process all surveys command
    all_parser = subparsers.add_parser(
        "all-surveys", help="Process all available surveys"
    )
    all_parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples per survey"
    )
    all_parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0, 50.0],
        help="Analysis scales in Mpc",
    )
    all_parser.add_argument("--output", "-o", help="Output directory for results")
    all_parser.set_defaults(func=process_all_surveys_command)

    # List surveys command
    surveys_parser = subparsers.add_parser("surveys", help="List available surveys")
    surveys_parser.set_defaults(func=list_surveys_command)

    # Show functions command
    func_parser = subparsers.add_parser("functions", help="Show available functions")
    func_parser.set_defaults(func=show_functions_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
