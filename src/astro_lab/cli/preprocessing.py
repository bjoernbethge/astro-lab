#!/usr/bin/env python3
"""
AstroLab Preprocessing CLI - Clean wrapper around astro_lab.data

Uses existing astro_lab.data functions directly without redundancy.
Automatically creates PyTorch Geometric Graphs (.pt) for all surveys - standard in GNNs!
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

from astro_lab.data.core import (
    create_cosmic_web_loader,
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


def print_stats(stats: Dict, verbose: bool = False) -> None:
    """Pretty print data statistics."""
    print("\n📈 Dataset Statistics:")
    print(f"  • Rows: {stats['n_rows']:,}")
    print(f"  • Columns: {stats['n_columns']}")
    print(f"  • Memory usage: {stats['memory_usage_mb']:.1f} MB")
    print(f"  • Numeric columns: {len(stats['numeric_columns'])}")

    if verbose and len(stats["columns"]) > 0:
        print("\n📋 Column details:")
        for col in stats["columns"][:20]:  # Show first 20 columns
            dtype = stats["dtypes"][col]
            print(f"  • {col}: {dtype}")

        if len(stats["columns"]) > 20:
            print(f"  ... and {len(stats['columns']) - 20} more columns")

    if stats["missing_data"]:
        print("\n🚨 Missing data:")
        for col, info in list(stats["missing_data"].items())[:10]:  # Show first 10
            print(f"  • {col}: {info['null_count']:,} ({info['null_percentage']:.1f}%)")

        if len(stats["missing_data"]) > 10:
            print(
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
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print(f"📂 Loading catalog: {input_path}")

    # Load using data module
    try:
        df = load_catalog(input_path)
        print(f"📊 Original data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        sys.exit(1)

    # Show original statistics if requested
    if args.stats:
        print("\n📈 Original data statistics:")
        stats = get_data_statistics(df)
        print_stats(stats)

    # Preprocess using data module function
    df_clean = preprocess_catalog(df, **extract_preprocess_args(args))
    print(f"📈 Processed data: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")

    # Handle splits or save processed data
    if args.create_splits:
        print("\n🔄 Creating training splits...")
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
                        print(f"🗑️  Deleted existing: {split_file.name}")

            save_splits_to_parquet(train, val, test, output_path, dataset_name)
            print(f"✅ Splits saved to: {output_path}")

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
            print("💡 Use --output to save splits to disk")

    elif args.output:
        # Save processed catalog
        output_path = Path(args.output)
        if output_path.is_dir():
            output_file = output_path / f"{input_path.stem}_processed.parquet"
        else:
            output_file = output_path

        if args.force and output_file.exists():
            output_file.unlink()
            print(f"🗑️  Deleted existing: {output_file.name}")

        df_clean.write_parquet(output_file)
        print(f"💾 Processed catalog saved to: {output_file}")

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
            print(f"🗑️  Deleted existing: {output_file.name}")

        df_clean.write_parquet(output_file)
        print(f"💾 Processed catalog saved to: {output_file}")

        # AUTOMATIC single graph creation - Standard in GNNs!
        create_graph_from_dataframe(
            df_clean,
            survey_type,
            k_neighbors=getattr(args, "k_neighbors", 8),
            distance_threshold=getattr(args, "distance_threshold", 50.0),
            output_path=processed_dir,
        )

    print(f"\n✅ Processing complete: {df_clean.shape[0]:,} rows retained")


def stats_command(args):
    """Show statistics for a catalog - direct wrapper."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print(f"📂 Analyzing: {input_path}")
    try:
        df = load_catalog(input_path)
        stats = get_data_statistics(df)
        print_stats(stats, verbose=args.verbose)
    except Exception as e:
        print(f"❌ Error analyzing catalog: {e}")
        sys.exit(1)


def load_splits_command(args):
    """Load and display splits - direct wrapper."""
    base_path = Path(args.path)
    if not base_path.exists():
        print(f"❌ Path not found: {base_path}")
        sys.exit(1)

    try:
        train, val, test = load_splits_from_parquet(base_path, args.dataset)
        print(f"📊 Loaded splits from: {base_path}")
        print(f"  • Train: {len(train):,} rows")
        print(f"  • Val: {len(val):,} rows")
        print(f"  • Test: {len(test):,} rows")
    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        sys.exit(1)


def list_catalogs_command(args):
    """List available catalogs - direct wrapper."""
    try:
        catalogs = list_catalogs()
        print("📚 Available catalogs:")
        for catalog in catalogs:
            print(f"  • {catalog}")
    except Exception as e:
        print(f"❌ Error listing catalogs: {e}")
        sys.exit(1)


def cosmic_web_command(args):
    """Perform cosmic web analysis using integrated Data module functions."""
    print(f"🌌 Cosmic Web Analysis for survey: {args.survey}")
    
    try:
        # Use integrated cosmic web analysis
        cosmic_web_results = create_cosmic_web_loader(
            survey=args.survey,
            max_samples=args.max_samples,
            scales_mpc=args.scales,
        )
        
        print(f"✅ Analysis complete: {cosmic_web_results['n_objects']:,} objects processed")
        
        # Save results if output specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save coordinates
            coords_tensor = torch.tensor(cosmic_web_results["coordinates"])
            torch.save(coords_tensor, output_dir / f"{args.survey}_coords_3d_mpc.pt")
            
            # Save summary
            with open(output_dir / f"{args.survey}_cosmic_web_summary.txt", "w") as f:
                f.write(f"{args.survey.upper()} Cosmic Web Analysis (Integrated Data Module)\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Survey: {cosmic_web_results['survey_name']}\n")
                f.write(f"Total objects: {cosmic_web_results['n_objects']:,}\n")
                f.write(f"Total volume: {cosmic_web_results['total_volume']:.0f} Mpc³\n")
                f.write(f"Global density: {cosmic_web_results['global_density']:.2e} obj/Mpc³\n\n")
                
                f.write("Multi-scale clustering results:\n")
                for scale, result in cosmic_web_results["results_by_scale"].items():
                    f.write(f"  {scale} Mpc:\n")
                    f.write(f"    Groups: {result['n_clusters']}\n")
                    f.write(f"    Grouped: {result['grouped_fraction'] * 100:.1f}%\n")
                    f.write(f"    Time: {result['time_s']:.1f}s\n")
                    f.write(f"    Local density: {result['mean_local_density']:.2e} ± {result['density_variation']:.2e} obj/pc³\n")
                    f.write(f"    Density stats: min={result['local_density_stats']['min']:.2e}, ")
                    f.write(f"median={result['local_density_stats']['median']:.2e}, ")
                    f.write(f"max={result['local_density_stats']['max']:.2e}\n\n")
            
            print(f"💾 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error in cosmic web analysis: {e}")
        sys.exit(1)


def show_functions_command():
    """Show available functions - direct wrapper."""
    print("🔧 Available Data Module Functions:")
    print("  • load_catalog() - Load any catalog file")
    print("  • preprocess_catalog() - Clean and prepare data")
    print("  • create_training_splits() - Split data for training")
    print("  • create_graph_from_dataframe() - Create PyTorch Geometric graphs")
    print("  • create_cosmic_web_loader() - Perform cosmic web analysis")
    print("  • get_data_statistics() - Get detailed statistics")
    print("  • list_catalogs() - List available catalogs")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AstroLab Data Preprocessing CLI - Clean wrapper around astro_lab.data"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process catalog command
    process_parser = subparsers.add_parser("process", help="Process a catalog file")
    process_parser.add_argument("input", help="Input catalog file")
    process_parser.add_argument("--output", "-o", help="Output file or directory")
    process_parser.add_argument("--stats", action="store_true", help="Show statistics")
    process_parser.add_argument("--create-splits", action="store_true", help="Create train/val/test splits")
    process_parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    process_parser.add_argument("--clean-nulls", action="store_true", help="Remove null columns")
    process_parser.add_argument("--min-observations", type=int, default=100, help="Minimum observations")
    process_parser.add_argument("--magnitude-columns", help="Comma-separated magnitude columns")
    process_parser.add_argument("--coordinate-columns", help="Comma-separated coordinate columns")
    process_parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    process_parser.add_argument("--val-size", type=float, default=0.1, help="Validation split size")
    process_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    process_parser.add_argument("--shuffle", action="store_true", help="Shuffle data")
    process_parser.add_argument("--k-neighbors", type=int, default=8, help="Number of neighbors for graph")
    process_parser.add_argument("--distance-threshold", type=float, default=50.0, help="Distance threshold for graph")
    process_parser.set_defaults(func=process_catalog_command)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show catalog statistics")
    stats_parser.add_argument("input", help="Input catalog file")
    stats_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
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
    cosmic_parser = subparsers.add_parser("cosmic-web", help="Perform cosmic web analysis")
    cosmic_parser.add_argument("survey", help="Survey name (gaia, sdss, nsa, linear, tng, exoplanet)")
    cosmic_parser.add_argument("--max-samples", type=int, help="Maximum number of samples")
    cosmic_parser.add_argument("--scales", nargs="+", type=float, default=[5.0, 10.0, 20.0, 50.0], help="Analysis scales in Mpc")
    cosmic_parser.add_argument("--output", "-o", help="Output directory for results")
    cosmic_parser.set_defaults(func=cosmic_web_command)

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
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
