#!/usr/bin/env python3
"""
AstroLab Preprocessing CLI - Schlanker Wrapper um astro_lab.data

Nutzt die vorhandenen astro_lab.data Funktionen direkt ohne Redundanz.
Erzeugt automatisch PyTorch Geometric Graphs (.pt) für alle Surveys - Standard in GNNs!
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import torch
from astro_torch.data.datasets import TNG50GraphDataset, TNG50TemporalDataset

from astro_lab.data import (
    create_graph_datasets_from_splits,
    # 🔗 NEW: Graph creation functions from data module
    create_graph_from_dataframe,
    # Core functions - bereits implementiert
    create_training_splits,
    detect_survey_type,
    get_data_statistics,
    list_catalogs,
    load_catalog,
    load_splits_from_parquet,
    preprocess_catalog,
    save_splits_to_parquet,
)
from astro_lab.data.config import DataConfig
from astro_lab.data.core import load_gaia_data, load_nsa_data, load_sdss_data
from astro_lab.data.processing import SimpleAstroProcessor, SimpleProcessingConfig


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


# Graph creation functions moved to astro_lab.data.core - DRY principle!


def process_catalog_command(args):
    """Process a catalog file - schlanker Wrapper mit automatischer Graph-Erzeugung."""
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

    print(f"\n✅ Processing complete: {df_clean.shape[0]:,} rows retained")


def stats_command(args):
    """Show statistics for a catalog - direkter Wrapper."""
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
    """Load and display splits - direkter Wrapper."""
    base_path = Path(args.path)
    dataset_name = args.dataset

    try:
        train, val, test = load_splits_from_parquet(base_path, dataset_name)

        print(f"📊 Loaded splits for '{dataset_name}':")
        print(f"  • Training: {train.shape[0]:,} rows")
        print(f"  • Validation: {val.shape[0]:,} rows")
        print(f"  • Test: {test.shape[0]:,} rows")
        print(f"  • Total: {train.shape[0] + val.shape[0] + test.shape[0]:,} rows")

        if args.stats:
            stats = get_data_statistics(train)
            print("\n📈 Training set statistics:")
            print_stats(stats)

    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        sys.exit(1)


def list_catalogs_command(args):
    """List available catalogs - direkter Wrapper."""
    try:
        catalogs_df = list_catalogs()

        if len(catalogs_df) == 0:
            print("📋 No catalogs found in data directory")
            return

        print("📋 Available catalogs:")
        print(f"{'Name':<25} {'Type':<15} {'Size (MB)':<12} {'Rows':<10}")
        print("-" * 65)

        for row in catalogs_df.iter_rows(named=True):
            name = row.get("name", "N/A")[:24]
            catalog_type = row.get("type", "N/A")[:14]
            size_mb = row.get("size_mb", 0)
            n_rows = row.get("n_rows", 0)

            print(f"{name:<25} {catalog_type:<15} {size_mb:<12.1f} {n_rows:<10,}")

    except Exception as e:
        print(f"❌ Error listing catalogs: {e}")
        sys.exit(1)


def get_available_particle_types(hdf5_file: Path) -> List[str]:
    """Scans an HDF5 file and returns a list of found particle type keys."""
    if not hdf5_file.exists():
        print(f"❌ HDF5 file not found: {hdf5_file}")
        return []
    try:
        with h5py.File(hdf5_file, "r") as f:
            return [key for key in f.keys() if key.startswith("PartType")]
    except Exception as e:
        print(f"Could not read HDF5 file {hdf5_file.name}: {e}")
        return []


def process_tng50_command(args):
    """
    Process TNG50 simulation data by creating a single combined graph dataset.
    This function now leverages the updated TNG50GraphDataset class,
    which processes ALL particle types from ALL snapshots into one large graph.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Output directory: {output_dir}")
    print("🌌 Processing ALL particle types from ALL snapshots into a single graph...")

    # Define file paths for the combined dataset
    raw_parquet_file = Path("data/raw/tng50") / "tng50_combined.parquet"

    # The processed file name for the combined dataset
    temp_dataset_for_naming = TNG50GraphDataset(
        root=str(output_dir),
        radius=args.distance_threshold,
    )
    processed_graph_file = Path(temp_dataset_for_naming.processed_paths[0])

    # Handle --force flag: remove existing files
    if args.force:
        if raw_parquet_file.exists():
            raw_parquet_file.unlink()
            print(f"🗑️  Deleted existing combined raw file: {raw_parquet_file.name}")
        if processed_graph_file.exists():
            processed_graph_file.unlink()
            print(
                f"🗑️  Deleted existing combined graph file: {processed_graph_file.name}"
            )

    print("⏳ Initializing combined dataset...")
    print(f"   - Max samples: {args.max_samples}")
    print(f"   - Distance threshold (radius): {args.distance_threshold}")
    print(f"   - Expecting raw file at: {raw_parquet_file}")
    print(f"   - Expecting processed file at: {processed_graph_file}")

    # The TNG50GraphDataset will handle everything:
    # 1. __init__ is called.
    # 2. It checks for processed files.
    # 3. If not found, it calls process().
    # 4. process() checks for raw files.
    # 5. If not found, it calls download().
    # 6. download() extracts from ALL HDF5 snapshots and combines ALL particle types.
    try:
        dataset = TNG50GraphDataset(
            root=str(output_dir),
            radius=args.distance_threshold,
            max_particles=args.max_samples,
        )

        if len(dataset) > 0:
            print("✅ Successfully processed ALL particle types from ALL snapshots.")
            print(f"   Graph saved to: {dataset.processed_paths[0]}")
            print(f"   Total particles: {dataset[0].num_nodes:,}")
            print(f"   Total edges: {dataset[0].num_edges:,}")
        else:
            print("⚠️ Warning: Processing resulted in an empty dataset.")

    except FileNotFoundError as e:
        print(f"❌ Error during processing: {e}")
        print(
            "   Please ensure raw TNG50 HDF5 snapshot files are available in 'data/raw/TNG50-4/output/snapdir_099/'."
        )
    except Exception as e:
        print(f"❌ An unexpected error occurred during processing: {e}")

    print("\n✅ TNG50 processing complete.")


def process_single_tng50_snapshot(snap_file: Path, args):
    """
    DEPRECATED: This function is no longer used. The logic is now
    encapsulated within the TNG50GraphDataset class.
    """
    raise DeprecationWarning(
        "process_single_tng50_snapshot is deprecated and should not be used. "
        "Logic is now in TNG50GraphDataset."
    )


def list_tng50_snapshots_command(args):
    """List available snapshots and particle types in the TNG50 directory."""
    base_dir = (
        Path(args.directory) if args.directory else Path("data/raw/TNG50-4/output")
    )

    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        sys.exit(1)

    snap_dirs = sorted([d for d in base_dir.glob("snapdir_*") if d.is_dir()])
    if not snap_dirs:
        print(f"🤷 No 'snapdir_*' directories found in {base_dir}")
        return

    print(f"Found {len(snap_dirs)} snapshot directories in {base_dir}:")
    for snap_dir in snap_dirs:
        print(f"\n📁 {snap_dir.name}")
        snap_files = sorted(snap_dir.glob("snap_*.hdf5"))
        if snap_files:
            print(f"  - Found {len(snap_files)} HDF5 files.")
            # Show particle types from the first file
            try:
                particle_types = get_available_particle_types(snap_files[0])
                if particle_types:
                    print(f"  - Available particle types: {', '.join(particle_types)}")
                else:
                    print("  - No particle types found in snapshot.")
            except Exception as e:
                print(f"  - Could not read particle types: {e}")
        else:
            print("  - No HDF5 files found.")


def show_functions_command():
    """Show available functions in the data module."""
    print("📦 Available astro_lab.data Functions:")
    print()
    # List key functions from the data module
    funcs = [
        "load_catalog",
        "preprocess_catalog",
        "create_training_splits",
        "save_splits_to_parquet",
        "load_splits_from_parquet",
        "create_graph_from_dataframe",
        "create_graph_datasets_from_splits",
        "get_data_statistics",
        "list_catalogs",
        "load_gaia_data",
        "load_nsa_data",
        "load_sdss_data",
    ]
    for func in sorted(funcs):
        print(f"  • {func}")


def process_tng50_temporal_command(args):
    """
    Process TNG50 simulation data as temporal graphs with cosmological evolution.
    This function leverages the TNG50TemporalDataset class to create a sequence
    of spatial graphs from ALL particle types across ALL snapshots, preserving
    the temporal evolution and cosmological redshift information.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Output directory: {output_dir}")
    print("🌌 Processing TNG50 as temporal graphs with cosmological evolution...")
    print(
        f"   - Temporal edges: {'Enabled' if args.enable_temporal_edges else 'Disabled'}"
    )
    print(f"   - Temporal edge weight: {args.temporal_edge_weight}")

    # Define file paths for the temporal dataset
    raw_parquet_file = Path("data/raw/tng50") / "tng50_temporal_combined.parquet"

    # The processed file name for the temporal dataset
    temp_dataset_for_naming = TNG50TemporalDataset(
        root=str(output_dir),
        radius=args.distance_threshold,
        enable_temporal_edges=args.enable_temporal_edges,
        temporal_edge_weight=args.temporal_edge_weight,
    )
    processed_graph_file = Path(temp_dataset_for_naming.processed_paths[0])

    # Handle --force flag: remove existing files
    if args.force:
        if raw_parquet_file.exists():
            raw_parquet_file.unlink()
            print(f"🗑️  Deleted existing temporal raw file: {raw_parquet_file.name}")
        if processed_graph_file.exists():
            processed_graph_file.unlink()
            print(
                f"🗑️  Deleted existing temporal graph file: {processed_graph_file.name}"
            )

    print("⏳ Initializing temporal dataset...")
    print(f"   - Max samples per snapshot: {args.max_samples}")
    print(f"   - Distance threshold (radius): {args.distance_threshold}")
    print(f"   - Expecting raw file at: {raw_parquet_file}")
    print(f"   - Expecting processed file at: {processed_graph_file}")

    # The TNG50TemporalDataset will handle everything:
    # 1. __init__ is called.
    # 2. It checks for processed files.
    # 3. If not found, it calls process().
    # 4. process() checks for raw files.
    # 5. If not found, it calls download().
    # 6. download() extracts from ALL HDF5 snapshots with temporal info.
    # 7. process() creates separate graphs for each snapshot with temporal edges.
    try:
        dataset = TNG50TemporalDataset(
            root=str(output_dir),
            radius=args.distance_threshold,
            max_particles=args.max_samples,
            enable_temporal_edges=args.enable_temporal_edges,
            temporal_edge_weight=args.temporal_edge_weight,
        )

        if len(dataset) > 0:
            print("✅ Successfully processed TNG50 as temporal graphs.")
            print(f"   Temporal graphs saved to: {dataset.processed_paths[0]}")
            print(f"   Number of temporal graphs: {len(dataset)}")

            # Show info about each temporal graph
            total_particles = 0
            total_edges = 0
            for i, graph in enumerate(dataset):
                particles = graph.num_nodes
                edges = graph.num_edges
                redshift = graph.redshift.item()
                time_gyr = graph.time_gyr.item()
                total_particles += particles
                total_edges += edges
                print(
                    f"   Graph {i}: {particles:,} particles, {edges:,} edges, z={redshift:.2f}, {time_gyr:.1f} Gyr ago"
                )

            print(f"   Total particles across all snapshots: {total_particles:,}")
            print(f"   Total edges across all snapshots: {total_edges:,}")

            if args.enable_temporal_edges:
                print("   Temporal edges enabled between consecutive snapshots")
        else:
            print("⚠️ Warning: Processing resulted in an empty temporal dataset.")

    except FileNotFoundError as e:
        print(f"❌ Error during processing: {e}")
        print(
            "   Please ensure raw TNG50 HDF5 snapshot files are available in 'data/raw/TNG50-4/output/snapdir_099/'."
        )
    except Exception as e:
        print(f"❌ An unexpected error occurred during processing: {e}")

    print("\n✅ TNG50 temporal processing complete.")


def preprocess_data(
    survey: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_samples: int = 10000,
    enable_features: bool = True,
    enable_clustering: bool = False,
    enable_statistics: bool = False,
    enable_crossmatch: bool = False,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    force: bool = False,
    particle_type: Optional[str] = None,
    all_snapshots: bool = False,
) -> None:
    """
    Hauptfunktion für das Preprocessing von verschiedenen astronomischen Surveys.
    Diese Funktion automatisiert das Laden, Verarbeiten und Speichern von Daten
    für Surveys wie Gaia, NSA, SDSS und TNG50.
    """
    print(f"🚀 Starting preprocessing for survey: {survey.upper()}")

    # Lade Konfiguration
    if config_path:
        data_config = DataConfig.from_yaml(config_path)
    else:
        # Fallback auf eine Standardkonfiguration, falls kein Pfad angegeben ist
        data_config = DataConfig()
        print("⚠️ No config path provided, using default DataConfig.")

    # Setze das Output-Verzeichnis
    if output_dir:
        data_config.set_output_dir(output_dir)
    output_path = Path(data_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory set to: {output_path}")

    # Survey-spezifische Logik
    if survey == "tng50":
        # Hier wird direkt die neue, refaktorisierte TNG50-Logik genutzt
        # Wir erstellen ein "args"-ähnliches Objekt, um die Funktion wiederzuverwenden
        tng50_args = argparse.Namespace(
            output_dir=str(output_path),
            particle_type=particle_type,
            max_samples=max_samples,
            k_neighbors=k_neighbors,
            distance_threshold=distance_threshold,
            force=force,
            all_snapshots=all_snapshots,
        )
        process_tng50_command(tng50_args)
        return

    # Lade Daten für andere Surveys
    try:
        if survey == "gaia":
            df = load_gaia_data(max_g_mag=18, max_rows=max_samples)
        elif survey == "nsa":
            df = load_nsa_data(max_rows=max_samples)
        elif survey == "sdss":
            df = load_sdss_data(max_rows=max_samples)
        else:
            print(f"❌ Unknown survey: {survey}")
            return
        print(f"✅ Loaded {len(df):,} rows for {survey.upper()} survey.")
    except Exception as e:
        print(f"❌ Failed to load data for {survey.upper()}: {e}")
        return

    # Datenverarbeitung
    processor_config = SimpleProcessingConfig(
        k_neighbors=k_neighbors, distance_threshold=distance_threshold
    )
    processor = SimpleAstroProcessor(config=processor_config)
    processed_data = processor.process(
        dataframe=df,
        enable_features=enable_features,
        enable_clustering=enable_clustering,
        enable_statistics=enable_statistics,
        enable_crossmatch=enable_crossmatch,
    )
    print("✅ Data processing complete.")

    # Speichere die Ergebnisse
    # Der Prozessor gibt ein Dictionary mit den Ergebnissen zurück.
    # Wir speichern jedes Ergebnis als separate Parquet-Datei.
    for key, result_df in processed_data.items():
        if result_df is None or result_df.is_empty():
            continue

        file_name = f"{survey}_{key}.parquet"
        file_path = output_path / file_name

        if force and file_path.exists():
            file_path.unlink()
            print(f"🗑️  Deleted existing file: {file_path.name}")

        result_df.write_parquet(file_path)
        print(f"💾 Saved '{key}' data to {file_path}")

    print("\n🎉 Preprocessing finished successfully!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AstroLab Data Preprocessing CLI")
    parser.add_argument(
        "-v", "--version", action="version", version="AstroLab CLI 0.1.0"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # process command
    process_parser = subparsers.add_parser(
        "process",
        help="Load, preprocess, and save a catalog file, including graph generation.",
    )
    process_parser.add_argument("input", help="Path to the input catalog file")
    process_parser.add_argument("--output", help="Path to the output file or directory")
    process_parser.add_argument(
        "--stats", action="store_true", help="Show statistics of the original data"
    )
    process_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files"
    )
    process_parser.add_argument(
        "--clean-nulls",
        type=float,
        default=0.9,
        help="Threshold for dropping columns with nulls",
    )
    process_parser.add_argument(
        "--min-observations", type=int, default=10, help="Minimum observations filter"
    )
    process_parser.add_argument(
        "--magnitude-columns", help="Comma-separated magnitude column names"
    )
    process_parser.add_argument(
        "--coordinate-columns", help="Comma-separated coordinate column names"
    )
    process_parser.add_argument(
        "--create-splits", action="store_true", help="Create train/val/test splits"
    )
    process_parser.add_argument("--test-size", type=float, default=0.2)
    process_parser.add_argument("--val-size", type=float, default=0.2)
    process_parser.add_argument("--random-state", type=int, default=42)
    process_parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle data before splitting"
    )
    process_parser.set_defaults(func=process_catalog_command)

    # stats command
    stats_parser = subparsers.add_parser(
        "stats", help="Show detailed statistics for a catalog file"
    )
    stats_parser.add_argument("input", help="Path to the catalog file to analyze")
    stats_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed column info"
    )
    stats_parser.set_defaults(func=stats_command)

    # load-splits command
    load_splits_parser = subparsers.add_parser(
        "load-splits", help="Load and display information about saved splits"
    )
    load_splits_parser.add_argument("path", help="Path to the directory with splits")
    load_splits_parser.add_argument("dataset", help="Base name of the dataset")
    load_splits_parser.add_argument(
        "--stats", action="store_true", help="Show stats for the training split"
    )
    load_splits_parser.set_defaults(func=load_splits_command)

    # list-catalogs command
    list_catalogs_parser = subparsers.add_parser(
        "list-catalogs", help="List available raw catalogs"
    )
    list_catalogs_parser.set_defaults(func=list_catalogs_command)

    # Unified 'run' command for different surveys
    run_parser = subparsers.add_parser(
        "run", help="Run preprocessing for a specific survey (e.g., gaia, nsa, tng50)"
    )
    run_parser.add_argument("survey", choices=["gaia", "nsa", "sdss", "tng50"])
    run_parser.add_argument(
        "--config-path", help="Path to the data configuration YAML."
    )
    run_parser.add_argument("--output-dir", help="Directory to save processed data.")
    run_parser.add_argument(
        "--max-samples", type=int, default=10000, help="Max rows to load."
    )
    run_parser.add_argument(
        "--k-neighbors",
        type=int,
        default=8,
        help="Number of nearest neighbors for graph construction.",
    )
    run_parser.add_argument(
        "--distance-threshold",
        type=float,
        default=50.0,
        help="Max distance for graph connections.",
    )
    run_parser.add_argument("--force", action="store_true", help="Force reprocessing.")
    # TNG50 specific args for the 'run' command
    run_parser.add_argument(
        "--particle-type",
        default="PartType4",
        help="TNG50 particle type to process.",
    )
    run_parser.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Process all TNG50 particle types.",
    )
    run_parser.set_defaults(
        func=lambda args: preprocess_data(
            survey=args.survey,
            config_path=args.config_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            k_neighbors=args.k_neighbors,
            distance_threshold=args.distance_threshold,
            force=args.force,
            particle_type=args.particle_type,
            all_snapshots=args.all_snapshots,
        )
    )

    # Legacy survey-specific commands (will be deprecated later)
    gaia_parser = subparsers.add_parser("gaia", help="Process Gaia data.")
    nsa_parser = subparsers.add_parser("nsa", help="Process NSA catalog data.")
    sdss_parser = subparsers.add_parser("sdss", help="Process SDSS spectral data.")
    exoplanet_parser = subparsers.add_parser(
        "exoplanet", help="Process NASA exoplanet data."
    )

    # tng50 (legacy, but now refactored)
    tng50_parser = subparsers.add_parser(
        "tng50", help="Process TNG50 simulation data into a single combined graph."
    )
    tng50_parser.add_argument(
        "--output-dir",
        default="data/results/tng50",
        help="Directory to save processed data.",
    )
    tng50_parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of particles to sample (total across all types/snapshots).",
    )
    tng50_parser.add_argument(
        "-k",
        "--k-neighbors",
        type=int,
        default=8,
        help="Number of nearest neighbors (not used for TNG50 radius search).",
    )
    tng50_parser.add_argument(
        "--distance-threshold",
        type=float,
        default=1.0,  # Default radius for TNG50
        help="Connection radius for graph construction (in simulation units).",
    )
    tng50_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing and overwrite existing files.",
    )
    tng50_parser.set_defaults(func=process_tng50_command)

    # tng50-temporal (new temporal graph version)
    tng50_temporal_parser = subparsers.add_parser(
        "tng50-temporal",
        help="Process TNG50 simulation data as temporal graphs with cosmological evolution.",
    )
    tng50_temporal_parser.add_argument(
        "--output-dir",
        default="data/processed/tng50_temporal",
        help="Directory to save processed temporal data.",
    )
    tng50_temporal_parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of particles per snapshot (total across all types).",
    )
    tng50_temporal_parser.add_argument(
        "--distance-threshold",
        type=float,
        default=1.0,  # Default radius for TNG50
        help="Connection radius for graph construction (in simulation units).",
    )
    tng50_temporal_parser.add_argument(
        "--enable-temporal-edges",
        action="store_true",
        default=True,
        help="Enable temporal edges between consecutive snapshots.",
    )
    tng50_temporal_parser.add_argument(
        "--temporal-edge-weight",
        type=float,
        default=1.0,
        help="Weight for temporal edges between snapshots.",
    )
    tng50_temporal_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing and overwrite existing files.",
    )
    tng50_temporal_parser.set_defaults(func=process_tng50_temporal_command)

    # list-tng50-snapshots
    list_tng50_parser = subparsers.add_parser(
        "list-tng50-snapshots",
        help="List available TNG50 snapshot files and particle types.",
    )
    list_tng50_parser.add_argument(
        "-d",
        "--directory",
        help="Directory to search for TNG50 snapshots (default: data/raw/TNG50-4/output).",
    )
    list_tng50_parser.set_defaults(func=list_tng50_snapshots_command)

    # show-functions
    show_functions_parser = subparsers.add_parser(
        "show-functions", help="Show available data processing functions."
    )
    show_functions_parser.set_defaults(func=show_functions_command)

    # Add shared arguments to relevant subparsers
    for p in [
        process_parser,
        gaia_parser,
        nsa_parser,
        sdss_parser,
        exoplanet_parser,
    ]:
        p.add_argument(
            "-k",
            "--k-neighbors",
            type=int,
            default=8,
            help="Number of nearest neighbors for graph construction.",
        )
        p.add_argument(
            "--distance-threshold",
            type=float,
            default=50.0,
            help="Maximum distance for graph connections.",
        )

    args = parser.parse_args()

    # Check for dependencies before running any command
    try:
        import astro_torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError:
        print("❌ Missing essential dependencies.")
        print("   Please install them by running: uv pip install torch torch-geometric")
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
