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

import torch

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
    """Get available particle types from TNG50 HDF5 file."""
    try:
        import h5py

        particle_types = []
        with h5py.File(hdf5_file, "r") as f:
            for key in f.keys():
                if key.startswith("PartType"):
                    try:
                        coords = f[key]["Coordinates"]  # type: ignore
                        n_particles = len(coords)  # type: ignore
                        if n_particles > 0:
                            particle_types.append(key)
                    except:
                        pass  # Skip particle types without proper structure

        return sorted(particle_types)
    except ImportError:
        print("❌ h5py required to scan particle types")
        return []
    except Exception as e:
        print(f"❌ Error scanning particle types: {e}")
        return []


def process_tng50_command(args):
    """Process TNG50 data - spezialisierte Funktion."""
    try:
        from astro_torch.data.datasets import TNG50GraphDataset

        # Handle --all-snapshots mode
        if args.all_snapshots:
            base_dir = (
                Path(args.input) if args.input else Path("data/raw/TNG50-4/output")
            )

            if not base_dir.exists():
                print(f"❌ Base directory not found: {base_dir}")
                sys.exit(1)

            # Find all snapshots
            snap_files = []
            for snapdir in base_dir.glob("snapdir_*"):
                if snapdir.is_dir():
                    snap_files.extend(snapdir.glob("snap_*.hdf5"))

            if not snap_files:
                print(f"❌ No snapshot files found in: {base_dir}")
                sys.exit(1)

            snap_files.sort()
            print(f"🌌 Processing {len(snap_files)} TNG50 snapshots into graphs")

            success_count = 0
            for i, snap_file in enumerate(snap_files, 1):
                print(f"\n📊 Snapshot {i}/{len(snap_files)}: {snap_file.name}")
                try:
                    process_single_tng50_snapshot(snap_file, args)
                    success_count += 1
                except Exception as e:
                    print(f"❌ Error processing {snap_file.name}: {e}")

            print(
                f"\n✅ Batch processing complete: {success_count}/{len(snap_files)} snapshots processed"
            )
            return

        # Single snapshot mode
        if not args.input:
            print("❌ Input snapshot file required (or use --all-snapshots)")
            sys.exit(1)

        snap_file = Path(args.input)
        if not snap_file.exists():
            print(f"❌ Snapshot file not found: {snap_file}")
            sys.exit(1)

        print(f"🌌 Processing TNG50 snapshot into graphs: {snap_file.name}")
        process_single_tng50_snapshot(snap_file, args)

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Please install: uv add torch torch-geometric")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing TNG50 graphs: {e}")
        sys.exit(1)


def process_single_tng50_snapshot(snap_file: Path, args):
    """Process a single TNG50 snapshot."""
    from astro_torch.data.datasets import TNG50GraphDataset

    # Determine particle types to process
    if args.all:
        print("🔍 Scanning snapshot for all available particle types...")
        all_particle_types = get_available_particle_types(snap_file)
        if not all_particle_types:
            print("❌ No particle types found in snapshot")
            sys.exit(1)
        particle_types = all_particle_types
        print(f"📋 Found particle types: {', '.join(particle_types)}")
    else:
        particle_types = [p.strip() for p in args.particle_types.split(",")]

    # English particle type names
    particle_names = {
        "PartType0": "gas",
        "PartType1": "dark_matter",
        "PartType4": "stars",
        "PartType5": "black_holes",
    }

    for ptype in particle_types:
        if ptype not in particle_names:
            print(f"⚠️  Unknown particle type: {ptype}")
            continue

        english_name = particle_names[ptype]
        print(f"  🔄 {english_name.title()} ({ptype})")

        # Create output directory
        if args.output:
            output_dir = Path(args.output) / english_name
        else:
            output_dir = Path("data/processed/tng50_graphs") / english_name

        output_dir.mkdir(parents=True, exist_ok=True)

        # Force mode: delete existing processed files
        if args.force:
            processed_dir = output_dir / "processed"
            if processed_dir.exists():
                for pt_file in processed_dir.glob("*.pt"):
                    pt_file.unlink()
                    print(f"🗑️  Deleted existing graph: {pt_file.name}")

        # Create TNG50 graph dataset
        dataset = TNG50GraphDataset(
            root=str(output_dir),
            snapshot_file=str(snap_file),
            particle_type=ptype,
            radius=args.radius,
            max_particles=args.max_particles,
        )

        print(f"    ✅ Graph saved to: {output_dir}")

        if args.stats and len(dataset) > 0:
            data = dataset[0]
            print(f"    📊 Nodes: {data.num_nodes:,}")  # type: ignore
            print(f"    📊 Edges: {data.num_edges:,}")  # type: ignore
            print(f"    📊 Features: {data.x.shape[1] if data.x is not None else 0}")  # type: ignore

    print("✅ TNG50 graph processing complete!")


def list_tng50_snapshots_command(args):
    """List available TNG50 snapshot files."""
    tng50_dir = (
        Path(args.directory) if args.directory else Path("data/raw/TNG50-4/output")
    )

    if not tng50_dir.exists():
        print(f"❌ TNG50 directory not found: {tng50_dir}")
        sys.exit(1)

    print(f"📂 TNG50 snapshots in: {tng50_dir}")

    # Search recursively for snapshot files
    snap_files = []
    for snapdir in tng50_dir.glob("snapdir_*"):
        if snapdir.is_dir():
            snap_files.extend(snapdir.glob("snap_*.hdf5"))

    if not snap_files:
        print("❌ No TNG50 snapshot files found")
        print("💡 Tried searching in snapdir_* subdirectories")
        return

    snap_files.sort()

    print(f"\n🌌 Found {len(snap_files)} snapshot files:")
    for snap_file in snap_files:
        size_mb = snap_file.stat().st_size / (1024 * 1024)
        print(f"  • {snap_file.name} ({size_mb:.0f} MB)")

    if args.inspect:
        print(f"\n🔍 Inspecting first snapshot: {snap_files[0].name}")
        try:
            import h5py

            with h5py.File(snap_files[0], "r") as f:
                # Show header information if available
                if "Header" in f:
                    print("  📊 Simulation info: Header available")

                # Show available particle types
                print("  📋 Available particle types:")
                for key in f.keys():
                    if key.startswith("PartType"):
                        try:
                            coords = f[key]["Coordinates"]  # type: ignore
                            n_particles = len(coords)  # type: ignore
                            print(f"    • {key}: {n_particles:,} particles")
                        except:
                            print(f"    • {key}: (structure varies)")

        except ImportError:
            print("  ⚠️  h5py not available for inspection")
        except Exception as e:
            print(f"  ❌ Error inspecting snapshot: {e}")


def show_functions_command():
    """Show available astro_lab.data functions."""
    print("📦 Available astro_lab.data Functions:")
    print()
    print("🔧 Preprocessing:")
    print("  • preprocess_catalog(df)")
    print("  • create_training_splits(df)")
    print("  • save_splits_to_parquet(train, val, test, path, name)")
    print("  • load_splits_from_parquet(path, name)")
    print("  • get_data_statistics(df)")
    print()
    print("🌌 TNG50 Simulation:")
    print("  • TNG50GraphDataset (optimized graph processing)")
    print("  • create_tng50_dataloader()")
    print("  • AstroDataManager.import_tng50_hdf5() (low-level)")
    print("  • import_tng50() (legacy support)")
    print()
    print("📊 Data Management:")
    print("  • AstroDataManager()")
    print("  • list_catalogs()")
    print("  • load_catalog(path)")
    print("  • download_gaia(region, magnitude_limit)")
    print("  • download_bright_all_sky(magnitude_limit)")
    print()
    print("📈 PyTorch Geometric Datasets:")
    print("  • GaiaGraphDataset, NSAGraphDataset")
    print("  • ExoplanetGraphDataset, RRLyraeDataset")
    print("  • SatelliteOrbitDataset, SDSSSpectralDataset")
    print("  • create_*_dataloader() functions")
    print()
    print("🔬 Transformations:")
    print("  • AddAstronomicalColors, AddDistanceFeatures")
    print("  • CoordinateSystemTransform")
    print("  • get_stellar_transforms(), get_galaxy_transforms()")


def preprocess_data(
    survey: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_samples: int = 10000,
    enable_features: bool = True,
    enable_clustering: bool = False,
    enable_statistics: bool = False,
    enable_crossmatch: bool = False,
) -> None:
    """
    Preprocess astronomical data with advanced tensor operations.

    Parameters
    ----------
    survey : str
        Survey name (gaia, sdss, nsa)
    config_path : str, optional
        Path to configuration file
    output_dir : str, optional
        Output directory for results
    max_samples : int
        Maximum number of samples to process
    enable_features : bool
        Enable feature engineering
    enable_clustering : bool
        Enable clustering analysis
    enable_statistics : bool
        Enable statistical analysis
    enable_crossmatch : bool
        Enable cross-matching
    """
    print(f"🚀 Starting advanced preprocessing for {survey}")

    # Use simplified processing
    from ..data.processing import SimpleAstroProcessor, SimpleProcessingConfig

    # Create processing config
    processing_config = SimpleProcessingConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1000,
        enable_feature_engineering=enable_features,
        enable_clustering=enable_clustering,
        enable_statistics=enable_statistics,
    )

    # Create processor
    processor = SimpleAstroProcessor(processing_config)

    # Load survey data as tensor
    print(f"📊 Loading {survey} data...")
    try:
        if survey.lower() == "gaia":
            survey_tensor = load_gaia_data(max_samples=max_samples, return_tensor=True)
        elif survey.lower() == "sdss":
            survey_tensor = load_sdss_data(max_samples=max_samples, return_tensor=True)
        elif survey.lower() == "nsa":
            survey_tensor = load_nsa_data(max_samples=max_samples, return_tensor=True)
        elif survey.lower() == "exoplanet":
            print("🪐 Exoplanet cosmic web analysis - use direct script:")
            print("   python process_exoplanet_cosmic_web.py")
            return
        elif survey.lower() == "tng50":
            print("🌌 TNG50 simulation processing - using integrated tensor system:")
            # Use the new integrated TNG50 tensor loading
            from astro_lab.data.core import load_tng50_data

            # Set output directory for TNG50 first
            if output_dir is None:
                output_dir_path = Path("results") / survey / "preprocessing"
            else:
                output_dir_path = Path(output_dir)

            output_dir_path.mkdir(parents=True, exist_ok=True)

            # Process all available particle types
            particle_types = ["PartType0", "PartType1", "PartType4", "PartType5"]
            all_results = {}

            for particle_type in particle_types:
                try:
                    print(f"\n🔄 Processing {particle_type}...")
                    survey_tensor = load_tng50_data(
                        max_samples=max_samples,
                        particle_type=particle_type,
                        return_tensor=True,
                    )

                    # Process with tensor operations
                    results = processor.process(survey_tensor)
                    all_results[particle_type] = results

                    print(
                        f"   ✅ {particle_type}: {len(survey_tensor):,} particles processed"
                    )

                except FileNotFoundError:
                    print(f"   ⚠️ {particle_type}: Data not found, skipping")
                    continue
                except Exception as e:
                    print(f"   ❌ {particle_type}: Error - {e}")
                    continue

            if all_results:
                print(
                    f"\n✅ TNG50 processing complete: {len(all_results)} particle types processed"
                )

                # Save combined results
                results_file = output_dir_path / "tng50_preprocessing_results.txt"
                with open(results_file, "w") as f:
                    f.write("TNG50 Preprocessing Results\n")
                    f.write("=" * 30 + "\n\n")
                    for ptype, results in all_results.items():
                        f.write(f"{ptype}:\n")
                        if "feature_tensor" in results:
                            f.write(f"  Features: {results['n_features']}\n")
                        if "cluster_tensor" in results:
                            f.write(f"  Clusters: {results['n_clusters']}\n")
                        if "stats_tensor" in results:
                            f.write(f"  Statistics: {results['n_functions']}\n")
                        f.write("\n")

                print(f"   📄 Results saved to: {results_file}")
            else:
                print("❌ No TNG50 data could be processed")
            return
        else:
            raise ValueError(f"Unknown survey: {survey}")
    except Exception as e:
        print(f"❌ Failed to load {survey} data: {e}")
        return

    # Set output directory
    if output_dir is None:
        output_dir_path = Path("results") / survey / "preprocessing"
    else:
        output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Process the data
    print("🔬 Processing tensor data...")
    results = processor.process(survey_tensor)

    # Print summary
    print(f"\n✅ Preprocessing completed for {survey}")
    print(f"   Survey: {survey_tensor.survey_name}")
    print(f"   Objects processed: {len(survey_tensor)}")
    print(f"   Data shape: {survey_tensor._data.shape}")
    print(f"   Results saved to: {output_dir}")

    # Print detailed results
    if "feature_tensor" in results:
        print(f"   ✅ Feature engineering: {results['n_features']} features")
    if "feature_error" in results:
        print(f"   ⚠️ Feature engineering failed: {results['feature_error']}")

    if "cluster_tensor" in results:
        print(
            f"   ✅ Clustering: {results['n_clusters']} clusters, {results['n_noise']} noise points"
        )
    if "clustering_error" in results:
        print(f"   ⚠️ Clustering failed: {results['clustering_error']}")

    if "stats_tensor" in results:
        print(f"   ✅ Statistics: {results['n_functions']} functions computed")
    if "statistics_error" in results:
        print(f"   ⚠️ Statistics failed: {results['statistics_error']}")

    # Save basic results
    results_file = output_dir / f"{survey}_preprocessing_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Preprocessing Results for {survey}\n")
        f.write(f"Survey: {survey_tensor.survey_name}\n")
        f.write(f"Objects: {len(survey_tensor)}\n")
        f.write(f"Shape: {survey_tensor._data.shape}\n")
        f.write(f"Device: {survey_tensor._data.device}\n")

    print(f"   📄 Summary saved to: {results_file}")


def main():
    """Main preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="Advanced astronomical data preprocessing with tensor operations"
    )

    parser.add_argument(
        "survey",
        choices=["gaia", "sdss", "nsa", "exoplanet", "tng50"],
        help="Survey to preprocess",
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")

    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples to process",
    )

    parser.add_argument(
        "--no-features", action="store_true", help="Disable feature engineering"
    )

    parser.add_argument(
        "--enable-clustering", action="store_true", help="Enable clustering analysis"
    )

    parser.add_argument(
        "--enable-statistics", action="store_true", help="Enable statistical analysis"
    )

    parser.add_argument(
        "--enable-crossmatch",
        action="store_true",
        help="Enable cross-matching capabilities",
    )

    args = parser.parse_args()

    preprocess_data(
        survey=args.survey,
        config_path=args.config,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        enable_features=not args.no_features,
        enable_clustering=args.enable_clustering,
        enable_statistics=args.enable_statistics,
        enable_crossmatch=args.enable_crossmatch,
    )


if __name__ == "__main__":
    main()
