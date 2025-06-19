#!/usr/bin/env python3
"""
AstroLab Preprocessing CLI - Optimiert

Nutzt die vorhandenen astro_lab.data Funktionen optimal.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl

from astro_lab.data import (
    # Core processing functions
    create_training_splits,
    get_data_statistics,
    load_splits_from_parquet,
    preprocess_catalog,
    save_splits_to_parquet,
    # Data management
    data_manager,
    list_catalogs,
    load_catalog,
)
# TNG50Loader removed - using data module functions instead


def get_available_particle_types(hdf5_file: Path) -> List[str]:
    """Get all available particle types from a TNG50 HDF5 file."""
    try:
        import h5py
        
        particle_types = []
        with h5py.File(hdf5_file, 'r') as f:
            for key in f.keys():
                if key.startswith('PartType'):
                    # Check if this particle type has data (same logic as in list command)
                    try:
                        coords = f[key]['Coordinates']  # type: ignore
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


def process_catalog_command(args):
    """Process a catalog file with preprocessing."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"📂 Loading catalog: {input_path}")
    
    # Use data manager to load catalog
    try:
        df = load_catalog(input_path)
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        sys.exit(1)
    
    print(f"📊 Original data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Get statistics if requested
    if args.stats:
        print("\n📈 Original data statistics:")
        stats = get_data_statistics(df)
        print(f"  • Rows: {stats['n_rows']:,}")
        print(f"  • Columns: {stats['n_columns']}")
        print(f"  • Memory: {stats['memory_usage_mb']:.1f} MB")
        print(f"  • Numeric columns: {len(stats['numeric_columns'])}")
        if stats['missing_data']:
            print(f"  • Columns with missing data: {len(stats['missing_data'])}")
    
    # Preprocess the catalog using data module function
    magnitude_cols = args.magnitude_columns.split(',') if args.magnitude_columns else None
    coordinate_cols = args.coordinate_columns.split(',') if args.coordinate_columns else None
    
    df_clean = preprocess_catalog(
        df,
        clean_null_columns=args.clean_nulls,
        min_observations=args.min_observations,
        magnitude_columns=magnitude_cols,
        coordinate_columns=coordinate_cols
    )
    
    print(f"📈 Processed data: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
    
    # Create training splits if requested
    if args.create_splits:
        print("\n🔄 Creating training splits...")
        train, val, test = create_training_splits(
            df_clean,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            shuffle=args.shuffle
        )
        
        # Save splits if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            dataset_name = input_path.stem
            
            # Check if force mode and delete existing splits
            if args.force:
                split_files = [
                    output_path / f"{dataset_name}_train.parquet",
                    output_path / f"{dataset_name}_val.parquet", 
                    output_path / f"{dataset_name}_test.parquet"
                ]
                for split_file in split_files:
                    if split_file.exists():
                        split_file.unlink()
                        print(f"🗑️  Deleted existing: {split_file.name}")
            
            save_splits_to_parquet(train, val, test, output_path, dataset_name)
            print(f"✅ Splits saved to: {output_path}")
        else:
            print("💡 Use --output to save splits to disk")
    
    # Save processed catalog if output path provided and no splits
    elif args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_file = output_path / f"{input_path.stem}_processed.parquet"
        else:
            output_file = output_path
        
        # Check if force mode and delete existing file
        if args.force and output_file.exists():
            output_file.unlink()
            print(f"🗑️  Deleted existing: {output_file.name}")
        
        df_clean.write_parquet(output_file)
        print(f"💾 Processed catalog saved to: {output_file}")
    
    print(f"\n✅ Processing complete: {df_clean.shape[0]:,} rows retained")


def stats_command(args):
    """Show statistics for a catalog file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"📂 Analyzing: {input_path}")
    
    # Use data manager to load catalog
    try:
        df = load_catalog(input_path)
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        sys.exit(1)
    
    # Get and display statistics using data module function
    stats = get_data_statistics(df)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  • Rows: {stats['n_rows']:,}")
    print(f"  • Columns: {stats['n_columns']}")
    print(f"  • Memory usage: {stats['memory_usage_mb']:.1f} MB")
    print(f"  • Numeric columns: {len(stats['numeric_columns'])}")
    
    if args.verbose:
        print(f"\n📋 Column details:")
        for col in stats['columns'][:20]:  # Show first 20 columns
            dtype = stats['dtypes'][col]
            print(f"  • {col}: {dtype}")
        
        if len(stats['columns']) > 20:
            print(f"  ... and {len(stats['columns']) - 20} more columns")
    
    if stats['missing_data']:
        print(f"\n🚨 Missing data:")
        for col, info in list(stats['missing_data'].items())[:10]:  # Show first 10
            print(f"  • {col}: {info['null_count']:,} ({info['null_percentage']:.1f}%)")
        
        if len(stats['missing_data']) > 10:
            print(f"  ... and {len(stats['missing_data']) - 10} more columns with missing data")


def load_splits_command(args):
    """Load and display information about saved splits."""
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
            print(f"\n📈 Training set statistics:")
            stats = get_data_statistics(train)
            print(f"  • Columns: {stats['n_columns']}")
            print(f"  • Memory: {stats['memory_usage_mb']:.1f} MB")
            print(f"  • Numeric columns: {len(stats['numeric_columns'])}")
        
    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        sys.exit(1)


def process_tng50_command(args):
    """Process TNG50 data into graph format (.pt files)."""
    try:
        from astro_lab.data.datasets.astronomical import TNG50GraphDataset
        
        # Handle --all-snapshots mode
        if args.all_snapshots:
            if args.input:
                base_dir = Path(args.input)
            else:
                base_dir = Path("data/raw/TNG50-4/output")
            
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
            
            print(f"\n✅ Batch processing complete: {success_count}/{len(snap_files)} snapshots processed")
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
    from astro_lab.data.datasets.astronomical import TNG50GraphDataset
    
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
        particle_types = [p.strip() for p in args.particle_types.split(',')]
    
    # English particle type names
    particle_names = {
        "PartType0": "gas",
        "PartType1": "dark_matter", 
        "PartType4": "stars",
        "PartType5": "black_holes"
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
        
        # If force mode, delete existing processed files
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
            max_particles=args.max_particles
        )
        
        # The dataset will automatically process and save as .pt file
        print(f"    ✅ Graph saved to: {output_dir}")
        
        if args.stats and len(dataset) > 0:
            data = dataset[0]
            print(f"    📊 Nodes: {data.num_nodes:,}")  # type: ignore
            print(f"    📊 Edges: {data.num_edges:,}")  # type: ignore
            print(f"    📊 Features: {data.x.shape[1] if data.x is not None else 0}")  # type: ignore
    
    print(f"✅ TNG50 graph processing complete!")


def list_tng50_snapshots_command(args):
    """List available TNG50 snapshot files."""
    tng50_dir = Path(args.directory) if args.directory else Path("data/raw/TNG50-4/output")
    
    if not tng50_dir.exists():
        print(f"❌ TNG50 directory not found: {tng50_dir}")
        sys.exit(1)
    
    print(f"📂 TNG50 snapshots in: {tng50_dir}")
    
    # Search recursively for snapshot files in snapdir_* subdirectories
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
            # Use AstroDataManager for inspection - it has robust h5py handling
            from astro_lab.data import AstroDataManager
            manager = AstroDataManager()
            
            import h5py
            with h5py.File(snap_files[0], 'r') as f:
                # Show header information if available
                if 'Header' in f:
                    header = f['Header']
                    print(f"  📊 Simulation info:")
                    print(f"    • Box size: Available in header")
                    print(f"    • Redshift: Available in header") 
                    print(f"    • Hubble param: Available in header")
                
                # Show available particle types
                print("  📋 Available particle types:")
                for key in f.keys():
                    if key.startswith('PartType'):
                        try:
                            coords = f[key]['Coordinates']  # type: ignore
                            n_particles = len(coords)  # type: ignore
                            print(f"    • {key}: {n_particles:,} particles")
                        except:
                            print(f"    • {key}: (structure varies)")
                            
        except ImportError:
            print("  ⚠️  h5py not available for inspection")
        except Exception as e:
            print(f"  ❌ Error inspecting snapshot: {e}")


def list_catalogs_command(args):
    """List available catalogs using data manager."""
    try:
        catalogs_df = list_catalogs()
        
        if len(catalogs_df) == 0:
            print("📋 No catalogs found in data directory")
            return
        
        print("📋 Available catalogs:")
        print(f"{'Name':<25} {'Type':<15} {'Size (MB)':<12} {'Rows':<10}")
        print("-" * 65)
        
        for row in catalogs_df.iter_rows(named=True):
            name = row.get('name', 'N/A')[:24]
            catalog_type = row.get('type', 'N/A')[:14]
            size_mb = row.get('size_mb', 0)
            n_rows = row.get('n_rows', 0)
            
            print(f"{name:<25} {catalog_type:<15} {size_mb:<12.1f} {n_rows:<10,}")
            
    except Exception as e:
        print(f"❌ Error listing catalogs: {e}")


def main():
    """Main CLI function with subcommands."""
    parser = argparse.ArgumentParser(
        description="🚀 AstroLab Preprocessing CLI - Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 Examples:

# Process a catalog with statistics
astro-lab preprocess process catalog.parquet --stats --create-splits --output processed/

# Force reprocessing even if files exist
astro-lab preprocess process catalog.parquet --force --output processed/

# Show catalog statistics
astro-lab preprocess stats catalog.parquet --verbose

# Process TNG50 data into optimized graphs
astro-lab preprocess tng50 snap_099.0.hdf5 --particle-types PartType4,PartType5 --max-particles 1000

# Process ALL particle types in TNG50 snapshot
astro-lab preprocess tng50 snap_099.0.hdf5 --all --max-particles 5000 --radius 2.0

# Process ALL snapshots in directory
astro-lab preprocess tng50 --all-snapshots --all --max-particles 2000

# Force TNG50 reprocessing
astro-lab preprocess tng50 snap_099.0.hdf5 --force --particle-types PartType4,PartType5

# List available catalogs
astro-lab preprocess list-catalogs

# List TNG50 snapshots
astro-lab preprocess tng50-list --inspect
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process and clean a catalog')
    process_parser.add_argument('input', help='Input catalog file (.parquet or .csv)')
    process_parser.add_argument('-o', '--output', help='Output directory or file')
    process_parser.add_argument('--force', action='store_true',
                               help='Force reprocessing even if output files exist')
    process_parser.add_argument('--clean-nulls', action='store_true', default=True,
                               help='Remove columns with >95%% null values')
    process_parser.add_argument('--min-observations', type=int,
                               help='Minimum number of valid observations per row')
    process_parser.add_argument('--magnitude-columns',
                               help='Comma-separated list of magnitude columns to clean')
    process_parser.add_argument('--coordinate-columns',
                               help='Comma-separated list of coordinate columns to validate')
    process_parser.add_argument('--create-splits', action='store_true',
                               help='Create train/val/test splits')
    process_parser.add_argument('--test-size', type=float, default=0.2,
                               help='Test set size (default: 0.2)')
    process_parser.add_argument('--val-size', type=float, default=0.1,
                               help='Validation set size (default: 0.1)')
    process_parser.add_argument('--random-state', type=int, default=42,
                               help='Random seed for splits (default: 42)')
    process_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                               help='Disable shuffling for splits')
    process_parser.add_argument('--stats', action='store_true',
                               help='Show statistics before and after processing')
    
    # TNG50 processing command (optimized graph format)
    tng50_parser = subparsers.add_parser('tng50', help='Process TNG50 simulation data into graph format')
    tng50_parser.add_argument('input', nargs='?', help='TNG50 snapshot file (.hdf5) or base directory for --all-snapshots')
    tng50_parser.add_argument('-o', '--output', help='Output directory (default: data/processed/tng50_graphs)')
    tng50_parser.add_argument('--particle-types', default='PartType4,PartType5',
                             help='Comma-separated particle types (default: PartType4,PartType5)')
    tng50_parser.add_argument('--all', action='store_true',
                             help='Process all available particle types in the snapshot')
    tng50_parser.add_argument('--all-snapshots', action='store_true',
                             help='Process all snapshots in directory (input becomes base directory)')
    tng50_parser.add_argument('--max-particles', type=int, default=10000,
                             help='Maximum particles per type (default: 10000)')
    tng50_parser.add_argument('--radius', type=float, default=1.0,
                             help='Connection radius in Mpc (default: 1.0)')
    tng50_parser.add_argument('--force', action='store_true',
                             help='Force reprocessing even if graph files exist')
    tng50_parser.add_argument('--stats', action='store_true',
                             help='Show graph statistics')

    # TNG50 list command
    tng50_list_parser = subparsers.add_parser('tng50-list', help='List TNG50 snapshot files')
    tng50_list_parser.add_argument('--directory', help='TNG50 base directory (default: data/raw/TNG50-4/output)')
    tng50_list_parser.add_argument('--inspect', action='store_true',
                                  help='Inspect first snapshot file')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show catalog statistics')
    stats_parser.add_argument('input', help='Input catalog file (.parquet or .csv)')
    stats_parser.add_argument('-v', '--verbose', action='store_true',
                             help='Show detailed column information')
    
    # Load splits command
    splits_parser = subparsers.add_parser('load-splits', help='Load and inspect saved splits')
    splits_parser.add_argument('path', help='Base path containing split files')
    splits_parser.add_argument('dataset', help='Dataset name')
    splits_parser.add_argument('--stats', action='store_true',
                              help='Show statistics for loaded splits')
    
    # List catalogs command
    list_parser = subparsers.add_parser('list-catalogs', help='List available catalogs')
    
    # Show functions command
    functions_parser = subparsers.add_parser('functions', help='Show available functions')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_catalog_command(args)
    elif args.command == 'tng50':
        process_tng50_command(args)
    elif args.command == 'tng50-list':
        list_tng50_snapshots_command(args)
    elif args.command == 'stats':
        stats_command(args)
    elif args.command == 'load-splits':
        load_splits_command(args)
    elif args.command == 'list-catalogs':
        list_catalogs_command(args)
    elif args.command == 'functions':
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 