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
    AstroDataManager,
    data_manager,
    list_catalogs,
    load_catalog,
    # TNG50 support
    import_tng50,
)
# TNG50Loader removed - using data module functions instead


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
    """Process TNG50 simulation data using data module functions."""
    snap_file = Path(args.input)
    if not snap_file.exists():
        print(f"❌ TNG50 snapshot file not found: {snap_file}")
        sys.exit(1)
    
    print(f"🌌 Processing TNG50 snapshot: {snap_file.name}")
    
    try:
        # Use data module's import_tng50 function
        particle_types = args.particle_types.split(',') if args.particle_types else ["PartType5"]
        
        # Process each particle type separately using data module
        processed_files = []
        for ptype in particle_types:
            print(f"📊 Processing {ptype}...")
            
            # Import using data module function
            parquet_file = import_tng50(snap_file, dataset_name=ptype)
            processed_files.append(parquet_file)
            
            # Load the processed data
            df = load_catalog(parquet_file)
            
            # Limit particles if requested
            if len(df) > args.max_particles:
                df = df.sample(args.max_particles, seed=args.random_state)
                print(f"  • Limited to {args.max_particles:,} particles")
            
            print(f"  • {ptype}: {len(df):,} particles, {len(df.columns)} columns")
            
            # Show statistics if requested
            if args.stats:
                stats = get_data_statistics(df)
                print(f"    Memory: {stats['memory_usage_mb']:.1f} MB")
                print(f"    Numeric columns: {len(stats['numeric_columns'])}")
            
            # Save processed data or create splits
            if args.create_splits:
                print(f"  🔄 Creating training splits for {ptype}...")
                train, val, test = create_training_splits(
                    df,
                    test_size=args.test_size,
                    val_size=args.val_size,
                    random_state=args.random_state,
                    shuffle=args.shuffle
                )
                
                if args.output:
                    output_path = Path(args.output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    dataset_name = f"tng50_{snap_file.stem}_{ptype.lower()}"
                    save_splits_to_parquet(train, val, test, output_path, dataset_name)
                    print(f"    ✅ Splits saved to: {output_path}")
            
            elif args.output:
                output_path = Path(args.output)
                if output_path.is_dir():
                    output_file = output_path / f"tng50_{snap_file.stem}_{ptype.lower()}.parquet"
                else:
                    output_file = output_path.parent / f"{output_path.stem}_{ptype.lower()}.parquet"
                
                df.write_parquet(output_file)
                print(f"    💾 Data saved to: {output_file}")
        
        print(f"\n✅ TNG50 processing complete! Processed {len(particle_types)} particle types.")
        
    except Exception as e:
        print(f"❌ Error processing TNG50 data: {e}")
        sys.exit(1)


def process_tng50_graphs_command(args):
    """Process TNG50 data into graph format (.pt files)."""
    try:
        from astro_lab.data.datasets.astronomical import TNG50GraphDataset
        
        snap_file = Path(args.input)
        if not snap_file.exists():
            print(f"❌ Snapshot file not found: {snap_file}")
            sys.exit(1)
        
        print(f"🌌 Processing TNG50 snapshot into graphs: {snap_file.name}")
        
        # Parse particle types
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
        
        print(f"\n✅ TNG50 graph processing complete!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Please install: uv add torch torch-geometric")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing TNG50 graphs: {e}")
        sys.exit(1)


def list_tng50_snapshots_command(args):
    """List available TNG50 snapshot files."""
    tng50_dir = Path(args.directory) if args.directory else Path("data/raw/TNG50-4/output/snapdir_099")
    
    if not tng50_dir.exists():
        print(f"❌ TNG50 directory not found: {tng50_dir}")
        sys.exit(1)
    
    print(f"📂 TNG50 snapshots in: {tng50_dir}")
    
    snap_files = list(tng50_dir.glob("snap_*.hdf5"))
    if not snap_files:
        print("❌ No TNG50 snapshot files found")
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

# Show catalog statistics
astro-lab preprocess stats catalog.parquet --verbose

# Process TNG50 data
astro-lab preprocess tng50 snap_099.0.hdf5 --particle-types PartType4,PartType5 --max-particles 1000

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
    
    # TNG50 process command
    tng50_parser = subparsers.add_parser('tng50', help='Process TNG50 simulation data')
    tng50_parser.add_argument('input', help='TNG50 snapshot file (.hdf5)')
    tng50_parser.add_argument('-o', '--output', help='Output directory or file')
    tng50_parser.add_argument('--particle-types', default='PartType5',
                             help='Comma-separated particle types (default: PartType5)')
    tng50_parser.add_argument('--max-particles', type=int, default=10000,
                             help='Maximum particles per type (default: 10000)')
    tng50_parser.add_argument('--create-splits', action='store_true',
                             help='Create train/val/test splits')
    tng50_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Test set size (default: 0.2)')
    tng50_parser.add_argument('--val-size', type=float, default=0.1,
                             help='Validation set size (default: 0.1)')
    tng50_parser.add_argument('--random-state', type=int, default=42,
                             help='Random seed for splits (default: 42)')
    tng50_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                             help='Disable shuffling for splits')
    tng50_parser.add_argument('--stats', action='store_true',
                             help='Show statistics for processed data')
    
    # TNG50 graph processing command (pt format)
    tng50_graph_parser = subparsers.add_parser('tng50-graphs', help='Process TNG50 into graph format (.pt files)')
    tng50_graph_parser.add_argument('input', help='TNG50 snapshot file (.hdf5)')
    tng50_graph_parser.add_argument('-o', '--output', help='Output directory (default: data/processed/tng50_graphs)')
    tng50_graph_parser.add_argument('--particle-types', default='PartType4,PartType5',
                                   help='Comma-separated particle types (default: PartType4,PartType5)')
    tng50_graph_parser.add_argument('--max-particles', type=int, default=10000,
                                   help='Maximum particles per type (default: 10000)')
    tng50_graph_parser.add_argument('--radius', type=float, default=1.0,
                                   help='Connection radius in Mpc (default: 1.0)')
    tng50_graph_parser.add_argument('--stats', action='store_true',
                                   help='Show graph statistics')

    # TNG50 list command
    tng50_list_parser = subparsers.add_parser('tng50-list', help='List TNG50 snapshot files')
    tng50_list_parser.add_argument('--directory', help='TNG50 directory (default: data/raw/TNG50-4/output/snapdir_099)')
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
    elif args.command == 'tng50-graphs':
        process_tng50_graphs_command(args)
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
        print("  • import_tng50(hdf5_file, dataset_name)")
        print("  • TNG50Loader.load_snapshot()")
        print("  • TNG50GraphDataset")
        print("  • create_tng50_dataloader()")
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