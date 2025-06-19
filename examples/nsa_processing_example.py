#!/usr/bin/env python3
"""
NASA Sloan Atlas (NSA) Data Processing Example

This example demonstrates how to:
1. Load NSA data using the new data manager
2. Create graphs from galaxy catalogs
3. Use modern DataLoaders for training
4. Process astronomical data efficiently
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# Use the new data module
from astro_lab.data import (
    HAS_ENHANCED_FEATURES,
    NSAGraphDataset,
    create_nsa_dataloader,
    data_manager,
    get_data_dir,
    get_data_statistics,
)


def main():
    """Main example function demonstrating modern NSA data handling."""
    print("NASA Sloan Atlas (NSA) Modern Data Processing")
    print("=" * 50)

    # Check which features are available
    if HAS_ENHANCED_FEATURES:
        print("✅ Enhanced features available - full processing pipeline")
        demo_enhanced_processing()
    else:
        print("📊 Using PyTorch Geometric datasets only")
        demo_basic_graph_dataset()

    print("\n" + "=" * 50)
    print("✅ Demo completed successfully")


def demo_enhanced_processing():
    """Demo with enhanced processing features using data manager."""
    print("\n1. Loading NSA data with data manager...")

    try:
        # Try to load existing NSA catalog
        catalog_path = Path("data/processed/nsa/catalog.parquet")

        if catalog_path.exists():
            df = data_manager.load_catalog(catalog_path)
            print(f"✅ Loaded existing NSA catalog: {len(df):,} galaxies")

            # Get statistics
            stats = get_data_statistics(df)
            print(f"📊 Total columns: {stats['n_columns']}")
            print(f"📊 Numeric columns: {len(stats['numeric_columns'])}")

            # Show sample of data
            print(f"✅ Data sample shape: {df.shape}")
            if len(df) > 0:
                print(f"📋 Sample columns: {df.columns[:5]}")
        else:
            print("❌ No NSA catalog found")
            print("💡 To create NSA data, you need to:")
            print("   1. Install astroML: uv add astroML")
            print("   2. Run NSA download script")

        # Now demo the graph dataset
        demo_basic_graph_dataset()

    except Exception as e:
        print(f"❌ Error in enhanced processing: {e}")
        demo_basic_graph_dataset()


def demo_basic_graph_dataset():
    """Demo using PyTorch Geometric NSA dataset."""
    print("\n2. Creating NSA Graph Dataset...")

    try:
        # Create graph dataset
        dataset = NSAGraphDataset(
            max_galaxies=1000,  # Smaller for demo
            k_neighbors=8,
            distance_threshold=50.0,
        )

        print(f"✅ Dataset created with {len(dataset)} graphs")

        if len(dataset) > 0:
            # Show first graph
            data = dataset[0]  # type: ignore
            print("📊 Graph structure:")
            print(f"   - Nodes: {data.num_nodes:,}")  # type: ignore
            print(f"   - Edges: {data.num_edges:,}")  # type: ignore
            print(f"   - Features: {data.x.shape[1]}")  # type: ignore
            print(f"   - 3D positions: {data.pos.shape}")  # type: ignore

            # Create DataLoader
            dataloader = create_nsa_dataloader(
                max_galaxies=1000,
                batch_size=1,
                shuffle=False,
                use_galaxy_transforms=True,
            )

            print("✅ DataLoader created successfully")

            # Test batch loading
            batch = next(iter(dataloader))
            print(f"📦 Batch shape: {batch.x.shape}")  # type: ignore
            print(f"📦 Batch edges: {batch.edge_index.shape}")  # type: ignore

        else:
            print("❌ Dataset is empty - no NSA data available")
            print(
                "💡 Make sure NSA data is available in data/processed/nsa/catalog.parquet"
            )

    except FileNotFoundError as e:
        print(f"❌ NSA data not found: {e}")
        print("💡 To create NSA data:")
        print("   1. Install astroML: uv add astroML")
        print("   2. Use data manager to download/import NSA data")
        demo_mock_data()
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        demo_mock_data()


def demo_mock_data():
    """Demo with mock data if real data not available."""
    print("\n3. Mock Data Demo...")

    # Create mock NSA-like data
    n_galaxies = 100
    mock_data = {
        "RA": np.random.uniform(0, 360, n_galaxies),
        "DEC": np.random.uniform(-30, 30, n_galaxies),
        "ZDIST": np.random.uniform(0.01, 0.1, n_galaxies),
        "PETROMAG_R": np.random.uniform(12, 18, n_galaxies),
        "MASS": np.random.uniform(1e9, 1e12, n_galaxies),
    }

    print(f"✅ Created mock data with {n_galaxies} galaxies")

    # Show data ranges
    for key, values in mock_data.items():
        print(f"   {key}: {values.min():.3f} - {values.max():.3f}")


if __name__ == "__main__":
    main()
