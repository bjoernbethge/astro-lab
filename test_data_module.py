#!/usr/bin/env python3
"""
Test script for astro-lab data module.
"""

import sys

sys.path.insert(0, "src")

from astro_lab.data import (
    get_data_dir,
    get_data_statistics,
    preprocess_catalog,
    create_training_splits,
)

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


def main():
    print("🌌 astro-lab Data Module Test")
    print("=" * 40)

    # Show initial data directory
    data_dir = get_data_dir()
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Directory exists: {data_dir.exists()}")

    if not POLARS_AVAILABLE:
        print("❌ Polars not available - skipping data tests")
        return

    # Test with sample data if available
    sample_files = [
        data_dir / "datasets" / "nsa" / "catalog_sample_50.parquet",
        data_dir / "raw" / "nsa" / "nsa_raw.parquet",
    ]
    
    test_file = None
    for file_path in sample_files:
        if file_path.exists():
            test_file = file_path
            break
    
    if test_file:
        print(f"\n📊 Testing with: {test_file.name}")
        
        # Load and analyze data
        df = pl.read_parquet(test_file)
        print(f"📈 Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Get statistics
        stats = get_data_statistics(df)
        print(f"📊 Statistics:")
        print(f"  • Memory usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"  • Numeric columns: {len(stats['numeric_columns'])}")
        
        # Test preprocessing
        print(f"\n🧹 Testing preprocessing...")
        df_clean = preprocess_catalog(df)
        print(f"✅ Preprocessing successful: {df_clean.shape[0]:,} rows")
        
        # Test splits
        if len(df_clean) >= 10:  # Only if we have enough data
            print(f"\n🔄 Testing training splits...")
            train, val, test = create_training_splits(df_clean)
            total = len(train) + len(val) + len(test)
            print(f"✅ Splits created: {total:,} total rows")
        
    else:
        print("📂 No test data files found - skipping data tests")

    print("\n✅ Data module test completed!")


if __name__ == "__main__":
    main() 