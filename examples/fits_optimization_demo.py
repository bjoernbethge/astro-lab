#!/usr/bin/env python3
"""
FITS Optimization Demo - Optimized FITS Loading with Memory Mapping

This demo showcases the new optimized FITS loading capabilities in astro-lab,
based on Astropy FITS documentation best practices:
- Memory mapping for large files
- Proper handling of scaled data
- Data sections for memory efficiency
- Comprehensive file information
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use the new data module
from astro_lab.data import (
    get_data_dir,
    get_fits_info,
    load_fits_optimized,
    load_fits_table_optimized,
)


def demo_fits_info():
    """Demonstrate FITS file information extraction."""
    print("=" * 60)
    print("🔍 FITS FILE INFORMATION DEMO")
    print("=" * 60)

    data_dir = get_data_dir()

    # Look for FITS files in data directory
    fits_files = list(data_dir.rglob("*.fits"))

    if not fits_files:
        print("❌ No FITS files found in data directory")
        print(f"   Data directory: {data_dir}")
        return

    print(f"Found {len(fits_files)} FITS files:")
    for fits_file in fits_files[:3]:  # Show first 3
        print(f"  📁 {fits_file.name}")

    # Analyze first FITS file
    test_file = fits_files[0]
    print(f"\n🔍 Analyzing: {test_file.name}")

    info = get_fits_info(test_file)

    if "error" in info:
        print(f"❌ Error: {info['error']}")
        return

    print(f"📊 File size: {info['file_size_mb']:.2f} MB")
    print(f"📋 Number of HDUs: {len(info['hdus'])}")

    for hdu_info in info["hdus"]:
        print(f"\n  HDU {hdu_info['index']}: {hdu_info['type']}")
        print(f"    Name: {hdu_info['name']}")

        if "shape" in hdu_info:
            print(f"    Shape: {hdu_info['shape']}")
            print(f"    Data type: {hdu_info['dtype']}")
            print(f"    Data size: {hdu_info['data_size_mb']:.2f} MB")

            if hdu_info.get("scaled", False):
                print(
                    f"    ⚠️  Scaled data: BSCALE={hdu_info['bscale']}, BZERO={hdu_info['bzero']}"
                )
            else:
                print("    ✅ Unscaled data")

        if "table_columns" in hdu_info:
            print(
                f"    Table: {hdu_info['table_rows']} rows, {hdu_info['table_columns']} columns"
            )
            if "column_names" in hdu_info:
                print(
                    f"    Columns: {', '.join(hdu_info['column_names'][:5])}{'...' if len(hdu_info['column_names']) > 5 else ''}"
                )


def demo_optimized_loading():
    """Demonstrate optimized FITS loading."""
    print("\n" + "=" * 60)
    print("🚀 OPTIMIZED FITS LOADING DEMO")
    print("=" * 60)

    data_dir = get_data_dir()
    fits_files = list(data_dir.rglob("*.fits"))

    if not fits_files:
        print("❌ No FITS files found for loading demo")
        return

    test_file = fits_files[0]
    print(f"🔍 Loading: {test_file.name}")

    # Demo 1: Standard loading with memory mapping
    print("\n1️⃣ Standard loading with memory mapping:")
    data = load_fits_optimized(test_file, memmap=True)

    if data is not None:
        if hasattr(data, "shape"):
            print(f"   ✅ Loaded data shape: {data.shape}")
            print(f"   📊 Data type: {data.dtype}")
            print(f"   💾 Memory usage: {data.nbytes / (1024 * 1024):.2f} MB")
        else:
            print(f"   ✅ Loaded header with {len(data)} cards")
    else:
        print("   ❌ Failed to load data")

    # Demo 2: Loading with scaling disabled
    print("\n2️⃣ Loading with scaling disabled:")
    data_unscaled = load_fits_optimized(test_file, do_not_scale=True)

    if data_unscaled is not None and hasattr(data_unscaled, "shape"):
        print(f"   ✅ Loaded unscaled data shape: {data_unscaled.shape}")
        print(f"   📊 Data type: {data_unscaled.dtype}")
    else:
        print("   ❌ No scalable data found")

    # Demo 3: Section loading (if data is large enough)
    if data is not None and hasattr(data, "shape") and len(data.shape) >= 2:
        print("\n3️⃣ Section loading (memory efficient):")
        try:
            # Load a 100x100 section from the center
            h, w = data.shape[:2]
            if h > 200 and w > 200:
                center_h, center_w = h // 2, w // 2
                section = (
                    slice(center_h - 50, center_h + 50),
                    slice(center_w - 50, center_w + 50),
                )

                section_data = load_fits_optimized(test_file, section=section)
                if section_data is not None:
                    print(f"   ✅ Loaded section shape: {section_data.shape}")
                    print(f"   💾 Section memory: {section_data.nbytes / 1024:.1f} KB")
                else:
                    print("   ❌ Failed to load section")
            else:
                print("   ℹ️  Image too small for section demo")
        except Exception as e:
            print(f"   ❌ Section loading error: {e}")


def demo_table_loading():
    """Demonstrate optimized table loading."""
    print("\n" + "=" * 60)
    print("📊 FITS TABLE LOADING DEMO")
    print("=" * 60)

    data_dir = get_data_dir()
    fits_files = list(data_dir.rglob("*.fits"))

    if not fits_files:
        print("❌ No FITS files found for table demo")
        return

    # Find a file with table data
    table_file = None
    for fits_file in fits_files:
        info = get_fits_info(fits_file)
        if "error" not in info:
            for hdu_info in info["hdus"]:
                if "table_columns" in hdu_info or hdu_info["type"] == "BinTableHDU":
                    table_file = fits_file
                    break
        if table_file:
            break

    if not table_file:
        print("❌ No FITS tables found")
        # Try to load table from HDU 1 anyway
        test_file = fits_files[0]
        print(f"🔍 Attempting table load from HDU 1: {test_file.name}")
        table_file = test_file

    print(f"🔍 Loading table from: {table_file.name}")

    # Demo 1: Full table loading
    print("\n1️⃣ Full table loading:")
    df = load_fits_table_optimized(table_file, as_polars=True)

    if df is not None:
        print(f"   ✅ Loaded table: {len(df)} rows, {len(df.columns)} columns")
        print(
            f"   📋 Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}"
        )

        # Show data types
        print("   📊 Data types:")
        for col in df.columns[:5]:
            print(f"      {col}: {df[col].dtype}")
    else:
        print("   ❌ Failed to load table")
        return

    # Demo 2: Limited rows
    print("\n2️⃣ Limited row loading:")
    df_limited = load_fits_table_optimized(table_file, max_rows=100, as_polars=True)

    if df_limited is not None:
        print(f"   ✅ Loaded limited table: {len(df_limited)} rows")
    else:
        print("   ❌ Failed to load limited table")

    # Demo 3: Specific columns
    if df is not None and len(df.columns) > 3:
        print("\n3️⃣ Specific column loading:")
        selected_cols = df.columns[:3]
        df_cols = load_fits_table_optimized(
            table_file, columns=selected_cols, as_polars=True
        )

        if df_cols is not None:
            print(f"   ✅ Loaded {len(df_cols.columns)} selected columns")
            print(f"   📋 Columns: {', '.join(df_cols.columns)}")
        else:
            print("   ❌ Failed to load selected columns")


def demo_memory_efficiency():
    """Demonstrate memory efficiency features."""
    print("\n" + "=" * 60)
    print("💾 MEMORY EFFICIENCY DEMO")
    print("=" * 60)

    data_dir = get_data_dir()
    fits_files = list(data_dir.rglob("*.fits"))

    if not fits_files:
        print("❌ No FITS files found for memory demo")
        return

    # Find largest file
    largest_file = max(fits_files, key=lambda f: f.stat().st_size)
    file_size_mb = largest_file.stat().st_size / (1024 * 1024)

    print(f"🔍 Testing with largest file: {largest_file.name} ({file_size_mb:.2f} MB)")

    # Demo 1: Memory mapping vs. regular loading
    print("\n1️⃣ Memory mapping comparison:")

    # Without memory mapping
    data_no_mmap = load_fits_optimized(largest_file, memmap=False)
    if data_no_mmap is not None and hasattr(data_no_mmap, "nbytes"):
        print(
            f"   📊 Without memmap: {data_no_mmap.nbytes / (1024 * 1024):.2f} MB loaded"
        )

    # With memory mapping
    data_mmap = load_fits_optimized(largest_file, memmap=True)
    if data_mmap is not None and hasattr(data_mmap, "nbytes"):
        print(f"   🚀 With memmap: {data_mmap.nbytes / (1024 * 1024):.2f} MB mapped")

    # Demo 2: Memory limit handling
    print("\n2️⃣ Memory limit handling:")
    data_limited = load_fits_optimized(largest_file, max_memory_mb=10.0)
    if data_limited is not None:
        print("   ✅ Loading completed with memory limit")
    else:
        print("   ❌ Loading failed or blocked by memory limit")


def demo_pt_vs_fits_comparison():
    """Compare .pt and FITS formats for astronomical data."""
    print("\n" + "=" * 60)
    print("⚖️  PT vs FITS FORMAT COMPARISON")
    print("=" * 60)

    data_dir = get_data_dir()

    # Look for .pt files
    pt_files = list(data_dir.rglob("*.pt"))
    fits_files = list(data_dir.rglob("*.fits"))

    print("📊 Format comparison:")
    print(f"   .pt files found: {len(pt_files)}")
    print(f"   .fits files found: {len(fits_files)}")

    if pt_files:
        pt_file = pt_files[0]
        pt_size = pt_file.stat().st_size / 1024  # KB
        print(f"\n🔍 Example .pt file: {pt_file.name}")
        print(f"   Size: {pt_size:.1f} KB")
        print("   Use case: ML-ready tensors, fast loading")

    if fits_files:
        fits_file = fits_files[0]
        fits_size = fits_file.stat().st_size / (1024 * 1024)  # MB
        print(f"\n🔍 Example FITS file: {fits_file.name}")
        print(f"   Size: {fits_size:.1f} MB")
        print("   Use case: Standard astronomy format, metadata-rich")

    print("\n📋 Format recommendations:")
    print("   🔬 FITS: Raw data, archives, metadata preservation")
    print("   ⚡ PT: Processed tensors, ML pipelines, fast iteration")
    print("   🔄 Workflow: FITS → Processing → PT → ML Models")


def main():
    """Run all FITS optimization demos."""
    print("🌟 ASTRO-LAB FITS OPTIMIZATION DEMO")
    print("Based on Astropy FITS documentation best practices\n")

    try:
        demo_fits_info()
        demo_optimized_loading()
        demo_table_loading()
        demo_memory_efficiency()
        demo_pt_vs_fits_comparison()

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\n🚀 Key improvements implemented:")
        print("   • Memory mapping for large files")
        print("   • Proper scaled data handling")
        print("   • Data section loading")
        print("   • Comprehensive file analysis")
        print("   • Optimized table loading")
        print("\n📖 For more info, see:")
        print("   https://docs.astropy.org/en/stable/io/fits/usage/image.html")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure astropy is installed: uv add astropy")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
