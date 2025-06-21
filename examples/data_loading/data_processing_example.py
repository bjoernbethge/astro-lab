#!/usr/bin/env python3
"""
Data Processing Example
=======================

Comprehensive example demonstrating data download, preprocessing, and loading
using AstroLab with Polars for efficient data handling.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import directly from src to avoid Cosmograph dependencies
sys.path.append(str(project_root / "src"))
from astro_lab.data.core import load_gaia_data, load_sdss_data, create_cosmic_web_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_preprocess_gaia():
    """Download and preprocess Gaia DR3 data."""
    logger.info("🌟 Starting Gaia DR3 data processing")
    logger.info("=" * 50)
    
    # Load Gaia data (download happens automatically if not cached)
    gaia_tensor = load_gaia_data(
        max_samples=1000,
        return_tensor=True
    )
    
    logger.info(f"✅ Loaded {len(gaia_tensor)} Gaia stars")
    logger.info(f"📊 Shape: {gaia_tensor.shape}")
    logger.info(f"🎯 Survey: {gaia_tensor.survey_name}")
    
    # Access underlying Polars DataFrame
    if hasattr(gaia_tensor, 'data'):
        df = gaia_tensor.data
        logger.info(f"📋 Columns: {list(df.columns)}")
        logger.info(f"📈 Data types: {df.schema}")
    
    return gaia_tensor


def download_and_preprocess_sdss():
    """Download and preprocess SDSS data."""
    logger.info("🌌 Starting SDSS data processing")
    logger.info("=" * 45)
    
    # Load SDSS data (download happens automatically if not cached)
    sdss_tensor = load_sdss_data(
        max_samples=500,
        return_tensor=True
    )
    
    logger.info(f"✅ Loaded {len(sdss_tensor)} SDSS galaxies")
    logger.info(f"📊 Shape: {sdss_tensor.shape}")
    logger.info(f"🎯 Survey: {sdss_tensor.survey_name}")
    
    # Access underlying Polars DataFrame
    if hasattr(sdss_tensor, 'data'):
        df = sdss_tensor.data
        logger.info(f"📋 Columns: {list(df.columns)}")
        logger.info(f"📈 Data types: {df.schema}")
    
    return sdss_tensor


def perform_cosmic_web_analysis():
    """Perform cosmic web analysis on processed data."""
    logger.info("🌐 Starting cosmic web analysis")
    logger.info("=" * 40)
    
    # Perform cosmic web analysis
    results = create_cosmic_web_loader(
        survey="gaia",
        max_samples=500,
        scales_mpc=[5.0, 10.0, 20.0]
    )
    
    logger.info(f"✅ Analyzed {results['n_objects']} objects")
    logger.info(f"📊 Volume: {results['total_volume']:.0f} Mpc³")
    logger.info(f"🌍 Global density: {results['global_density']:.2e} obj/Mpc³")
    
    # Show multi-scale results
    for scale, result in results["results_by_scale"].items():
        logger.info(f"  {scale} Mpc: {result['n_clusters']} groups, "
                   f"{result['grouped_fraction']*100:.1f}% grouped")
    
    return results


def demonstrate_data_operations():
    """Demonstrate various data operations with Polars."""
    logger.info("🔧 Demonstrating data operations")
    logger.info("=" * 40)
    
    # Load data
    gaia_tensor = load_gaia_data(max_samples=100, return_tensor=True)
    
    if hasattr(gaia_tensor, 'data'):
        df = gaia_tensor.data
        
        # Demonstrate Polars operations
        logger.info("📊 Basic statistics:")
        logger.info(f"  Mean distance: {df['distance'].mean():.2f} pc")
        logger.info(f"  Distance std: {df['distance'].std():.2f} pc")
        logger.info(f"  Min magnitude: {df['phot_g_mean_mag'].min():.2f}")
        logger.info(f"  Max magnitude: {df['phot_g_mean_mag'].max():.2f}")
        
        # Filtering example
        bright_stars = df.filter(df['phot_g_mean_mag'] < 10.0)
        logger.info(f"  Bright stars (G < 10): {len(bright_stars)}")
        
        # Grouping example
        if 'ra' in df.columns:
            ra_bins = df.with_columns(
                (df['ra'] // 30).alias('ra_bin')
            ).group_by('ra_bin').count()
            logger.info(f"  RA bins: {len(ra_bins)} bins created")


def main():
    """Main data processing workflow."""
    logger.info("🌌 AstroLab Data Processing Example")
    logger.info("=" * 50)
    
    try:
        # Download and preprocess data
        gaia_data = download_and_preprocess_gaia()
        sdss_data = download_and_preprocess_sdss()
        
        # Perform analysis
        cosmic_results = perform_cosmic_web_analysis()
        
        # Demonstrate operations
        demonstrate_data_operations()
        
        logger.info("🎉 All data processing completed successfully!")
        logger.info(f"   - Gaia: {len(gaia_data)} stars")
        logger.info(f"   - SDSS: {len(sdss_data)} galaxies")
        logger.info(f"   - Cosmic Web: {cosmic_results['n_objects']} objects analyzed")
        
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
