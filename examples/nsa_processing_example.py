#!/usr/bin/env python3
"""
NSA Processing Example
======================

Demonstrates processing of NASA Sloan Atlas (NSA) galaxy data
using AstroLab with Polars for efficient data handling.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import load_nsa_data, create_cosmic_web_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_nsa_galaxy_data():
    """Load NSA galaxy data."""
    logger.info("🌌 Loading NSA galaxy data")
    logger.info("=" * 35)
    
    # Load NSA data
    nsa_tensor = load_nsa_data(
        max_samples=1000,
        return_tensor=True
    )
    
    logger.info(f"✅ Loaded {len(nsa_tensor)} NSA galaxies")
    logger.info(f"📊 Shape: {nsa_tensor.shape}")
    logger.info(f"🎯 Survey: {nsa_tensor.survey_name}")
    
    # Access underlying Polars DataFrame
    if hasattr(nsa_tensor, 'data'):
        df = nsa_tensor.data
        logger.info(f"📋 Columns: {list(df.columns)}")
        logger.info(f"📈 Data types: {df.schema}")
    
    return nsa_tensor


def perform_nsa_cosmic_web_analysis():
    """Perform cosmic web analysis on NSA data."""
    logger.info("🌐 Performing NSA cosmic web analysis")
    logger.info("=" * 45)
    
    # Perform cosmic web analysis
    results = create_cosmic_web_loader(
        survey="nsa",
        max_samples=500,
        scales_mpc=[10.0, 20.0, 50.0]
    )
    
    logger.info(f"✅ Analyzed {results['n_objects']} galaxies")
    logger.info(f"📊 Volume: {results['total_volume']:.0f} Mpc³")
    logger.info(f"🌍 Global density: {results['global_density']:.2e} obj/Mpc³")
    
    # Show multi-scale results
    for scale, result in results["results_by_scale"].items():
        logger.info(f"  {scale} Mpc: {result['n_clusters']} groups, "
                   f"{result['grouped_fraction']*100:.1f}% grouped")
    
    return results


def demonstrate_nsa_data_operations():
    """Demonstrate NSA-specific data operations."""
    logger.info("🔧 Demonstrating NSA data operations")
    logger.info("=" * 40)
    
    # Load data
    nsa_tensor = load_nsa_data(max_samples=100, return_tensor=True)
    
    if hasattr(nsa_tensor, 'data'):
        df = nsa_tensor.data
        
        # Demonstrate NSA-specific operations
        logger.info("📊 NSA galaxy statistics:")
        
        if 'z' in df.columns:
            logger.info(f"  Mean redshift: {df['z'].mean():.3f}")
            logger.info(f"  Redshift range: {df['z'].min():.3f} - {df['z'].max():.3f}")
        
        if 'absmag' in df.columns:
            logger.info(f"  Mean absolute magnitude: {df['absmag'].mean():.2f}")
        
        if 'mass' in df.columns:
            logger.info(f"  Mean stellar mass: {df['mass'].mean():.2e} M☉")


def main():
    """Main NSA processing workflow."""
    logger.info("🌌 AstroLab NSA Processing Example")
    logger.info("=" * 45)
    
    try:
        # Load NSA data
        nsa_data = load_nsa_galaxy_data()
        
        # Perform analysis
        cosmic_results = perform_nsa_cosmic_web_analysis()
        
        # Demonstrate operations
        demonstrate_nsa_data_operations()
        
        logger.info("🎉 NSA processing completed successfully!")
        logger.info(f"   - NSA: {len(nsa_data)} galaxies")
        logger.info(f"   - Cosmic Web: {cosmic_results['n_objects']} objects analyzed")
        
    except Exception as e:
        logger.error(f"❌ NSA processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
