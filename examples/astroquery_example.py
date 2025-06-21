#!/usr/bin/env python3
"""
AstroQuery Integration Example
==============================

Demonstrates integration with AstroQuery for external astronomical data sources.
Shows how to load data from various surveys using AstroLab's data loading functions.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import load_gaia_data, load_sdss_data, create_cosmic_web_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_gaia_survey_data():
    """Load Gaia DR3 survey data."""
    logger.info("🌟 Loading Gaia DR3 survey data")
    logger.info("=" * 40)
    
    # Load Gaia data
    gaia_tensor = load_gaia_data(
        max_samples=1000,
        return_tensor=True
    )
    
    logger.info(f"✅ Loaded {len(gaia_tensor)} Gaia stars")
    logger.info(f"📊 Shape: {gaia_tensor.shape}")
    logger.info(f"🎯 Survey: {gaia_tensor.survey_name}")
    
    return gaia_tensor


def load_sdss_survey_data():
    """Load SDSS survey data."""
    logger.info("🌌 Loading SDSS survey data")
    logger.info("=" * 35)
    
    # Load SDSS data
    sdss_tensor = load_sdss_data(
        max_samples=500,
        return_tensor=True
    )
    
    logger.info(f"✅ Loaded {len(sdss_tensor)} SDSS galaxies")
    logger.info(f"📊 Shape: {sdss_tensor.shape}")
    logger.info(f"🎯 Survey: {sdss_tensor.survey_name}")
    
    return sdss_tensor


def perform_cosmic_web_analysis():
    """Perform cosmic web analysis on survey data."""
    logger.info("🌐 Performing cosmic web analysis")
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


def main():
    """Main AstroQuery integration workflow."""
    logger.info("🌌 AstroQuery Integration Example")
    logger.info("=" * 45)
    
    try:
        # Load survey data
        gaia_data = load_gaia_survey_data()
        sdss_data = load_sdss_survey_data()
        
        # Perform analysis
        cosmic_results = perform_cosmic_web_analysis()
        
        logger.info("🎉 All survey data loaded successfully!")
        logger.info(f"   - Gaia: {len(gaia_data)} stars")
        logger.info(f"   - SDSS: {len(sdss_data)} galaxies")
        logger.info(f"   - Cosmic Web: {cosmic_results['n_objects']} objects analyzed")
        
    except Exception as e:
        logger.error(f"❌ Survey data loading failed: {e}")
        raise


if __name__ == "__main__":
    main()
