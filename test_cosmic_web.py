#!/usr/bin/env python3
"""Test cosmic web functionality."""

import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Test direct import first
    from astro_lab.data.loaders import list_available_catalogs, load_survey_catalog
    from astro_lab.data.cosmic_web import CosmicWebAnalyzer
    from astro_lab.tensors.tensordict_astro import SpatialTensorDict
    
    logger.info("Imports successful!")
    
    # Create analyzer
    analyzer = CosmicWebAnalyzer()
    
    # Check available catalogs
    logger.info("Checking available catalogs...")
    catalogs = list_available_catalogs(survey="nsa")
    
    if len(catalogs) > 0:
        logger.info(f"Found {len(catalogs)} NSA catalogs")
        print(catalogs)
        
        # Try to analyze with small dataset
        logger.info("Running cosmic web analysis...")
        results = analyzer.analyze_nsa_cosmic_web(
            clustering_scales=[5.0, 10.0],
            min_samples=3,
            redshift_limit=0.1,
        )
        logger.info(f"Analysis complete!")
        logger.info(f"  Galaxies analyzed: {results['n_galaxies']}")
        for scale, stats in results["clustering_results"].items():
            logger.info(f"  {scale}: {stats['n_clusters']} clusters")
    else:
        logger.warning("No NSA catalogs found")
        
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure to restart the Python kernel or use 'python -m astro_lab.cli cosmic-web nsa'")
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()

