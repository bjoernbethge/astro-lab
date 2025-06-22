#!/usr/bin/env python3
"""
AstroQuery Integration Example

Demonstrates the use of AstroLab's integrated AstroQuery functions
and shows how to work with downloaded data.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.manager import (
    download_gaia,
    download_bright_all_sky,
    load_gaia_bright_stars,
    list_catalogs,
    load_catalog
)
from astro_lab.widgets.astro_lab import AstroLabWidget


def demonstrate_gaia_download():
    """Demonstrate downloading Gaia data using integrated AstroQuery functions."""
    # Download bright all-sky catalog
    bright_catalog_path = download_bright_all_sky(magnitude_limit=12.0)
    
    # Download specific region
    lmc_catalog_path = download_gaia(region="lmc", magnitude_limit=15.0)
    
    return bright_catalog_path, lmc_catalog_path


def demonstrate_data_analysis():
    """Demonstrate analyzing downloaded Gaia data."""
    # Load bright stars
    bright_stars = load_gaia_bright_stars(magnitude_limit=12.0)
    
    # Create widget for visualization
    widget = AstroLabWidget()
    
    # Create visualizations
    widget.plot(bright_stars, 'scatter',
               title='Bright Gaia Stars')
    
    widget.plot(bright_stars, 'histogram',
               title='Bright Star Magnitude Distribution')
    
    return bright_stars


def demonstrate_catalog_management():
    """Demonstrate catalog management functions."""
    # List available catalogs
    catalogs = list_catalogs()
    return catalogs


def main():
    """Main function: demonstrate integrated AstroQuery functionality."""
    # Download data
    bright_path, lmc_path = demonstrate_gaia_download()
    
    # Analyze data
    bright_stars = demonstrate_data_analysis()
    
    # Manage catalogs
    catalogs = demonstrate_catalog_management()


if __name__ == "__main__":
    main()
