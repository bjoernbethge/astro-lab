#!/usr/bin/env python3
"""
AstroLab Quick Start Example

Simple example: load Gaia data and visualize it.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import load_gaia_data
from astro_lab.widgets.astro_lab import AstroLabWidget


def main():
    """
    Quick start: load processed data and create one visualization.
    """
    # Load processed Gaia data
    gaia_data = load_gaia_data(max_samples=10000, return_tensor=True)

    # Create one visualization to demonstrate functionality
    widget = AstroLabWidget()
    widget.plot(gaia_data, 'scatter', title='Gaia Stars')


if __name__ == "__main__":
    main()
