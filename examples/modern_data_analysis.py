#!/usr/bin/env python3
"""
Modern Astronomical Data Analysis Example

Demonstrates analysis and visualization with processed Gaia and NSA data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from astro_lab.data.core import load_gaia_data, load_nsa_data
from astro_lab.widgets.astro_lab import AstroLabWidget

def main():
    gaia_data = load_gaia_data(max_samples=25000, return_tensor=True)
    nsa_data = load_nsa_data(max_samples=25000, return_tensor=True)
    widget = AstroLabWidget()
    widget.plot(gaia_data, 'blender', title='Gaia 3D')
    widget.plot(gaia_data, 'histogram', title='Gaia Magnitude')
    widget.plot(nsa_data, 'radar', title='NSA Features')
    widget.plot(nsa_data, 'histogram', title='NSA Redshift')

if __name__ == "__main__":
    main()
