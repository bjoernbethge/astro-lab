#!/usr/bin/env python3
"""
NSA Processing Example

Demonstrates NSA (NASA-Sloan Atlas) data processing and visualization.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from astro_lab.data.core import load_nsa_data
from astro_lab.widgets.astro_lab import AstroLabWidget


def demonstrate_nsa_analysis():
    """Demonstrate NSA data analysis."""
    return load_nsa_data(max_samples=1000, return_tensor=True)


def create_nsa_visualizations(nsa_data):
    """Create visualizations of NSA galaxy data."""
    widget = AstroLabWidget()
    
    # Feature analysis (no spatial coordinates needed)
    widget.plot(nsa_data, 'radar',
               title='NSA Galaxy Features')
    
    # Distribution analysis (no spatial coordinates needed)
    widget.plot(nsa_data, 'histogram',
               title='NSA Redshift Distribution')
    
    # Correlation analysis (no spatial coordinates needed)
    widget.plot(nsa_data, 'scatter',
               title='NSA Galaxy Properties')


def main():
    """Main function to run NSA processing examples."""
    # Load and analyze data
    nsa_data = demonstrate_nsa_analysis()
    
    # Create visualizations
    create_nsa_visualizations(nsa_data)


if __name__ == "__main__":
    main() 