#!/usr/bin/env python3
"""
🌌 Simple Cosmograph Demo

Demonstrates basic cosmic web analysis and visualization
using AstroLab's cosmic web analysis and CosmographBridge.
"""

from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz.cosmograph_bridge import (
    CosmographBridge, create_cosmograph_visualization
)

def demo_gaia_cosmic_web():
    """Demo with Gaia stellar data."""
    print("🌟 Gaia Cosmic Web Analysis")
    print("=" * 30)
    
    # Load Gaia data with cosmic web analysis
    results = create_cosmic_web_loader(
        survey="gaia",
        max_samples=1000,
        scales_mpc=[5.0, 10.0, 20.0]
    )
    
    print(f"Found {results['n_objects']} objects")
    print(f"Volume: {results['total_volume']:.0f} Mpc³")
    
    # Create interactive visualization
    bridge = CosmographBridge()
    widget = bridge.from_cosmic_web_results(
        results,
        survey_name="gaia",
        radius=3.0,
        background_color='#000011'
    )
    
    return widget

def demo_sdss_cosmic_web():
    """Demo with SDSS galaxy data."""
    print("🌌 SDSS Cosmic Web Analysis")
    print("=" * 30)
    
    # Load SDSS data
    results = create_cosmic_web_loader(
        survey="sdss",
        max_samples=500,
        scales_mpc=[10.0, 20.0, 50.0]
    )
    
    print(f"Found {results['n_objects']} galaxies")
    print(f"Volume: {results['total_volume']:.0f} Mpc³")
    
    # Create visualization
    bridge = CosmographBridge()
    widget = bridge.from_cosmic_web_results(
        results,
        survey_name="sdss",
        radius=5.0,
        background_color='#000011'
    )
    
    return widget

def demo_tng50_cosmic_web():
    """Demo with TNG50 simulation data."""
    print("🌌 TNG50 Cosmic Web Analysis")
    print("=" * 30)
    
    # Load TNG50 data
    results = create_cosmic_web_loader(
        survey="tng50",
        max_samples=300,
        scales_mpc=[5.0, 10.0, 20.0]
    )
    
    print(f"Found {results['n_objects']} particles")
    print(f"Volume: {results['total_volume']:.0f} Mpc³")
    
    # Create visualization
    bridge = CosmographBridge()
    widget = bridge.from_cosmic_web_results(
        results,
        survey_name="tng50",
        radius=4.0,
        background_color='#000011'
    )
    
    return widget

if __name__ == "__main__":
    print("🌌 AstroLab Cosmograph Demo")
    print("=" * 40)
    
    # Run demos
    try:
        gaia_widget = demo_gaia_cosmic_web()
        print("✅ Gaia demo completed")
    except Exception as e:
        print(f"❌ Gaia demo failed: {e}")
    
    try:
        sdss_widget = demo_sdss_cosmic_web()
        print("✅ SDSS demo completed")
    except Exception as e:
        print(f"❌ SDSS demo failed: {e}")
    
    try:
        tng50_widget = demo_tng50_cosmic_web()
        print("✅ TNG50 demo completed")
    except Exception as e:
        print(f"❌ TNG50 demo failed: {e}")
    
    print("\n🎉 All demos completed!")
    print("Check the widgets for interactive visualization.") 