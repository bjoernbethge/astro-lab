#!/usr/bin/env python3
"""
Simple Cosmograph Demo - Using CosmographBridge with AstroLab data.

Shows how to create interactive graph visualizations from AstroLab tensors
and survey data using the CosmographBridge class.
"""

from src.astro_lab.data.core import create_cosmic_web_loader
from src.astro_lab.utils.viz.cosmograph_bridge import (
    CosmographBridge,
    create_cosmograph_visualization,
)


def demo_gaia_stars():
    """Demo with Gaia stellar data."""
    print("Loading Gaia data...")
    
    # Load Gaia data using cosmic web loader
    results = create_cosmic_web_loader(survey="gaia", max_samples=500)
    
    # Create visualization using the new cosmic web method
    bridge = CosmographBridge()
    widget = bridge.from_cosmic_web_results(
        results, 
        survey_name="gaia",
        radius=3.0,
        point_color='#ffd700'  # Gold for stars
    )
    
    print(f"✅ Gaia visualization created: {len(results['coordinates'])} stars")
    return widget


def demo_sdss_galaxies():
    """Demo with SDSS galaxy data."""
    print("Loading SDSS data...")
    
    # Load SDSS data
    results = create_cosmic_web_loader(survey="sdss", max_samples=300)
    
    # Use convenience function with cosmic web results
    widget = create_cosmograph_visualization(
        results,
        survey_name="sdss",
        radius=8.0,
        background_color='#001122'
    )
    
    print(f"✅ SDSS visualization created: {len(results['coordinates'])} galaxies")
    return widget


def demo_tng50_simulation():
    """Demo with TNG50 simulation data."""
    print("Loading TNG50 data...")
    
    # Load TNG50 data
    results = create_cosmic_web_loader(survey="tng50", max_samples=400)
    
    # Create visualization
    bridge = CosmographBridge()
    widget = bridge.from_cosmic_web_results(
        results,
        survey_name="tng50", 
        radius=12.0,
        point_color='#00ff00',  # Green for simulation
        simulation_gravity=0.05,
        simulation_repulsion=0.4
    )
    
    print(f"✅ TNG50 visualization created: {len(results['coordinates'])} particles")
    return widget


def demo_nsa_galaxies():
    """Demo with NSA galaxy data."""
    print("Loading NSA data...")
    
    # Load NSA data
    results = create_cosmic_web_loader(survey="nsa", max_samples=200)
    
    # Create visualization
    bridge = CosmographBridge()
    widget = bridge.from_cosmic_web_results(
        results,
        survey_name="nsa",
        radius=10.0,
        point_color='#e24a4a',  # Red for NSA
        simulation_gravity=0.02,
        simulation_repulsion=0.3
    )
    
    print(f"✅ NSA visualization created: {len(results['coordinates'])} galaxies")
    return widget


def main():
    """Run all demos."""
    print("🌌 Simple Cosmograph Demo with Real Data")
    print("=" * 40)
    
    widgets = []
    
    try:
        # Demo 1: Gaia stars
        widgets.append(demo_gaia_stars())
        
        # Demo 2: SDSS galaxies  
        widgets.append(demo_sdss_galaxies())
        
        # Demo 3: TNG50 simulation
        widgets.append(demo_tng50_simulation())
        
        # Demo 4: NSA galaxies
        widgets.append(demo_nsa_galaxies())
        
        print(f"\n✅ All {len(widgets)} visualizations created successfully!")
        print("\n💡 Tips:")
        print("   - Click and drag to navigate")
        print("   - Scroll to zoom")
        print("   - Right-click for simulation control")
        
        return widgets
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


if __name__ == "__main__":
    main() 