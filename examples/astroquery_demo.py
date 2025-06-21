#!/usr/bin/env python3
"""
Demo script for AstroQuery integration with AstroLab.
Shows how to load astronomical data from various surveys.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import load_survey_data, create_cosmic_web_loader

def demo_gaia_data():
    """Demonstrate Gaia DR3 data loading."""
    print("🌟 Gaia DR3 Data Demo")
    print("=" * 25)
    
    # Load Gaia data
    data = load_survey_data(
        survey="gaia",
        magnitude_limit=12.0,
        max_samples=1000
    )
    
    print(f"Loaded {len(data)} Gaia stars")
    print(f"Columns: {list(data.columns)}")
    print(f"RA range: {data['ra'].min():.2f}° - {data['ra'].max():.2f}°")
    print(f"Dec range: {data['dec'].min():.2f}° - {data['dec'].max():.2f}°")
    print(f"Distance range: {data['distance'].min():.1f} - {data['distance'].max():.1f} pc")
    
    return data

def demo_sdss_data():
    """Demonstrate SDSS data loading."""
    print("\n🌌 SDSS Data Demo")
    print("=" * 20)
    
    # Load SDSS data
    data = load_survey_data(
        survey="sdss",
        max_samples=500
    )
    
    print(f"Loaded {len(data)} SDSS galaxies")
    print(f"Columns: {list(data.columns)}")
    print(f"Redshift range: {data['z'].min():.3f} - {data['z'].max():.3f}")
    print(f"Magnitude range: {data['r'].min():.2f} - {data['r'].max():.2f}")
    
    return data

def demo_cosmic_web_analysis():
    """Demonstrate cosmic web analysis."""
    print("\n🌐 Cosmic Web Analysis Demo")
    print("=" * 30)
    
    # Perform cosmic web analysis
    results = create_cosmic_web_loader(
        survey="gaia",
        max_samples=500,
        scales_mpc=[5.0, 10.0, 20.0]
    )
    
    print(f"Analyzed {results['n_objects']} objects")
    print(f"Total volume: {results['total_volume']:.0f} Mpc³")
    print(f"Global density: {results['global_density']:.2e} obj/Mpc³")
    
    # Show multi-scale results
    for scale, result in results["results_by_scale"].items():
        print(f"\n{scale} Mpc scale:")
        print(f"  Groups: {result['n_clusters']}")
        print(f"  Grouped fraction: {result['grouped_fraction']*100:.1f}%")
        print(f"  Mean local density: {result['mean_local_density']:.2e} obj/pc³")
    
    return results

def demo_multi_survey_comparison():
    """Demonstrate multi-survey comparison."""
    print("\n🔄 Multi-Survey Comparison Demo")
    print("=" * 35)
    
    surveys = ["gaia", "sdss", "nsa"]
    results = {}
    
    for survey in surveys:
        print(f"\n📊 Loading {survey} data...")
        try:
            data = load_survey_data(survey=survey, max_samples=200)
            results[survey] = {
                "n_objects": len(data),
                "columns": list(data.columns),
                "coverage": f"{data['ra'].min():.1f}° - {data['ra'].max():.1f}° RA"
            }
            print(f"✅ {survey}: {len(data)} objects")
        except Exception as e:
            print(f"❌ {survey}: {e}")
            results[survey] = {"error": str(e)}
    
    # Summary
    print(f"\n📋 Survey Summary:")
    for survey, info in results.items():
        if "error" not in info:
            print(f"  {survey}: {info['n_objects']} objects, {len(info['columns'])} columns")
        else:
            print(f"  {survey}: Error - {info['error']}")
    
    return results

if __name__ == "__main__":
    print("🌌 AstroQuery Integration Demo")
    print("=" * 35)
    
    try:
        # Run demos
        gaia_data = demo_gaia_data()
        sdss_data = demo_sdss_data()
        cosmic_results = demo_cosmic_web_analysis()
        survey_comparison = demo_multi_survey_comparison()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"   - Gaia: {len(gaia_data)} stars")
        print(f"   - SDSS: {len(sdss_data)} galaxies")
        print(f"   - Cosmic Web: {cosmic_results['n_objects']} objects analyzed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)
