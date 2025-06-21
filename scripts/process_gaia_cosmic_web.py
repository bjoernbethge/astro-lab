#!/usr/bin/env python3
"""
🌌 Gaia Cosmic Web Processing Script

Processes Gaia DR3 stellar data for cosmic web analysis.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.tensors.spatial_3d import Spatial3DTensor
from astro_lab.utils import calculate_volume, calculate_mean_density

def process_gaia_cosmic_web():
    """Process Gaia data for cosmic web analysis."""
    print("🌟 Processing Gaia Cosmic Web Data")
    print("=" * 40)
    
    # Load Gaia data with cosmic web analysis
    results = create_cosmic_web_loader(
        survey="gaia",
        max_samples=10000,
        scales_mpc=[1.0, 2.0, 5.0, 10.0],
        return_tensor=True
    )
    
    print(f"✅ Loaded {results['n_objects']} stellar objects")
    print(f"📊 Total volume: {results['total_volume']:.2f} Mpc³")
    print(f"🌌 Mean density: {results.get('mean_density', 'N/A')}")
    
    # Create spatial tensor
    if 'positions' in results:
        spatial_tensor = Spatial3DTensor.from_coordinates(results['positions'])
        print(f"🎯 Spatial tensor created: {spatial_tensor.shape}")
        
        # Calculate additional metrics
        volume = calculate_volume(spatial_tensor)
        density = calculate_mean_density(spatial_tensor)
        
        print(f"📏 Calculated volume: {volume:.2f} Mpc³")
        print(f"⚖️  Calculated density: {density:.2e} objects/Mpc³")
    
    # Save results
    output_dir = Path("data/processed/gaia_cosmic_web")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cosmic web results
    import pickle
    with open(output_dir / "cosmic_web_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"💾 Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    try:
        results = process_gaia_cosmic_web()
        print("\n🎉 Gaia cosmic web processing completed successfully!")
    except Exception as e:
        print(f"❌ Error processing Gaia cosmic web data: {e}")
        sys.exit(1)
