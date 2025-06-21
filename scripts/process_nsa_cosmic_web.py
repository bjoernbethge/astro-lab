#!/usr/bin/env python3
"""Process NSA galaxies with cosmic web analysis using integrated Data module functions."""

import sys
import time
from pathlib import Path

import torch

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import create_cosmic_web_loader


def main():
    print("🌌 NSA COSMIC WEB ANALYSIS (Integrated)")
    print("=" * 50)

    start_time = time.time()
    print("📊 Loading NSA survey and performing cosmic web analysis...")
    
    # Use integrated cosmic web analysis
    cosmic_web_results = create_cosmic_web_loader(
        survey="nsa",
        max_samples=None,  # Full dataset
        scales_mpc=[5.0, 10.0, 20.0, 50.0],
    )
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total time: {total_time:.1f}s")

    # Save results
    output_dir = Path("results/nsa_cosmic_web")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save coordinates
    coords_tensor = torch.tensor(cosmic_web_results["coordinates"])
    torch.save(coords_tensor, output_dir / "nsa_coords_3d_mpc.pt")
    
    # Save detailed summary
    with open(output_dir / "nsa_cosmic_web_summary.txt", "w") as f:
        f.write("NSA Cosmic Web Analysis (Integrated Data Module)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Survey: {cosmic_web_results['survey_name']}\n")
        f.write(f"Total objects: {cosmic_web_results['n_objects']:,}\n")
        f.write(f"Total volume: {cosmic_web_results['total_volume']:.0f} Mpc³\n")
        f.write(f"Global density: {cosmic_web_results['global_density']:.2e} obj/Mpc³\n\n")
        
        f.write("Multi-scale clustering results:\n")
        for scale, result in cosmic_web_results["results_by_scale"].items():
            f.write(f"  {scale} Mpc:\n")
            f.write(f"    Groups: {result['n_clusters']}\n")
            f.write(f"    Grouped: {result['grouped_fraction'] * 100:.1f}%\n")
            f.write(f"    Time: {result['time_s']:.1f}s\n")
            f.write(f"    Local density: {result['mean_local_density']:.2e} ± {result['density_variation']:.2e} obj/pc³\n")
            f.write(f"    Density stats: min={result['local_density_stats']['min']:.2e}, ")
            f.write(f"median={result['local_density_stats']['median']:.2e}, ")
            f.write(f"max={result['local_density_stats']['max']:.2e}\n\n")

    print(f"\n💾 Results saved to: {output_dir}")
    print("🎉 NSA cosmic web analysis complete!")


if __name__ == "__main__":
    main()
