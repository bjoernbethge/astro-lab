#!/usr/bin/env python3
"""Process LINEAR survey with cosmic web analysis using integrated Data module functions."""

import time
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import polars as pl
import torch

from astro_lab.data.core import create_cosmic_web_loader


def main():
    print("🌌 LINEAR COSMIC WEB ANALYSIS (Integrated)")
    print("=" * 50)

    # Lade echten LINEAR-Katalog
    data_path = Path("data/raw/linear/linear_raw.parquet")
    if not data_path.exists():
        print(f"❌ LINEAR-Katalog nicht gefunden: {data_path}")
        return

    start_time = time.time()
    print("📊 Lade LINEAR Survey und führe Cosmic Web-Analyse durch...")
    
    # Nutze die integrierte Cosmic Web-Analyse
    cosmic_web_results = create_cosmic_web_loader(
        survey="linear",
        max_samples=None,  # Vollständiger Datensatz
        scales_mpc=[5.0, 10.0, 20.0, 50.0],
    )
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Gesamtzeit: {total_time:.1f}s")

    # Ergebnisse speichern
    output_dir = Path("results/linear_cosmic_web")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Koordinaten speichern
    coords_tensor = torch.tensor(cosmic_web_results["coordinates"])
    torch.save(coords_tensor, output_dir / "linear_coords_3d_mpc.pt")
    
    # Detaillierte Zusammenfassung speichern
    with open(output_dir / "linear_cosmic_web_summary.txt", "w") as f:
        f.write("LINEAR Cosmic Web Analysis (Integrated Data Module)\n")
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
    print("🎉 LINEAR cosmic web analysis complete!")


if __name__ == "__main__":
    main() 