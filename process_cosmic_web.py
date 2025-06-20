#!/usr/bin/env python3
"""Process 3 million Gaia DR3 stars with cosmic web analysis."""

import time
from pathlib import Path

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import SkyCoord

from src.astro_lab.data.core import load_gaia_data
from src.astro_lab.tensors import Spatial3DTensor


def main():
    print("🌌 COSMIC WEB ANALYSIS - 3 Million Gaia DR3 Stars")
    print("=" * 60)

    # Load all 3 million stars
    start_time = time.time()
    print("📊 Loading 3,000,000 Gaia DR3 stars...")
    gaia_tensor = load_gaia_data(max_samples=3000000, return_tensor=True)
    load_time = time.time() - start_time
    print(f"✅ Loaded in {load_time:.1f}s: {len(gaia_tensor):,} stars")

    # Convert to 3D coordinates
    print("🌍 Converting to 3D Cartesian coordinates...")
    start_time = time.time()

    ra = gaia_tensor._data[:, 0].numpy()
    dec = gaia_tensor._data[:, 1].numpy()
    parallax = gaia_tensor._data[:, 2].numpy()

    # Convert to distance and 3D coordinates using Astropy
    distance_pc = 1000.0 / parallax
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, distance=distance_pc * u.pc)
    cartesian = coords.cartesian

    x = cartesian.x.to(u.pc).value
    y = cartesian.y.to(u.pc).value
    z = cartesian.z.to(u.pc).value

    coord_time = time.time() - start_time
    print(f"✅ Converted in {coord_time:.1f}s")

    # Create Spatial3DTensor
    coords_3d = torch.tensor(np.column_stack([x, y, z]), dtype=torch.float32)
    spatial_tensor = Spatial3DTensor(coords_3d, unit="pc")

    print("🌍 3D Volume:")
    print(f"  X: {x.min():.1f} to {x.max():.1f} pc")
    print(f"  Y: {y.min():.1f} to {y.max():.1f} pc")
    print(f"  Z: {z.min():.1f} to {z.max():.1f} pc")

    total_volume = (x.max() - x.min()) * (y.max() - y.min()) * (z.max() - z.min())
    mean_density = len(coords_3d) / total_volume
    print(f"📏 Total volume: {total_volume:.0f} pc³")
    print(f"📊 Mean density: {mean_density:.2e} stars/pc³")

    # Cosmic web clustering at optimal scale
    print("\n🕸️ Cosmic Web Clustering (5 pc scale):")
    start_time = time.time()

    try:
        results = spatial_tensor.cosmic_web_clustering(eps_pc=5.0, min_samples=10)

        cluster_time = time.time() - start_time
        print(f"⏱️ Clustering completed in {cluster_time:.1f}s")

        # Analyze results
        n_groups = results["n_clusters"]
        n_noise = results["n_noise"]

        if n_groups > 0:
            stats = results["cluster_stats"]

            # Find largest groups
            group_sizes = [(k, v["n_stars"]) for k, v in stats.items()]
            group_sizes.sort(key=lambda x: x[1], reverse=True)

            print("\n📈 Cosmic Web Structure:")
            print(f"  Stellar groups: {n_groups:,}")
            print(
                f"  Grouped stars: {len(coords_3d) - n_noise:,} ({(len(coords_3d) - n_noise) / len(coords_3d) * 100:.1f}%)"
            )
            print(
                f"  Isolated stars: {n_noise:,} ({n_noise / len(coords_3d) * 100:.1f}%)"
            )

            print("\n🌟 Largest stellar groups:")
            for i, (group_id, size) in enumerate(group_sizes[:10]):
                group_info = stats[group_id]
                print(
                    f"  #{i + 1}: {size:,} stars, radius {group_info['radius_pc']:.1f} pc, density {group_info['density']:.2e} stars/pc³"
                )

        # Save results
        output_dir = Path("results/cosmic_web_3M")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save cluster labels
        torch.save(results["cluster_labels"], output_dir / "cluster_labels.pt")
        torch.save(coords_3d, output_dir / "coords_3d_pc.pt")

        # Save summary
        with open(output_dir / "cosmic_web_summary.txt", "w") as f:
            f.write("Cosmic Web Analysis - 3 Million Gaia DR3 Stars\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total stars: {len(coords_3d):,}\n")
            f.write(f"Stellar groups: {n_groups:,}\n")
            f.write(
                f"Grouped stars: {len(coords_3d) - n_noise:,} ({(len(coords_3d) - n_noise) / len(coords_3d) * 100:.1f}%)\n"
            )
            f.write(
                f"Isolated stars: {n_noise:,} ({n_noise / len(coords_3d) * 100:.1f}%)\n"
            )
            f.write(f"Mean density: {mean_density:.2e} stars/pc³\n")
            f.write(f"Total volume: {total_volume:.0f} pc³\n")

        print(f"\n💾 Results saved to: {output_dir}")
        print("🎉 Cosmic web analysis complete!")

    except Exception as e:
        print(f"❌ Error in clustering: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
