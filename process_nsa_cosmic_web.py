#!/usr/bin/env python3
"""Process NSA galaxies with cosmic web analysis using 3D coordinates."""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.astro_lab.tensors import Spatial3DTensor


def load_nsa_data(max_samples=None):
    """Load NSA galaxy catalog data."""

    # Try different NSA files
    nsa_files = [
        "data/processed/nsa/nsa_v1_0_1_processed.parquet",
        "data/processed/nsa/nsa_catalog.parquet",
        "data/processed/nsa/nsa_v0_1_2_train.parquet",
    ]

    for file_path in nsa_files:
        if Path(file_path).exists():
            print(f"📁 Loading NSA data from: {file_path}")
            df = pl.read_parquet(file_path)

            if max_samples and len(df) > max_samples:
                df = df.sample(max_samples, seed=42)

            print(f"✅ Loaded {len(df):,} galaxies")
            return df

    raise FileNotFoundError("No NSA data files found")


def convert_to_3d_coordinates(df):
    """Convert NSA galaxy coordinates to 3D Cartesian."""

    # Extract coordinates
    ra = df["RA"].to_numpy() if "RA" in df.columns else df["ra"].to_numpy()
    dec = df["DEC"].to_numpy() if "DEC" in df.columns else df["dec"].to_numpy()

    # Get redshift - try different column names
    z_col = None
    for col in ["Z", "z", "ZDIST", "redshift"]:
        if col in df.columns:
            z_col = col
            break

    if z_col is None:
        raise ValueError("No redshift column found")

    z = df[z_col].to_numpy()

    print("📊 Coordinate ranges:")
    print(f"  RA: {ra.min():.1f}° - {ra.max():.1f}°")
    print(f"  Dec: {dec.min():.1f}° - {dec.max():.1f}°")
    print(f"  Redshift: {z.min():.4f} - {z.max():.4f}")

    # Convert redshift to comoving distance
    c_km_s = 299792.458  # km/s
    H0 = 70.0  # km/s/Mpc
    distance_mpc = (c_km_s * z) / H0  # Mpc

    print(f"  Distance: {distance_mpc.min():.1f} - {distance_mpc.max():.1f} Mpc")

    # Convert spherical to Cartesian coordinates
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    x = distance_mpc * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance_mpc * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = distance_mpc * np.sin(dec_rad)

    coords_3d = np.column_stack([x, y, z_coord])

    print("🌍 3D Volume:")
    print(f"  X: {x.min():.1f} to {x.max():.1f} Mpc")
    print(f"  Y: {y.min():.1f} to {y.max():.1f} Mpc")
    print(f"  Z: {z_coord.min():.1f} to {z_coord.max():.1f} Mpc")

    return coords_3d


def main():
    print("🌌 NSA COSMIC WEB ANALYSIS")
    print("=" * 50)

    # Load NSA data - full dataset
    start_time = time.time()
    print("📊 Loading NSA galaxy catalog...")

    try:
        df = load_nsa_data()  # Full dataset
        load_time = time.time() - start_time
        print(f"✅ Loaded in {load_time:.1f}s: {len(df):,} galaxies")

        # Convert to 3D coordinates
        print("🌍 Converting to 3D comoving coordinates...")
        start_time = time.time()

        coords_3d = convert_to_3d_coordinates(df)
        coord_time = time.time() - start_time
        print(f"✅ Converted in {coord_time:.1f}s")

        # Create Spatial3DTensor
        coords_tensor = torch.tensor(coords_3d, dtype=torch.float32)
        spatial_tensor = Spatial3DTensor(coords_tensor, unit="Mpc")

        # Calculate volume and density
        total_volume = (
            (coords_3d[:, 0].max() - coords_3d[:, 0].min())
            * (coords_3d[:, 1].max() - coords_3d[:, 1].min())
            * (coords_3d[:, 2].max() - coords_3d[:, 2].min())
        )
        mean_density = len(coords_3d) / total_volume

        print(f"📏 Total volume: {total_volume:.0f} Mpc³")
        print(f"📊 Mean density: {mean_density:.2e} galaxies/Mpc³")

        # Multi-scale cosmic web clustering
        scales = [5.0, 10.0, 20.0, 50.0]  # Mpc
        results_summary = {}

        for scale in scales:
            print(f"\n🕸️ Cosmic Web Clustering at {scale} Mpc scale:")
            start_time = time.time()

            try:
                results = spatial_tensor.cosmic_web_clustering(
                    eps_pc=scale * 1_000_000,  # Convert Mpc to pc
                    min_samples=5,
                    algorithm="dbscan",
                )

                cluster_time = time.time() - start_time
                print(f"⏱️ Clustering completed in {cluster_time:.1f}s")

                n_groups = results["n_clusters"]
                n_noise = results["n_noise"]

                results_summary[scale] = {
                    "n_clusters": n_groups,
                    "n_noise": n_noise,
                    "grouped_fraction": (len(coords_3d) - n_noise) / len(coords_3d),
                    "time_s": cluster_time,
                }

                print(f"  Galaxy groups: {n_groups:,}")
                print(
                    f"  Grouped galaxies: {len(coords_3d) - n_noise:,} ({(len(coords_3d) - n_noise) / len(coords_3d) * 100:.1f}%)"
                )
                print(
                    f"  Isolated galaxies: {n_noise:,} ({n_noise / len(coords_3d) * 100:.1f}%)"
                )

                if n_groups > 0 and n_groups < 20:
                    stats = results["cluster_stats"]
                    group_sizes = [(k, v["n_stars"]) for k, v in stats.items()]
                    group_sizes.sort(key=lambda x: x[1], reverse=True)
                    print(f"  Top 5 groups: {group_sizes[:5]}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        # Save results
        output_dir = Path("results/nsa_cosmic_web")
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(coords_tensor, output_dir / "nsa_coords_3d_mpc.pt")

        # Save summary
        with open(output_dir / "nsa_cosmic_web_summary.txt", "w") as f:
            f.write("NSA Cosmic Web Analysis\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total galaxies: {len(coords_3d):,}\n")

            # Handle different redshift column names
            z_col = "Z" if "Z" in df.columns else "z"
            f.write(f"Redshift range: {df[z_col].min():.4f} - {df[z_col].max():.4f}\n")
            f.write(
                f"Distance range: {coords_3d.min():.1f} - {coords_3d.max():.1f} Mpc\n"
            )
            f.write(f"Mean density: {mean_density:.2e} galaxies/Mpc³\n")
            f.write(f"Total volume: {total_volume:.0f} Mpc³\n\n")

            f.write("Multi-scale clustering results:\n")
            for scale, result in results_summary.items():
                f.write(f"  {scale} Mpc: {result['n_clusters']} groups, ")
                f.write(f"{result['grouped_fraction'] * 100:.1f}% grouped, ")
                f.write(f"{result['time_s']:.1f}s\n")

        print(f"\n💾 Results saved to: {output_dir}")
        print("🎉 NSA cosmic web analysis complete!")

        # Print final summary
        print("\n📈 SUMMARY:")
        print(f"  Dataset: {len(coords_3d):,} NSA galaxies")
        print(f"  Volume: {total_volume:.0f} Mpc³")
        print(f"  Density: {mean_density:.2e} gal/Mpc³")

        for scale, result in results_summary.items():
            print(
                f"  {scale:4.0f} Mpc: {result['n_clusters']:4d} groups ({result['grouped_fraction'] * 100:5.1f}% grouped)"
            )

    except Exception as e:
        print(f"❌ Error in NSA processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
