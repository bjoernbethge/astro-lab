#!/usr/bin/env python3
"""Process confirmed exoplanets with cosmic web analysis using 3D coordinates."""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.astro_lab.tensors import Spatial3DTensor


def load_exoplanet_data():
    """Load confirmed exoplanet catalog data."""

    file_path = "data/processed/exoplanet_graphs/raw/confirmed_exoplanets.parquet"

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Exoplanet data not found: {file_path}")

    print(f"📁 Loading exoplanet data from: {file_path}")
    df = pl.read_parquet(file_path)

    # Filter out systems without distance measurements
    df = df.filter(pl.col("sy_dist").is_not_null())
    df = df.filter(pl.col("sy_dist") > 0)

    print(f"✅ Loaded {len(df):,} exoplanets with distance data")
    return df


def convert_to_3d_coordinates(df):
    """Convert exoplanet host star coordinates to 3D Cartesian."""

    # Extract coordinates
    ra = df["ra"].to_numpy()
    dec = df["dec"].to_numpy()
    distance_pc = df["sy_dist"].to_numpy()  # Already in parsecs

    print("📊 Coordinate ranges:")
    print(f"  RA: {ra.min():.1f}° - {ra.max():.1f}°")
    print(f"  Dec: {dec.min():.1f}° - {dec.max():.1f}°")
    print(f"  Distance: {distance_pc.min():.1f} - {distance_pc.max():.1f} pc")

    # Convert spherical to Cartesian coordinates
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    x = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance_pc * np.sin(dec_rad)

    coords_3d = np.column_stack([x, y, z])

    print("🌍 3D Volume:")
    print(f"  X: {x.min():.1f} to {x.max():.1f} pc")
    print(f"  Y: {y.min():.1f} to {y.max():.1f} pc")
    print(f"  Z: {z.min():.1f} to {z.max():.1f} pc")

    return coords_3d


def main():
    print("🪐 EXOPLANET COSMIC WEB ANALYSIS")
    print("=" * 50)

    # Load exoplanet data
    start_time = time.time()
    print("📊 Loading confirmed exoplanet catalog...")

    try:
        df = load_exoplanet_data()
        load_time = time.time() - start_time
        print(f"✅ Loaded in {load_time:.1f}s: {len(df):,} exoplanet systems")

        # Convert to 3D coordinates
        print("🌍 Converting to 3D stellar coordinates...")
        start_time = time.time()

        coords_3d = convert_to_3d_coordinates(df)
        coord_time = time.time() - start_time
        print(f"✅ Converted in {coord_time:.1f}s")

        # Create Spatial3DTensor
        coords_tensor = torch.tensor(coords_3d, dtype=torch.float32)
        spatial_tensor = Spatial3DTensor(coords_tensor, unit="pc")

        # Calculate volume and density
        total_volume = (
            (coords_3d[:, 0].max() - coords_3d[:, 0].min())
            * (coords_3d[:, 1].max() - coords_3d[:, 1].min())
            * (coords_3d[:, 2].max() - coords_3d[:, 2].min())
        )
        mean_density = len(coords_3d) / total_volume

        print(f"📏 Total volume: {total_volume:.0f} pc³")
        print(f"📊 Mean density: {mean_density:.2e} exoplanet systems/pc³")

        # Multi-scale cosmic web clustering for exoplanet host stars
        scales = [
            10.0,
            25.0,
            50.0,
            100.0,
            200.0,
        ]  # parsecs - larger scales for sparse exoplanets
        results_summary = {}

        for scale in scales:
            print(f"\n🕸️ Exoplanet Host Star Clustering at {scale} pc scale:")
            start_time = time.time()

            try:
                results = spatial_tensor.cosmic_web_clustering(
                    eps_pc=scale,
                    min_samples=3,  # Lower threshold for sparse exoplanet data
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

                print(f"  Star groups: {n_groups:,}")
                print(
                    f"  Grouped systems: {len(coords_3d) - n_noise:,} ({(len(coords_3d) - n_noise) / len(coords_3d) * 100:.1f}%)"
                )
                print(
                    f"  Isolated systems: {n_noise:,} ({n_noise / len(coords_3d) * 100:.1f}%)"
                )

                if (
                    n_groups > 0 and n_groups < 50
                ):  # Show details for reasonable number of groups
                    stats = results["cluster_stats"]
                    group_sizes = [(k, v["n_stars"]) for k, v in stats.items()]
                    group_sizes.sort(key=lambda x: x[1], reverse=True)

                    print(f"  Top 5 groups: {group_sizes[:5]}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        # Analyze planet properties by stellar groups
        print("\n🪐 Exoplanet Properties Analysis:")

        # Planet radii
        if "pl_rade" in df.columns:
            radii = df["pl_rade"].to_numpy()
            valid_radii = radii[~np.isnan(radii)]
            if len(valid_radii) > 0:
                print(f"  Planet radii: {len(valid_radii):,} planets")
                print(
                    f"    Earth-size (0.5-1.5 R⊕): {np.sum((valid_radii >= 0.5) & (valid_radii <= 1.5)):,}"
                )
                print(
                    f"    Super-Earths (1.5-4 R⊕): {np.sum((valid_radii > 1.5) & (valid_radii <= 4)):,}"
                )
                print(
                    f"    Mini-Neptunes (4-8 R⊕): {np.sum((valid_radii > 4) & (valid_radii <= 8)):,}"
                )
                print(f"    Gas giants (>8 R⊕): {np.sum(valid_radii > 8):,}")

        # Discovery methods
        if "discoverymethod" in df.columns:
            methods = df["discoverymethod"].value_counts()
            print("  Discovery methods:")
            for method, count in methods.head(5).iter_rows():
                print(f"    {method}: {count:,}")

        # Save results
        output_dir = Path("results/exoplanet_cosmic_web")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save coordinates and processed data
        torch.save(coords_tensor, output_dir / "exoplanet_coords_3d_pc.pt")

        # Save planetary properties
        planet_data = {
            "ra": df["ra"].to_numpy(),
            "dec": df["dec"].to_numpy(),
            "distance_pc": df["sy_dist"].to_numpy(),
            "discovery_year": df["disc_year"].to_numpy()
            if "disc_year" in df.columns
            else None,
            "planet_names": df["pl_name"].to_list()
            if "pl_name" in df.columns
            else None,
            "host_names": df["hostname"].to_list()
            if "hostname" in df.columns
            else None,
        }

        if "pl_rade" in df.columns:
            planet_data["planet_radius"] = df["pl_rade"].to_numpy()
        if "pl_masse" in df.columns:
            planet_data["planet_mass"] = df["pl_masse"].to_numpy()

        torch.save(planet_data, output_dir / "exoplanet_properties.pt")

        # Save summary
        with open(output_dir / "exoplanet_cosmic_web_summary.txt", "w") as f:
            f.write("Exoplanet Cosmic Web Analysis\n")
            f.write("=" * 35 + "\n\n")
            f.write(f"Total confirmed exoplanets: {len(df):,}\n")
            f.write(
                f"Distance range: {coords_3d.min():.1f} - {coords_3d.max():.1f} pc\n"
            )
            f.write(f"Mean density: {mean_density:.2e} systems/pc³\n")
            f.write(f"Total volume: {total_volume:.0f} pc³\n\n")

            f.write("Multi-scale clustering results:\n")
            for scale, result in results_summary.items():
                f.write(f"  {scale} pc: {result['n_clusters']} groups, ")
                f.write(f"{result['grouped_fraction'] * 100:.1f}% grouped, ")
                f.write(f"{result['time_s']:.1f}s\n")

        print(f"\n💾 Results saved to: {output_dir}")
        print("🎉 Exoplanet cosmic web analysis complete!")

        # Print final summary
        print("\n📈 SUMMARY:")
        print(f"  Dataset: {len(coords_3d):,} confirmed exoplanet systems")
        print(f"  Volume: {total_volume:.0f} pc³")
        print(f"  Density: {mean_density:.2e} systems/pc³")
        print(f"  Distance range: {coords_3d.min():.1f} - {coords_3d.max():.1f} pc")

        for scale, result in results_summary.items():
            print(
                f"  {scale:5.0f} pc: {result['n_clusters']:4d} groups ({result['grouped_fraction'] * 100:5.1f}% grouped)"
            )

    except Exception as e:
        print(f"❌ Error in exoplanet processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
