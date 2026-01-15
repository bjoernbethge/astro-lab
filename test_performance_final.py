"""Final performance test for the data pipeline."""

import time
from typing import Dict

import numpy as np
import polars as pl
import psutil
import torch

from astro_lab.data.samplers.RadiusSampler import RadiusSampler


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    cpu_mb = psutil.virtual_memory().used / (1024**2)
    gpu_mb = 0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / (1024**2)
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def benchmark_preprocessor(name: str, n_samples: int = 10000):
    """Benchmark a preprocessor."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {name} preprocessor with {n_samples:,} samples")
    print(f"{'=' * 60}")

    from astro_lab.data.preprocessors.gaia import GaiaPreprocessor

    # Create synthetic data
    if name == "gaia":
        data = {
            "source_id": list(range(n_samples)),
            "ra": np.random.uniform(0, 360, n_samples),
            "dec": np.random.uniform(-90, 90, n_samples),
            "parallax": np.random.uniform(0.1, 10, n_samples),
            "parallax_error": np.random.uniform(0.01, 0.1, n_samples),
            "pmra": np.random.normal(0, 10, n_samples),
            "pmdec": np.random.normal(0, 10, n_samples),
            "phot_g_mean_mag": np.random.uniform(10, 20, n_samples),
            "phot_bp_mean_mag": np.random.uniform(10, 20, n_samples),
            "phot_rp_mean_mag": np.random.uniform(10, 20, n_samples),
            "astrometric_excess_noise": np.random.uniform(0, 1, n_samples),
            "ruwe": np.random.uniform(0.8, 1.2, n_samples),
        }
    else:
        data = {
            "id": list(range(n_samples)),
            "ra": np.random.uniform(0, 360, n_samples),
            "dec": np.random.uniform(-90, 90, n_samples),
            "z": np.random.uniform(0, 0.1, n_samples),
        }

    df = pl.DataFrame(data)
    if name == "gaia":
        preprocessor = GaiaPreprocessor()
    else:
        # For other surveys, use Gaia as default for now
        preprocessor = GaiaPreprocessor()

    # Benchmark
    mem_start = get_memory_usage()
    start_time = time.time()

    filtered = preprocessor.filter(df)
    filter_time = time.time() - start_time

    transform_start = time.time()
    transformed = preprocessor.transform(filtered)
    transform_time = time.time() - transform_start

    feature_start = time.time()
    features = preprocessor.extract_features(transformed)
    feature_time = time.time() - feature_start

    total_time = time.time() - start_time
    mem_end = get_memory_usage()

    # Results
    print(f"Filter:    {filter_time:.3f}s ({len(df)} → {len(filtered)} objects)")
    print(f"Transform: {transform_time:.3f}s")
    print(f"Features:  {feature_time:.3f}s ({features.shape})")
    print(f"Total:     {total_time:.3f}s ({n_samples / total_time:.0f} objects/sec)")
    print(f"Memory:    +{(mem_end['cpu_mb'] - mem_start['cpu_mb']):.1f} MB CPU")

    return total_time


def benchmark_samplers(n_nodes: int = 5000):
    """Benchmark graph samplers."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking graph samplers with {n_nodes:,} nodes")
    print(f"{'=' * 60}")

    from astro_lab.data.samplers import (
        ClusterSampler,
        DBSCANClusterSampler,
        KNNSampler,
    )

    coords = torch.randn(n_nodes, 3)
    feats = torch.randn(n_nodes, 10)

    results = {}

    for sampler_type in ["knn", "radius", "dbscan", "cluster"]:
        print(f"\n{sampler_type.upper()} Sampler:")

        try:
            # Get sampler with appropriate config
            if sampler_type == "knn":
                config = {"k": 20}
            elif sampler_type == "radius":
                config = {"radius": 1.0}
            elif sampler_type == "dbscan":
                config = {"eps": 0.5, "min_samples": 5}
            elif sampler_type == "cluster":
                config = {"num_parts": 50}
            else:
                config = {}

            if sampler_type == "knn":
                sampler = KNNSampler(**config)
            elif sampler_type == "radius":
                sampler = RadiusSampler(**config)
            elif sampler_type == "dbscan":
                sampler = DBSCANClusterSampler(**config)
            elif sampler_type == "cluster":
                sampler = ClusterSampler(**config)
            else:
                sampler = KNNSampler(**config)

            # Benchmark graph creation
            start = time.time()
            graph = sampler.create_graph(coords, feats)
            elapsed = time.time() - start

            print(f"  Time:  {elapsed:.3f}s")
            print(f"  Nodes: {graph.num_nodes}")
            print(f"  Edges: {graph.num_edges}")
            print(f"  Avg degree: {graph.num_edges / graph.num_nodes:.1f}")

            results[sampler_type] = elapsed

        except Exception as e:
            print(f"  Error: {e}")
            results[sampler_type] = None

    return results


def benchmark_optimized():
    """Benchmark optimized vs standard preprocessor."""
    print(f"\n{'=' * 60}")
    print("Comparing standard vs optimized preprocessor")
    print(f"{'=' * 60}")

    from astro_lab.data.preprocessors.gaia import GaiaPreprocessor
    from astro_lab.data.preprocessors.optimized import get_optimized_preprocessor

    n_samples = 50000

    # Create test data
    data = {
        "source_id": list(range(n_samples)),
        "ra": np.random.uniform(0, 360, n_samples),
        "dec": np.random.uniform(-90, 90, n_samples),
        "parallax": np.random.uniform(0.1, 10, n_samples),
        "parallax_error": np.random.uniform(0.01, 0.1, n_samples),
        "pmra": np.random.normal(0, 10, n_samples),
        "pmdec": np.random.normal(0, 10, n_samples),
        "phot_g_mean_mag": np.random.uniform(10, 20, n_samples),
        "phot_bp_mean_mag": np.random.uniform(10, 20, n_samples),
        "phot_rp_mean_mag": np.random.uniform(10, 20, n_samples),
        "astrometric_excess_noise": np.random.uniform(0, 1, n_samples),
        "ruwe": np.random.uniform(0.8, 1.2, n_samples),
    }
    df = pl.DataFrame(data)

    # Standard preprocessor
    print("\nStandard Preprocessor:")
    std_preprocessor = GaiaPreprocessor()
    std_start = time.time()
    std_result = std_preprocessor.preprocess(df)
    std_time = time.time() - std_start
    print(f"  Time: {std_time:.3f}s ({n_samples / std_time:.0f} objects/sec)")

    # Optimized preprocessor
    print("\nOptimized Preprocessor:")
    opt_preprocessor = get_optimized_preprocessor("gaia")
    opt_start = time.time()
    opt_result = opt_preprocessor.preprocess(df)
    opt_time = time.time() - opt_start
    print(f"  Time: {opt_time:.3f}s ({n_samples / opt_time:.0f} objects/sec)")

    # Speedup
    speedup = std_time / opt_time
    print(f"\nSpeedup: {speedup:.2f}x")

    # Cache stats
    cache_stats = opt_preprocessor.get_cache_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate'] * 100:.1f}%")


def main():
    print("ASTROLAB DATA PIPELINE PERFORMANCE BENCHMARK")
    print("=" * 60)

    # 1. Benchmark preprocessors
    preprocessors = ["gaia", "sdss", "exoplanet"]
    for name in preprocessors:
        try:
            benchmark_preprocessor(name, n_samples=10000)
        except Exception as e:
            print(f"Error with {name}: {e}")

    # 2. Benchmark samplers
    benchmark_samplers(n_nodes=5000)

    # 3. Compare optimized vs standard
    benchmark_optimized()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
