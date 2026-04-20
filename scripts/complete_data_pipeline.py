#!/usr/bin/env python3
"""
AstroLab Catalog Generation Pipeline
=====================================

Builds the consolidated AstroLab catalog from all processed survey data
in data/processed/. Auto-discovers available surveys and merges them
with cosmic web structure classifications and visualizations.

Usage:
    python scripts/complete_data_pipeline.py
    python scripts/complete_data_pipeline.py --surveys gaia nsa panstarrs
    python scripts/complete_data_pipeline.py --skip-cosmic-web
"""

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
import torch

from astro_lab.config import get_data_paths
from astro_lab.data.analysis.cosmic_web import ScalableCosmicWebAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def discover_surveys(processed_dir: Path) -> dict[str, Path]:
    """Auto-discover all surveys with a main parquet in data/processed/."""
    surveys = {}
    for survey_dir in sorted(processed_dir.iterdir()):
        if not survey_dir.is_dir():
            continue
        main_pq = survey_dir / f"{survey_dir.name}.parquet"
        if main_pq.exists():
            surveys[survey_dir.name] = main_pq
    return surveys


def load_surveys(
    survey_paths: dict[str, Path],
) -> dict[str, pl.DataFrame]:
    """Load all surveys via DuckDB, ensure x/y/z coordinates exist."""
    con = duckdb.connect(":memory:")
    loaded = {}

    for name, path in survey_paths.items():
        try:
            df = con.execute(
                f"SELECT * FROM read_parquet('{path.as_posix()}')"
            ).pl()

            # Derive x/y/z from ra/dec/distance_pc if missing
            if "x" not in df.columns and all(
                c in df.columns for c in ["ra", "dec", "distance_pc"]
            ):
                ra_rad = pl.col("ra") * (math.pi / 180)
                dec_rad = pl.col("dec") * (math.pi / 180)
                d = pl.col("distance_pc")
                df = df.with_columns(
                    (ra_rad.cos() * dec_rad.cos() * d).alias("x"),
                    (ra_rad.sin() * dec_rad.cos() * d).alias("y"),
                    (dec_rad.sin() * d).alias("z"),
                )

            if not all(c in df.columns for c in ["x", "y", "z"]):
                logger.warning(f"  {name}: missing x/y/z coordinates, skipping")
                continue

            loaded[name] = df
            logger.info(f"  {name}: {len(df):>10,} sources, {df.shape[1]} cols")
        except Exception as e:
            logger.warning(f"  {name}: failed to load: {e}")

    con.close()
    return loaded


def build_catalog(
    surveys: dict[str, pl.DataFrame],
    clustering_scales: list[float],
    skip_cosmic_web: bool = False,
) -> pl.DataFrame:
    """Merge surveys and add cosmic web features."""

    # Tag each survey and collect
    frames = []
    for name, df in surveys.items():
        df = df.with_columns(pl.lit(name).alias("survey"))
        frames.append(df.select(
            ["x", "y", "z", "survey"]
            + [c for c in df.columns if c not in ["x", "y", "z", "survey"]]
        ))

    # Diagonal concat (fills missing columns with null)
    combined = pl.concat(frames, how="diagonal_relaxed")
    logger.info(f"Combined: {len(combined):,} sources, {combined.shape[1]} cols")

    # Cosmic web analysis
    if not skip_cosmic_web:
        logger.info(f"Cosmic web analysis at scales {clustering_scales} pc ...")
        coords = torch.tensor(
            combined.select(["x", "y", "z"]).to_numpy(), dtype=torch.float32
        )
        analyzer = ScalableCosmicWebAnalyzer(max_points_per_batch=100_000)
        try:
            cw_results = analyzer.analyze_cosmic_web(
                coordinates=coords,
                scales=clustering_scales,
                use_adaptive_sampling=True,
            )
            for scale in clustering_scales:
                scale_key = f"scale_{scale:.1f}"
                ms = cw_results.get("multi_scale", {}).get(scale_key, {})
                if "structure_class" in ms:
                    combined = combined.with_columns(
                        pl.Series(f"cosmic_web_class_{scale:.1f}pc", ms["structure_class"].cpu().numpy())
                    )
                if "density" in ms:
                    combined = combined.with_columns(
                        pl.Series(f"density_{scale:.1f}pc", ms["density"].cpu().numpy())
                    )
                if "anisotropy" in ms:
                    combined = combined.with_columns(
                        pl.Series(f"anisotropy_{scale:.1f}pc", ms["anisotropy"].cpu().numpy())
                    )
            logger.info("  Cosmic web features added")
        except Exception as e:
            logger.warning(f"  Cosmic web analysis failed: {e}")
    else:
        logger.info("Skipping cosmic web analysis")

    combined = combined.with_columns(
        pl.lit("v2.0").alias("catalog_version"),
        pl.lit(datetime.now().isoformat()).alias("processing_date"),
    )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build consolidated AstroLab catalog from processed survey data"
    )
    parser.add_argument(
        "--surveys", nargs="+", default=None,
        help="Surveys to include (default: all discovered)",
    )
    parser.add_argument(
        "--clustering-scales", type=float, nargs="+",
        default=[5.0, 10.0, 25.0, 50.0],
        help="Clustering scales in parsecs",
    )
    parser.add_argument(
        "--skip-cosmic-web", action="store_true",
        help="Skip cosmic web analysis (faster)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/catalogs"),
        help="Output directory",
    )
    args = parser.parse_args()

    paths = get_data_paths()
    processed_dir = Path(paths["processed_dir"])

    # Discover surveys
    available = discover_surveys(processed_dir)
    logger.info(f"Found {len(available)} surveys in {processed_dir}")

    if args.surveys:
        selected = {k: v for k, v in available.items() if k in args.surveys}
        missing = set(args.surveys) - set(selected)
        if missing:
            logger.warning(f"Not found: {missing}")
    else:
        selected = available

    if not selected:
        logger.error("No surveys to process")
        return 1

    # Load
    logger.info(f"\nLoading {len(selected)} surveys:")
    loaded = load_surveys(selected)
    if not loaded:
        logger.error("No surveys could be loaded")
        return 1

    # Build catalog
    catalog = build_catalog(
        loaded,
        clustering_scales=args.clustering_scales,
        skip_cosmic_web=args.skip_cosmic_web,
    )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "astrolab_catalog_v2.parquet"
    catalog.write_parquet(catalog_path, compression="zstd")

    sample_path = args.output_dir / "astrolab_catalog_v2_sample.parquet"
    catalog.head(10_000).write_parquet(sample_path, compression="zstd")

    logger.info(f"\nCatalog: {catalog_path} ({catalog_path.stat().st_size / (1024**2):.1f} MB)")
    logger.info(f"Sample:  {sample_path}")
    logger.info(f"Sources: {len(catalog):,} across {catalog['survey'].n_unique()} surveys")

    # Per-survey breakdown
    for row in catalog.group_by("survey").len().sort("len", descending=True).iter_rows():
        logger.info(f"  {row[0]}: {row[1]:,}")

    # Visualizations
    try:
        from generate_visualizations import generate_all_visualizations
        generate_all_visualizations(
            catalog_path=catalog_path,
            output_dir=Path("data/visualizations"),
        )
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
