#!/usr/bin/env python3
"""
Generate Consolidated AstroLab Catalog
=======================================

Creates a consolidated parquet catalog combining multiple astronomical surveys
with cosmic web structure classifications. Uses DuckDB for efficient data loading.

Features:
- Multi-survey support (Gaia, SDSS, 2MASS)
- Cosmic web classification (filaments, voids, nodes)
- 3D coordinate conversion
- Density field computation
- Multi-scale structure analysis

Usage:
    python scripts/generate_astrolab_catalog.py [--max-samples N] [--output-dir DIR]
"""

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import duckdb
import polars as pl
import torch

from astro_lab.config import get_data_paths
from astro_lab.data.analysis.cosmic_web import ScalableCosmicWebAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_PATHS = {
    "gaia": "gaia/gaia.parquet",
    "sdss": "sdss/sdss.parquet",
    "twomass": "twomass/twomass.parquet",
}


def generate_catalog(
    max_samples: Optional[int] = None,
    output_dir: Path = Path("data/catalogs"),
    clustering_scales: Optional[List[float]] = None,
    include_surveys: Optional[List[str]] = None,
) -> Path:
    """
    Generate consolidated AstroLab catalog with cosmic web features.
    Uses DuckDB for loading and Polars for downstream processing.
    """
    if clustering_scales is None:
        clustering_scales = [5.0, 10.0, 25.0, 50.0]
    if include_surveys is None:
        include_surveys = ["gaia"]

    paths = get_data_paths()
    processed_dir = Path(paths["processed_dir"])

    logger.info("=" * 80)
    logger.info("🌌 AstroLab Catalog Generation (DuckDB)")
    logger.info("=" * 80)
    logger.info(f"Max samples: {max_samples or 'all'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Clustering scales: {clustering_scales}")
    logger.info(f"Surveys: {', '.join(include_surveys)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load surveys via DuckDB
    logger.info("\n📊 Step 1: Loading Surveys (DuckDB)")
    logger.info("-" * 80)

    con = duckdb.connect(":memory:")
    surveys = {}

    for survey in include_surveys:
        try:
            rel_path = PROCESSED_PATHS.get(survey)
            parquet_path = processed_dir / rel_path
            if not parquet_path.exists():
                logger.warning(f"   ⚠ {survey}: file not found {parquet_path}")
                continue

            limit_clause = f" LIMIT {max_samples}" if max_samples else ""
            df = con.execute(
                f"SELECT * FROM read_parquet('{parquet_path.as_posix()}'){limit_clause}"
            ).pl()

            # Ensure x,y,z exist (spherical to Cartesian)
            if "x" not in df.columns and all(c in df.columns for c in ["ra", "dec", "distance_pc"]):
                ra_rad = pl.col("ra") * (math.pi / 180)
                dec_rad = pl.col("dec") * (math.pi / 180)
                d = pl.col("distance_pc")
                df = df.with_columns(
                    (ra_rad.cos() * dec_rad.cos() * d).alias("x"),
                    (ra_rad.sin() * dec_rad.cos() * d).alias("y"),
                    (dec_rad.sin() * d).alias("z"),
                )

            if "x" not in df.columns or "y" not in df.columns or "z" not in df.columns:
                logger.warning(f"   ⚠ {survey}: missing x,y,z coordinates")
                continue

            surveys[survey] = df
            logger.info(f"   ✓ {survey}: {len(df):,} sources")
        except Exception as e:
            logger.warning(f"   ⚠ {survey} failed: {e}")

    con.close()

    if not surveys:
        raise RuntimeError("No surveys could be loaded")

    # Single survey: use directly; multi-survey: simple concat on ra/dec (no cross-match for now)
    if len(surveys) == 1:
        combined_df = list(surveys.values())[0]
        coord_cols = ["x", "y", "z"]
        logger.info(f"\n📋 Using single survey: {list(surveys.keys())[0]}")
    else:
        # Multi-survey: use Gaia as reference, left-join others by nearest ra/dec
        ref = "gaia" if "gaia" in surveys else list(surveys.keys())[0]
        combined_df = surveys[ref]
        prefix = f"{ref}_"
        if prefix + "x" not in combined_df.columns:
            combined_df = combined_df.rename({c: f"{ref}_{c}" for c in ["x", "y", "z"] if c in combined_df.columns})
        coord_cols = [f"{ref}_x", f"{ref}_y", f"{ref}_z"]
        for s in surveys:
            if s == ref:
                continue
            # Simple merge: for now just use reference coords (full cross-match would need spatial join)
            logger.info(f"   Note: {s} loaded but using {ref} coordinates for cosmic web")
        logger.info(f"\n📋 Reference survey: {ref}")

    # Resolve coord columns
    if coord_cols[0] not in combined_df.columns:
        coord_cols = ["x", "y", "z"]

    # Step 2: Cosmic web analysis
    logger.info("\n🕸️  Step 2: Cosmic Web Structure Analysis")
    logger.info("-" * 80)

    coords_array = combined_df.select(coord_cols).to_numpy()
    coordinates = torch.tensor(coords_array, dtype=torch.float32)

    analyzer = ScalableCosmicWebAnalyzer(max_points_per_batch=100000)
    logger.info(f"Analyzing {coordinates.shape[0]:,} sources at {len(clustering_scales)} scales")
    cw_results = analyzer.analyze_cosmic_web(
        coordinates=coordinates,
        scales=clustering_scales,
        use_adaptive_sampling=True,
    )
    logger.info("   ✓ Cosmic web analysis complete")

    # Step 3: Add cosmic web features
    logger.info("\n📝 Step 3: Adding Cosmic Web Features")
    logger.info("-" * 80)

    for scale in clustering_scales:
        scale_key = f"scale_{scale:.1f}"
        if "multi_scale" in cw_results and scale_key in cw_results["multi_scale"]:
            scale_results = cw_results["multi_scale"][scale_key]
            if "structure_class" in scale_results:
                combined_df = combined_df.with_columns(
                    pl.Series(
                        f"cosmic_web_class_{scale:.1f}pc",
                        scale_results["structure_class"].cpu().numpy(),
                    )
                )
            if "density" in scale_results:
                combined_df = combined_df.with_columns(
                    pl.Series(f"density_{scale:.1f}pc", scale_results["density"].cpu().numpy())
                )
            if "anisotropy" in scale_results:
                combined_df = combined_df.with_columns(
                    pl.Series(f"anisotropy_{scale:.1f}pc", scale_results["anisotropy"].cpu().numpy())
                )

    # Step 4: Metadata
    combined_df = combined_df.with_columns(
        pl.lit("v1.0").alias("catalog_version"),
        pl.lit(datetime.now().isoformat()).alias("processing_date"),
    )

    # Step 5: Save with DuckDB (efficient parquet write)
    logger.info("\n💾 Step 5: Saving Catalog")
    logger.info("-" * 80)

    catalog_path = output_dir / "astrolab_catalog_v1.parquet"
    combined_df.write_parquet(catalog_path, compression="zstd")

    sample_path = output_dir / "astrolab_catalog_v1_sample.parquet"
    combined_df.head(10000).write_parquet(sample_path, compression="zstd")

    logger.info(f"   ✓ Full catalog: {catalog_path}")
    logger.info(f"   ✓ Sample: {sample_path}")
    logger.info(f"   ✓ Size: {catalog_path.stat().st_size / (1024**2):.1f} MB")
    logger.info(f"   ✓ Sources: {len(combined_df):,}")
    logger.info("\n" + "=" * 80)
    logger.info("✅ Catalog generation complete!")
    logger.info("=" * 80)

    return catalog_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate consolidated AstroLab catalog with cosmic web features"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (default: all)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/catalogs"),
        help="Output directory",
    )
    parser.add_argument(
        "--clustering-scales",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 25.0, 50.0],
        help="Clustering scales in pc",
    )
    parser.add_argument(
        "--surveys",
        nargs="+",
        default=["gaia"],
        help="Surveys to include",
    )
    args = parser.parse_args()

    try:
        catalog_path = generate_catalog(
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            clustering_scales=args.clustering_scales,
            include_surveys=args.surveys,
        )
        print(f"\n✅ Success! Catalog saved to: {catalog_path}")
        print(f"    import polars as pl")
        print(f"    catalog = pl.read_parquet('{catalog_path}')")
    except Exception as e:
        logger.error(f"❌ Catalog generation failed: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
