#!/usr/bin/env python3
"""
AstroLab Preprocess CLI
======================

Complete data preparation pipeline:
1. Load raw data (if needed)
2. Clean and preprocess data
3. Create SurveyTensorDicts with 3D spatial coordinates
4. Build graph structures for training
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from astro_lab.data.preprocessors import get_preprocessor


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def add_preprocess_arguments(parser):
    parser.add_argument(
        "--surveys",
        nargs="+",
        required=True,
        choices=[
            "gaia",
            "sdss",
            "nsa",
            "tng50",
            "exoplanet",
            "twomass",
            "wise",
            "panstarrs",
            "des",
            "euclid",
            "linear",
            "rrlyrae",
        ],
        help="Surveys to preprocess",
    )
    parser.add_argument(
        "--k-neighbors",
        "-k",
        type=int,
        default=20,
        help="Number of nearest neighbors for graph (default: 20)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-processing even if files exist",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")


def main(args=None) -> int:
    """Main entry point for preprocess command."""
    parser = argparse.ArgumentParser(
        description="Preprocess astronomical survey data for training",
        epilog="""
Examples:
  # Preprocess single survey
  astro-lab preprocess --surveys gaia

  # Preprocess multiple surveys  
  astro-lab preprocess --surveys gaia sdss nsa

  # Force re-processing
  astro-lab preprocess --surveys gaia --force
        """,
    )
    add_preprocess_arguments(parser)

    # Only parse args if not already a Namespace
    if not isinstance(args, argparse.Namespace):
        args = parser.parse_args(args)
    logger = setup_logging(getattr(args, "verbose", False))

    if not getattr(args, "surveys", None):
        logger.error("No surveys specified. Use --surveys SURVEY [SURVEY ...]")
        return 1

    try:
        for survey in args.surveys:
            logger.info(f"Preprocessing {survey} data...")

            # Simply trigger dataset creation which handles preprocessing
            from astro_lab.data import create_datamodule
            
            # This will automatically preprocess if needed
            dm = create_datamodule(
                survey=survey,
                task="graph",
                dataset_type="point_cloud",
                k_neighbors=args.k_neighbors,
                force_reload=args.force,
                num_workers=0,  # Single process for preprocessing
            )
            
            # Access dataset to ensure it's processed
            _ = len(dm.train_dataset) if hasattr(dm, 'train_dataset') else 0
            
            logger.info(f"✅ {survey} preprocessing complete!")

        return 0

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
