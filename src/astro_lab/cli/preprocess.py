#!/usr/bin/env python3
"""
AstroLab Preprocess CLI
======================

Preprocess raw astronomical data files into training-ready format.
"""

import logging
import sys
from pathlib import Path

from astro_lab.data import preprocess_survey


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def main(args=None) -> int:
    """Main entry point for preprocess command."""
    import argparse

    parser = argparse.ArgumentParser(description="AstroLab Preprocess CLI")
    parser.add_argument(
        "--surveys", nargs="+", required=True, help="Surveys to preprocess"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args() if args is None else args

    logger = setup_logging()
    try:
        for survey in args.surveys:
            logger.info(f"🔄 Preprocessing {survey} ...")

            # Determine output path
            output_path = None
            if args.output_dir:
                output_path = args.output_dir / survey / f"{survey}_processed.parquet"

            preprocess_survey(
                survey, max_samples=args.max_samples, output_path=output_path
            )
            logger.info(f"✅ {survey} preprocessed successfully")

        logger.info("✅ All surveys preprocessed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
