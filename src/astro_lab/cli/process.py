#!/usr/bin/env python3
"""
AstroLab Process CLI
===================

Process astronomical data for ML training.
"""

import logging
import sys
from typing import List

from astro_lab.data import create_astro_datamodule


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def main(args) -> int:
    """Main entry point for process command."""
    logger = setup_logging()
    try:
        logger.info(f"🔄 Processing surveys: {args.surveys}")
        logger.info(f"📊 K-neighbors: {args.k_neighbors}")
        if args.max_samples:
            logger.info(f"📦 Max samples: {args.max_samples}")
        for survey in args.surveys:
            logger.info(f"🔄 Processing {survey}...")
            datamodule = create_astro_datamodule(
                survey=survey,
                k_neighbors=args.k_neighbors,
                max_samples=args.max_samples,
            )
            datamodule.setup()
            logger.info(f"✅ {survey} processed successfully")
        logger.info("✅ All surveys processed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1
