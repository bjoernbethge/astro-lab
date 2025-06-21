#!/usr/bin/env python3
"""
🚀 AstroLab Quick Start Example

Simple example to get started with AstroLab data processing.
Perfect for beginners and quick exploration.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import create_cosmic_web_loader


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("astro_lab_quick_start")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def main():
    """Quick start example - analyze Gaia stellar data."""
    # Setup logging
    logger = setup_logging()

    logger.info("🚀 AstroLab Quick Start")
    logger.info("=" * 30)

    # Simple Gaia analysis
    logger.info("📊 Loading Gaia stellar data...")

    results = create_cosmic_web_loader(
        survey="gaia",
        max_samples=100,  # Small sample for quick start
        scales_mpc=[5.0, 10.0],
    )

    # Print results
    logger.info("✅ Analysis complete!")
    logger.info(f"📊 Found {results['n_objects']} stellar objects")
    logger.info(f"📏 Volume: {results['total_volume']:.0f} Mpc³")
    logger.info(f"🌍 Density: {results['global_density']:.2e} obj/Mpc³")

    # Show clustering results
    logger.info("🔗 Clustering results:")
    for scale, result in results["results_by_scale"].items():
        groups = result["n_clusters"]
        grouped = result["grouped_fraction"] * 100
        logger.info(f"  {scale} Mpc: {groups} groups ({grouped:.1f}% grouped)")

    logger.info("🎉 Quick start completed!")
    logger.info("💡 Try running with different surveys: nsa, exoplanet, linear")


if __name__ == "__main__":
    main()
