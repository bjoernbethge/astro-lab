#!/usr/bin/env python3
"""
🌌 Modern Astronomical Data Analysis Example

Demonstrates the latest AstroLab data processing capabilities including:
- Multi-survey cosmic web analysis
- Interactive 3D visualization
- Advanced data loading and processing
- Survey comparison and statistics

This example replaces the old individual survey scripts with a unified approach.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger("astro_lab_examples")
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


def demonstrate_single_survey_analysis(logger: Optional[logging.Logger] = None):
    """Demonstrate analysis of a single survey."""
    if logger is None:
        logger = logging.getLogger("astro_lab_examples")

    logger.info("🌟 Single Survey Analysis Example")
    logger.info("=" * 50)

    # Analyze Gaia stellar data
    logger.info("📊 Loading and analyzing Gaia DR3 stellar data...")

    start_time = time.time()
    results = create_cosmic_web_loader(
        survey="gaia", max_samples=1000, scales_mpc=[1.0, 2.0, 5.0, 10.0]
    )
    processing_time = time.time() - start_time

    logger.info(f"✅ Analysis completed in {processing_time:.1f}s")
    logger.info(f"📊 Found {results['n_objects']:,} stellar objects")
    logger.info(f"📏 Volume: {results['total_volume']:.0f} Mpc³")
    logger.info(f"🌍 Global density: {results['global_density']:.2e} obj/Mpc³")

    # Multi-scale clustering results
    logger.info("🔗 Multi-scale clustering results:")
    for scale, result in results["results_by_scale"].items():
        logger.info(
            f"  {scale:5.0f} Mpc: {result['n_clusters']:4d} groups "
            f"({result['grouped_fraction'] * 100:5.1f}% grouped)"
        )

    return results


def demonstrate_multi_survey_comparison(logger: Optional[logging.Logger] = None):
    """Demonstrate comparison of multiple surveys."""
    if logger is None:
        logger = logging.getLogger("astro_lab_examples")

    logger.info("🌌 Multi-Survey Comparison Example")
    logger.info("=" * 50)

    # Define surveys to compare
    surveys = ["gaia", "nsa", "exoplanet"]
    survey_names = {
        "gaia": "Gaia DR3 (Stars)",
        "nsa": "NSA (Galaxies)",
        "exoplanet": "Exoplanet Archive",
    }

    results = {}

    for survey in surveys:
        logger.info(f"📊 Processing {survey_names[survey]}...")

        try:
            survey_results = create_cosmic_web_loader(
                survey=survey,
                max_samples=500,  # Smaller sample for comparison
                scales_mpc=[5.0, 10.0, 20.0],
            )
            results[survey] = survey_results
            logger.info(f"✅ {survey_results['n_objects']:,} objects processed")

        except Exception as e:
            logger.error(f"❌ Failed to process {survey}: {e}")
            continue

    # Compare results
    if results:
        logger.info("📈 Survey Comparison Summary:")
        logger.info(
            f"{'Survey':<15} {'Objects':<10} {'Volume (Mpc³)':<15} {'Density (obj/Mpc³)':<20}"
        )
        logger.info("-" * 70)

        for survey, result in results.items():
            density = result["global_density"]
            logger.info(
                f"{survey_names[survey]:<15} {result['n_objects']:<10,} "
                f"{result['total_volume']:<15.0f} {density:<20.2e}"
            )

    return results


def demonstrate_interactive_visualization(logger: Optional[logging.Logger] = None):
    """Demonstrate interactive 3D visualization."""
    if logger is None:
        logger = logging.getLogger("astro_lab_examples")

    logger.info("🎨 Interactive Visualization Example")
    logger.info("=" * 50)

    try:
        # Load Gaia data for visualization
        logger.info("📊 Loading Gaia data for visualization...")
        results = create_cosmic_web_loader(
            survey="gaia",
            max_samples=200,  # Smaller sample for smooth visualization
            scales_mpc=[5.0, 10.0],
        )

        # Create interactive visualization
        logger.info("🎨 Creating interactive 3D visualization...")
        bridge = CosmographBridge()
        widget = bridge.from_cosmic_web_results(
            results,
            survey_name="gaia",
            radius=3.0,
            background_color="#000011",
            point_color="#ffd700",
            physics_enabled=True,
            gravity_strength=0.1,
        )

        logger.info("✅ Interactive visualization created!")
        logger.info(
            "💡 Use mouse to navigate: click and drag to rotate, scroll to zoom"
        )

        return widget

    except Exception as e:
        logger.warning(f"⚠️ Visualization failed: {e}")
        return None


def demonstrate_advanced_analysis(logger: Optional[logging.Logger] = None):
    """Demonstrate advanced analysis features."""
    if logger is None:
        logger = logging.getLogger("astro_lab_examples")

    logger.info("🔬 Advanced Analysis Example")
    logger.info("=" * 50)

    # Load multiple surveys for advanced analysis
    surveys = ["gaia", "nsa"]
    all_results = {}

    for survey in surveys:
        logger.info(f"📊 Loading {survey} data...")
        try:
            results = create_cosmic_web_loader(
                survey=survey, max_samples=300, scales_mpc=[5.0, 10.0, 20.0]
            )
            all_results[survey] = results
        except Exception as e:
            logger.error(f"❌ Failed to load {survey}: {e}")
            continue

    if len(all_results) >= 2:
        # Cross-survey analysis
        logger.info("🔗 Cross-Survey Analysis:")

        # Compare clustering efficiency
        for scale in [5.0, 10.0, 20.0]:
            logger.info(f"📏 Scale: {scale} Mpc")
            for survey, results in all_results.items():
                if scale in results["results_by_scale"]:
                    scale_result = results["results_by_scale"][scale]
                    efficiency = scale_result["grouped_fraction"] * 100
                    logger.info(
                        f"  {survey:8s}: {efficiency:5.1f}% grouped, "
                        f"{scale_result['n_clusters']:3d} clusters"
                    )

        # Density comparison
        logger.info("📊 Density Comparison:")
        for survey, results in all_results.items():
            density = results["global_density"]
            logger.info(f"  {survey:8s}: {density:.2e} obj/Mpc³")

    return all_results


def demonstrate_data_export(logger: Optional[logging.Logger] = None):
    """Demonstrate data export and saving capabilities."""
    if logger is None:
        logger = logging.getLogger("astro_lab_examples")

    logger.info("💾 Data Export Example")
    logger.info("=" * 50)

    # Process survey and save results
    logger.info("📊 Processing Gaia data for export...")
    results = create_cosmic_web_loader(
        survey="gaia", max_samples=100, scales_mpc=[5.0, 10.0]
    )

    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save coordinates
    coords_tensor = torch.tensor(results["coordinates"])
    torch.save(coords_tensor, output_dir / "gaia_coordinates.pt")

    # Save summary statistics
    summary_file = output_dir / "gaia_analysis_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Gaia DR3 Cosmic Web Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total objects: {results['n_objects']:,}\n")
        f.write(f"Total volume: {results['total_volume']:.0f} Mpc³\n")
        f.write(f"Global density: {results['global_density']:.2e} obj/Mpc³\n\n")

        f.write("Multi-scale clustering results:\n")
        for scale, result in results["results_by_scale"].items():
            f.write(
                f"  {scale} Mpc: {result['n_clusters']} groups, "
                f"{result['grouped_fraction'] * 100:.1f}% grouped\n"
            )

    logger.info(f"✅ Results saved to: {output_dir}")
    logger.info(f"  📁 Coordinates: {output_dir / 'gaia_coordinates.pt'}")
    logger.info(f"  📄 Summary: {output_dir / 'gaia_analysis_summary.txt'}")


def main():
    """Main function demonstrating all features."""
    # Setup logging
    logger = setup_logging()

    logger.info("🌌 AstroLab Modern Data Analysis Examples")
    logger.info("=" * 60)
    logger.info("This example demonstrates the latest AstroLab capabilities:")
    logger.info("• Single survey cosmic web analysis")
    logger.info("• Multi-survey comparison")
    logger.info("• Interactive 3D visualization")
    logger.info("• Advanced cross-survey analysis")
    logger.info("• Data export and saving")
    logger.info("")

    try:
        # 1. Single survey analysis
        gaia_results = demonstrate_single_survey_analysis(logger)

        # 2. Multi-survey comparison
        comparison_results = demonstrate_multi_survey_comparison(logger)

        # 3. Interactive visualization
        widget = demonstrate_interactive_visualization(logger)

        # 4. Advanced analysis
        advanced_results = demonstrate_advanced_analysis(logger)

        # 5. Data export
        demonstrate_data_export(logger)

        logger.info("🎉 All examples completed successfully!")
        logger.info("💡 Check the 'examples/output' directory for saved results")

        if widget:
            logger.info(
                "🎨 Interactive visualization is available in the widget variable"
            )

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
