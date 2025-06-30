#!/usr/bin/env python3
"""
Cosmic Web Analysis CLI
======================

Command-line interface for cosmic web structure analysis.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from astro_lab.data.analysis.cosmic_web import analyze_survey_cosmic_web

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(args: argparse.Namespace) -> int:
    """
    Main cosmic web analysis function.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose)

    logger.info("🌌 Starting Cosmic Web Analysis")
    logger.info(f"   Survey: {args.survey}")
    logger.info(f"   Max samples: {args.max_samples or 'All'}")
    logger.info(f"   Clustering scales: {args.clustering_scales}")

    try:
        # Prepare analysis parameters
        analysis_params = {
            "max_samples": args.max_samples,
            "clustering_scales": args.clustering_scales,
            "min_samples": args.min_samples,
            "include_photometry": True,  # Enable by default
            "include_crossmatch": False,  # Disable by default for performance
        }

        # Add survey-specific parameters
        if args.survey in ["gaia", "exoplanet"]:
            analysis_params["magnitude_limit"] = args.magnitude_limit
        elif args.survey in ["nsa", "sdss", "tng50"]:
            analysis_params["redshift_limit"] = args.redshift_limit

        # Add catalog path if specified
        if args.catalog_path:
            analysis_params["catalog_path"] = args.catalog_path

        # Run comprehensive analysis
        logger.info("🔬 Running comprehensive cosmic web analysis...")
        results = analyze_survey_cosmic_web(survey=args.survey, **analysis_params)

        # Report results
        logger.info("\n📊 Analysis Results:")
        n_objects = results.get("n_objects", 0)
        logger.info(f"   Total objects: {n_objects:,}")

        # Show clustering results
        if "spatial_clustering" in results and results["spatial_clustering"]:
            logger.info("\n🔍 Clustering Results:")
            for scale, stats in results["spatial_clustering"].items():
                logger.info(f"\n   Scale {scale}:")
                logger.info(f"      Clusters: {stats.get('n_clusters', 0)}")
                logger.info(
                    f"      Grouped: {stats.get('n_grouped', 0):,} ({stats.get('grouped_fraction', 0):.1%})"
                )
                logger.info(f"      Isolated: {stats.get('n_noise', 0):,}")

        # Show photometric analysis if available
        if "photometric_analysis" in results and results["photometric_analysis"]:
            photom = results["photometric_analysis"]
            if "colors" in photom:
                logger.info("\n📸 Photometric Analysis:")
                logger.info(
                    f"   Color indices: {len(photom['colors'].get('color_names', []))}"
                )

        # Show graph properties if available
        if "graph_properties" in results and results["graph_properties"]:
            graph_props = results["graph_properties"]
            logger.info("\n🕸️ Graph Properties:")
            logger.info(f"   Mean degree: {graph_props.get('mean_degree', 0):.1f}")
            logger.info(
                f"   Clustering coefficient: {graph_props.get('mean_clustering_coefficient', 0):.3f}"
            )

        # Save results if output directory specified
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save analysis tensor
            results_file = output_dir / f"{args.survey}_cosmic_web_analysis.pt"
            torch.save(results, results_file)
            logger.info(f"\n💾 Results saved to: {results_file}")

        # Create visualizations if requested
        if args.visualize:
            logger.info("\n🎨 Creating visualizations...")
            try:
                from astro_lab.widgets.cosmograph_bridge import CosmographBridge

                # Create visualization data
                viz_data = {
                    "coordinates": results.base_tensors["spatial"]["coordinates"]
                    .cpu()
                    .numpy(),
                    "survey": args.survey,
                    "n_objects": n_objects,
                }

                # Add cluster labels if available
                if "spatial_clustering" in results and results["spatial_clustering"]:
                    first_scale = list(results["spatial_clustering"].keys())[0]
                    cluster_labels = results["spatial_clustering"][first_scale].get(
                        "labels"
                    )
                    if cluster_labels is not None:
                        viz_data["cluster_labels"] = cluster_labels.cpu().numpy()

                # Create cosmograph visualization
                bridge = CosmographBridge()
                bridge.from_cosmic_web_results(viz_data, survey_name=args.survey)

                output_dir = Path(args.output_dir or "cosmic_web_results")
                output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"   Visualizations prepared for: {output_dir}")

            except ImportError:
                logger.warning("   Visualization modules not available - skipping")
            except Exception as e:
                logger.warning(f"   Visualization failed: {e}")

        logger.info("\n✅ Cosmic Web Analysis Complete!")
        return 0

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def create_parser(
    parent_parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create argument parser for cosmic web command."""
    if parent_parser:
        parser = parent_parser.add_subparsers().add_parser(
            "cosmic-web",
            help="Analyze cosmic web structure",
            description="Analyze cosmic web structure in astronomical surveys",
        )
    else:
        parser = argparse.ArgumentParser(
            description="Analyze cosmic web structure in astronomical surveys"
        )

    # Survey selection
    parser.add_argument(
        "survey",
        choices=["gaia", "nsa", "exoplanet", "sdss", "tng50"],
        help="Survey to analyze",
    )

    # Data options
    parser.add_argument(
        "--catalog-path",
        type=Path,
        help="Path to catalog file (uses default if not specified)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of objects to analyze",
    )

    # Clustering options
    parser.add_argument(
        "--clustering-scales",
        nargs="+",
        type=float,
        help="Clustering scales (parsecs for Gaia/exoplanet, Mpc for NSA/SDSS)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples for clustering (default: 5)",
    )

    # Survey-specific options
    parser.add_argument(
        "--magnitude-limit",
        type=float,
        default=12.0,
        help="Magnitude limit for Gaia (default: 12.0)",
    )
    parser.add_argument(
        "--redshift-limit",
        type=float,
        default=0.15,
        help="Redshift limit for NSA/SDSS (default: 0.15)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations",
    )

    # General options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    exit(main(args))
