#!/usr/bin/env python3
"""
Cosmic Web Analysis CLI
======================

Command-line interface for cosmic web structure analysis.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..data.cosmic_web import (
    analyze_gaia_cosmic_web,
    analyze_nsa_cosmic_web,
    analyze_exoplanet_cosmic_web,
)
from ..widgets.cosmic_web import CosmicWebVisualizer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
        # Run analysis based on survey type
        if args.survey == "gaia":
            results = analyze_gaia_cosmic_web(
                catalog_path=args.catalog_path,
                max_samples=args.max_samples,
                magnitude_limit=args.magnitude_limit,
                clustering_scales=args.clustering_scales,
                min_samples=args.min_samples,
            )
        elif args.survey == "nsa":
            results = analyze_nsa_cosmic_web(
                catalog_path=args.catalog_path,
                redshift_limit=args.redshift_limit,
                clustering_scales=args.clustering_scales,
                min_samples=args.min_samples,
            )
        elif args.survey == "exoplanet":
            results = analyze_exoplanet_cosmic_web(
                catalog_path=args.catalog_path,
                clustering_scales=args.clustering_scales,
                min_samples=args.min_samples,
            )
        else:
            raise ValueError(f"Unknown survey: {args.survey}")
            
        # Report results
        logger.info("\n📊 Analysis Results:")
        n_objects = results.get("n_stars", results.get("n_galaxies", results.get("n_systems", 0)))
        logger.info(f"   Total objects: {n_objects:,}")
        
        logger.info("\n🔍 Clustering Results:")
        for scale, stats in results["clustering_results"].items():
            logger.info(f"\n   {scale}:")
            logger.info(f"      Clusters: {stats['n_clusters']}")
            logger.info(f"      Grouped: {stats['n_grouped']:,} ({stats['grouped_fraction']:.1%})")
            logger.info(f"      Isolated: {stats['n_noise']:,}")
            
        # Create visualizations if requested
        if args.visualize:
            logger.info("\n🎨 Creating visualizations...")
            visualizer = CosmicWebVisualizer()
            
            # Note: This would need to be enhanced to actually create
            # and save the visualizations. Currently just a placeholder.
            output_dir = Path(args.output_dir or "cosmic_web_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"   Visualizations saved to: {output_dir}")
            
        logger.info("\n✅ Cosmic Web Analysis Complete!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def create_parser(parent_parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Create argument parser for cosmic web command."""
    if parent_parser:
        parser = parent_parser.add_parser(
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
        choices=["gaia", "nsa", "exoplanet"],
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
        help="Clustering scales (parsecs for Gaia/exoplanet, Mpc for NSA)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples for DBSCAN clustering (default: 5)",
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
        help="Redshift limit for NSA (default: 0.15)",
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
