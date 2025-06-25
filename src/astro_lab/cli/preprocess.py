"""
Preprocessing CLI for AstroLab - Thin wrapper around data preprocessing.
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    """Main entry point for preprocessing command."""
    parser = argparse.ArgumentParser(
        prog="astro-lab preprocess",
        description="Preprocess astronomical catalogs for machine learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options
    parser.add_argument("--input", "-i", type=Path, help="Path to input catalog file")
    parser.add_argument(
        "--survey",
        "-s",
        required=True,
        choices=["gaia", "sdss", "nsa", "linear", "exoplanet", "tng50"],
        help="Survey type",
    )
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to process")
    parser.add_argument(
        "--write-graph", action="store_true", help="Create graph representation"
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=8, help="Number of neighbors for graph"
    )
    parser.add_argument(
        "--distance-threshold", type=float, default=50.0, help="Distance threshold"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # Import only what we need
    from astro_lab.data import preprocess_catalog, process_survey

    try:
        if args.input:
            # Process single file
            processed_df = preprocess_catalog(
                input_path=args.input,
                survey_type=args.survey,
                max_samples=args.max_samples,
                output_dir=args.output,
                write_graph=args.write_graph,
                k_neighbors=args.k_neighbors,
                distance_threshold=args.distance_threshold,
            )
            logger.info(f"Preprocessed {len(processed_df):,} objects")
            return 0
        else:
            # Auto-process survey
            files = process_survey(
                survey=args.survey,
                max_samples=args.max_samples,
                k_neighbors=args.k_neighbors,
                force=False,
            )
            logger.info(f"Processed {args.survey} survey")
            for file_type, path in files.items():
                logger.info(f"   {file_type}: {path}")
            return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.error("Preprocessing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
