"""
Download CLI for AstroLab - Thin wrapper around data download functions.
"""

import argparse
import logging
import sys


def main():
    """Main entry point for download command."""
    parser = argparse.ArgumentParser(
        description="Download astronomical survey data from various catalogs"
    )
    parser.add_argument(
        "--survey", help="Survey to download (gaia, sdss, 2mass, wise, pan_starrs)"
    )
    parser.add_argument(
        "--magnitude-limit",
        type=float,
        default=12.0,
        help="Magnitude limit (default: 12.0)",
    )
    parser.add_argument(
        "--region",
        default="all_sky",
        help="Region to download (all_sky, lmc, smc, etc.)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
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
    from astro_lab.data import download_survey, list_catalogs

    try:
        if args.list:
            logger.info(list_catalogs())
            return 0
        elif args.survey:
            download_survey(
                survey=args.survey,
                region=args.region,
                magnitude_limit=args.magnitude_limit,
            )
            return 0
        else:
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        logger.error("Download interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
