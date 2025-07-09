"""
Download CLI for AstroLab - Thin wrapper around data download functions.
"""

import argparse
import logging
import sys


def main(args=None):
    """Main entry point for download command."""
    # If called directly (not from main CLI), create our own parser
    if args is None:
        parser = argparse.ArgumentParser(
            description="Download astronomical survey data from various catalogs"
        )
        parser.add_argument(
            "survey",
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
            ],
            help="Survey to download",
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
        parser.add_argument(
            "--list", action="store_true", help="List available datasets"
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )
        parser.add_argument(
            "--force", "-f", action="store_true", help="Force re-download"
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

    try:
        # Check if we have a survey argument (either positional or --survey)
        survey = getattr(args, "survey", None)
        list_surveys = getattr(args, "list", False)

        if list_surveys:
            logger.info("Available surveys:")
            surveys = [
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
            ]
            for survey in surveys:
                logger.info(f"  - {survey}")
            return 0
        elif survey:
            logger.info(f"Downloading {survey} data...")

            # Import the appropriate collector
            collector_map = {
                "gaia": "astro_lab.data.collectors.gaia.GaiaCollector",
                "sdss": "astro_lab.data.collectors.sdss.SDSSCollector",
                "nsa": "astro_lab.data.collectors.nsa.NSACollector",
                "tng50": "astro_lab.data.collectors.tng50.TNG50Collector",
                "exoplanet": "astro_lab.data.collectors.exoplanet.ExoplanetCollector",
                "twomass": "astro_lab.data.collectors.twomass.TwoMASSCollector",
                "wise": "astro_lab.data.collectors.wise.WISECollector",
                "panstarrs": "astro_lab.data.collectors.panstarrs.PanSTARRSCollector",
                "des": "astro_lab.data.collectors.des.DESCollector",
                "euclid": "astro_lab.data.collectors.euclid.EUCLIDCollector",
            }

            if survey not in collector_map:
                logger.error(f"Survey {survey} not supported")
                return 1

            # Dynamic import
            import importlib

            module_name, class_name = collector_map[survey].rsplit(".", 1)
            module = importlib.import_module(module_name)
            collector_class = getattr(module, class_name)

            # Create collector and download
            collector = collector_class(survey)
            downloaded_files = collector.download(force=args.force)

            logger.info(f"✅ Downloaded {survey} data to:")
            for file_path in downloaded_files:
                logger.info(f"   {file_path}")

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
