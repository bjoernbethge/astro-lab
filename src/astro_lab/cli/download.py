import argparse
import sys

from astro_lab.data import (
    download_2mass,
    download_gaia,
    download_pan_starrs,
    download_sdss,
    download_survey,
    download_wise,
    list_catalogs,
)


def main():
    parser = argparse.ArgumentParser(description="AstroLab Download CLI")
    parser.add_argument(
        "--survey", help="Survey to download (gaia, sdss, 2mass, wise, pan_starrs)"
    )
    parser.add_argument(
        "--magnitude-limit", type=float, default=12.0, help="Magnitude limit"
    )
    parser.add_argument(
        "--region",
        default="all_sky",
        help="Region to download (all_sky, lmc, smc, etc.)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print(list_catalogs())
    elif args.survey:
        # Use the generic download function
        download_survey(
            survey=args.survey, region=args.region, magnitude_limit=args.magnitude_limit
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
