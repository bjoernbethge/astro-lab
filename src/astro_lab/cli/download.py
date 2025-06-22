#!/usr/bin/env python3
"""
AstroLab Download CLI - SIMPLIFIED

Simplified CLI that refers to the astro_lab.data module.
Most download functions are now implemented directly in the datasets.
"""

import argparse
import sys
from pathlib import Path

from astro_lab.data import (
    AstroDataManager,
    data_manager,
    download_bright_all_sky,
    download_gaia,
    list_catalogs,
)


def main():
    """Main function - shows modern usage."""
    parser = argparse.ArgumentParser(
        description="AstroLab Download CLI - Use astro_lab.data module instead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 MODERNE VERWENDUNG:

# Gaia-Daten herunterladen:
from astro_lab.data import download_bright_all_sky
download_bright_all_sky(magnitude_limit=12.0)

# Exoplanet-Daten (automatisch beim ersten Zugriff):
from astro_lab.data import create_astro_dataloader
loader = create_astro_dataloader("exoplanet")  # Download passiert automatisch!

# All available datasets:
from astro_lab.data import list_catalogs
catalogs = list_catalogs()

💡 Die Download-Funktionen sind jetzt direkt in den Dataset-Klassen implementiert!
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Gaia command (only remaining useful CLI function)
    gaia_parser = subparsers.add_parser("gaia", help="Download Gaia DR3 data")
    gaia_parser.add_argument(
        "--magnitude-limit",
        type=float,
        default=12.0,
        help="Magnitude limit for Gaia stars (default: 12.0)",
    )

    # List command
    subparsers.add_parser("list", help="List available datasets")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "gaia":
        print(f"🌟 Downloading Gaia DR3 data (mag < {args.magnitude_limit})")
        try:
            result = download_bright_all_sky(magnitude_limit=args.magnitude_limit)
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif args.command == "list":
        print("📋 Available datasets:")
        try:
            catalogs = list_catalogs()
            print(catalogs)
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
