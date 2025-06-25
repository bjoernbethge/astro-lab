"""
Preprocessing CLI for astro-lab.
"""

import argparse
import sys
from pathlib import Path

from astro_lab.data.manager import AstroDataManager
from astro_lab.data.preprocessing import preprocess_catalog
from astro_lab.utils.config.loader import ConfigLoader


def main():
    """Main preprocessing CLI function."""
    parser = argparse.ArgumentParser(
        description="Preprocess astronomical catalogs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, help="Path to preprocessing configuration file"
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Input catalog file path"
    )

    parser.add_argument(
        "--output", type=str, help="Output directory for processed data"
    )

    parser.add_argument(
        "--survey", type=str, help="Survey type (gaia, sdss, nsa, etc.)"
    )

    parser.add_argument(
        "--magnitude-limit", type=float, help="Magnitude limit for filtering"
    )

    parser.add_argument(
        "--region",
        type=str,
        help="Region filter (e.g., 'ra_min,ra_max,dec_min,dec_max')",
    )

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)

    # Initialize data manager
    manager = AstroDataManager()

    # Determine output directory
    output_dir = args.output or manager.processed_dir

    # Process the catalog
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file {args.input} does not exist")
            sys.exit(1)

        # Load and preprocess catalog
        processed_df = preprocess_catalog(
            input_path,
            survey=args.survey,
            magnitude_limit=args.magnitude_limit,
            region=args.region,
            config=config,
        )

        # Save processed data
        output_path = Path(output_dir) / f"{input_path.stem}_processed.parquet"
        processed_df.write_parquet(output_path)

        print("Preprocessing completed successfully!")
        print(f"Processed data saved to: {output_path}")
        print(f"Input rows: {len(processed_df)}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
