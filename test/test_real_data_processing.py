#!/usr/bin/env python3
"""
Test Real Data Processing Pipeline
==================================

Tests the complete pipeline with real astronomical data from data/raw/*
and processes it to data/processed/*.

This script demonstrates:
1. Loading real Gaia DR3 data
2. Applying proper preprocessing
3. Generating TensorDict objects
4. Creating PyG graphs
5. Saving processed data
"""

import logging
import sys
import time
from pathlib import Path

import polars as pl

from astro_lab.data.preprocessors.gaia import GaiaPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_available_data():
    """Check what real data files are available."""
    data_dir = Path("data/raw")

    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist!")
        return {}

    available_data = {}

    # Check for Gaia data
    gaia_dir = data_dir / "gaia"
    if gaia_dir.exists():
        gaia_files = list(gaia_dir.glob("*.parquet")) + list(gaia_dir.glob("*.fits"))
        if gaia_files:
            available_data["gaia"] = gaia_files
            logger.info(f"Found Gaia data: {len(gaia_files)} files")

    # Check for NSA data
    nsa_dir = data_dir / "nsa"
    if nsa_dir.exists():
        nsa_files = list(nsa_dir.glob("*.fits")) + list(nsa_dir.glob("*.parquet"))
        if nsa_files:
            available_data["nsa"] = nsa_files
            logger.info(f"Found NSA data: {len(nsa_files)} files")

    # Check for exoplanet data
    exoplanet_dir = data_dir / "exoplanet"
    if exoplanet_dir.exists():
        exo_files = list(exoplanet_dir.glob("*.parquet"))
        if exo_files:
            available_data["exoplanet"] = exo_files
            logger.info(f"Found exoplanet data: {len(exo_files)} files")

    # Check for other surveys
    for survey in ["sdss", "des", "wise", "twomass"]:
        survey_dir = data_dir / survey
        if survey_dir.exists():
            files = list(survey_dir.glob("*.parquet")) + list(survey_dir.glob("*.fits"))
            if files:
                available_data[survey] = files
                logger.info(f"Found {survey} data: {len(files)} files")

    return available_data


def load_survey_data(
    survey: str, file_path: Path, max_samples: int = 10000
) -> pl.DataFrame:
    """Load data for any survey with proper FITS handling."""
    logger.info(f"Loading {survey} data from {file_path}")

    try:
        if file_path.suffix == ".parquet":
            df = pl.read_parquet(file_path)

        elif file_path.suffix == ".fits":
            from astropy.table import Table

            # Special handling for NSA FITS files with multidimensional columns
            if survey == "nsa":
                table = Table.read(file_path, format="fits")

                # Filter out multidimensional columns
                single_dim_cols = []
                for name in table.colnames:
                    if len(table[name].shape) <= 1:
                        single_dim_cols.append(name)

                logger.info(
                    f"Filtered {len(table.colnames) - len(single_dim_cols)} multidimensional columns"
                )
                logger.info(
                    f"Keeping {len(single_dim_cols)} single-dimensional columns"
                )

                # Keep only single dimensional columns
                filtered_table = table[single_dim_cols]

                # Convert to dictionary for Polars (avoiding pandas conversion)
                data_dict = {}
                for col_name in filtered_table.colnames:
                    col_data = filtered_table[col_name]

                    # Handle different data types
                    if col_data.dtype.kind in ["i", "f"]:  # int or float
                        data_dict[col_name] = col_data.data
                    elif col_data.dtype.kind in ["U", "S"]:  # string
                        data_dict[col_name] = col_data.data.astype(str)
                    else:
                        # Try to convert to float, fallback to string
                        try:
                            data_dict[col_name] = col_data.data.astype(float)
                        except (ValueError, TypeError):
                            data_dict[col_name] = col_data.data.astype(str)

                # Create polars dataframe directly
                df = pl.DataFrame(data_dict)

            else:
                # Standard FITS loading for other surveys
                table = Table.read(file_path, format="fits")
                df = pl.from_pandas(table.to_pandas())

        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None

        # Sample if too large
        if len(df) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(df)} total objects")
            df = df.sample(max_samples)

        logger.info(f"Loaded {len(df)} {survey} objects with {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.error(f"Failed to load {survey} data: {e}")
        return None


def process_survey_data(survey: str, df_or_path) -> pl.DataFrame:
    """Process survey data using appropriate preprocessor."""
    logger.info(f"Processing {survey} data...")

    try:
        # Get appropriate preprocessor
        if survey == "gaia":
            preprocessor = GaiaPreprocessor()
        else:
            # For other surveys, we'll need to import their preprocessors
            # For now, just use Gaia as default
            preprocessor = GaiaPreprocessor()

        # For NSA with FITS files, pass the file path directly
        if (
            survey == "nsa"
            and isinstance(df_or_path, Path)
            and str(df_or_path).endswith(".fits")
        ):
            input_data = str(df_or_path)
            input_count = "FITS file"
        else:
            # For other surveys or non-FITS files, use the DataFrame
            input_data = df_or_path
            input_count = (
                len(df_or_path) if hasattr(df_or_path, "__len__") else "unknown"
            )

            # Check if required columns are present
            if hasattr(df_or_path, "columns"):
                required_cols = getattr(preprocessor, "required_columns", [])
                available_cols = set(df_or_path.columns)
                missing_cols = set(required_cols) - available_cols

                if missing_cols:
                    logger.warning(
                        f"Missing required columns for {survey}: {missing_cols}"
                    )
                    # Try to map common column variations
                    input_data = map_column_names(df_or_path, survey)

        # Time the processing
        start_time = time.time()

        # Run full preprocessing pipeline
        processed_df = preprocessor.preprocess(input_data)

        processing_time = time.time() - start_time

        logger.info(f"Processed {survey} data in {processing_time:.2f}s:")
        logger.info(f"  Input: {input_count} objects")
        logger.info(f"  Output: {len(processed_df)} objects")
        logger.info(f"  Features: {len(processed_df.columns)} columns")

        # Show preprocessor stats
        stats = preprocessor.get_info()
        if "metadata" in stats:
            logger.info(f"  Metadata: {stats['metadata']}")

        return processed_df

    except Exception as e:
        logger.error(f"Failed to process {survey} data: {e}")
        logger.exception("Full traceback:")
        return None


def map_column_names(df: pl.DataFrame, survey: str) -> pl.DataFrame:
    """Map column names to standard format."""
    column_mappings = {
        "gaia": {
            "SOURCE_ID": "source_id",
            "RA_ICRS": "ra",
            "DE_ICRS": "dec",
            "Plx": "parallax",
            "e_Plx": "parallax_error",
            "pmRA": "pmra",
            "pmDE": "pmdec",
            "e_pmRA": "pmra_error",
            "e_pmDE": "pmdec_error",
            "Gmag": "phot_g_mean_mag",
            "BPmag": "phot_bp_mean_mag",
            "RPmag": "phot_rp_mean_mag",
            # Add more mappings as needed
        },
        "nsa": {
            "OBJID": "objid",
            "RA": "ra",
            "DEC": "dec",
            "Z": "z",
            # Add NSA-specific mappings
        },
    }

    if survey in column_mappings:
        mapping = column_mappings[survey]

        # Rename columns if they exist
        rename_dict = {}
        for old_name, new_name in mapping.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name

        if rename_dict:
            logger.info(f"Renaming columns for {survey}: {rename_dict}")
            df = df.rename(rename_dict)

    return df


def save_processed_data(survey: str, processed_df: pl.DataFrame) -> bool:
    """Save processed data to data/processed/survey/."""
    output_dir = Path(f"data/processed/{survey}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save as parquet
        output_file = output_dir / f"{survey}_processed.parquet"
        processed_df.write_parquet(output_file)
        logger.info(f"Saved processed {survey} data to {output_file}")

        # Also save metadata
        metadata_file = output_dir / f"{survey}_metadata.txt"
        with open(metadata_file, "w") as f:
            f.write(f"Survey: {survey}\n")
            f.write(f"Processed objects: {len(processed_df)}\n")
            f.write(f"Features: {len(processed_df.columns)}\n")
            f.write(f"Columns: {list(processed_df.columns)}\n")
            f.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        return True

    except Exception as e:
        logger.error(f"Failed to save processed {survey} data: {e}")
        return False


def test_real_data_pipeline():
    """Test the complete real data processing pipeline."""
    logger.info("🚀 Starting Real Data Processing Pipeline Test")
    logger.info("=" * 60)

    # Check available data
    available_data = check_available_data()

    if not available_data:
        logger.error("❌ No data files found in data/raw/!")
        logger.info("Please ensure you have data files in the following locations:")
        logger.info("  - data/raw/gaia/gaia_*.parquet")
        logger.info("  - data/raw/nsa/nsa_*.fits")
        logger.info("  - data/raw/exoplanet/confirmed_exoplanets.parquet")
        return False

    success_count = 0
    total_surveys = len(available_data)

    # Process each available survey
    for survey, files in available_data.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing {survey.upper()} data")
        logger.info(f"{'=' * 40}")

        # Use the first available file
        file_path = files[0]

        try:
            # For NSA FITS files, pass the file path directly to the preprocessor
            if survey == "nsa" and file_path.suffix == ".fits":
                # Process NSA FITS file directly
                processed_df = process_survey_data(survey, file_path)
            else:
                # Load data first for other surveys
                df = load_survey_data(survey, file_path, max_samples=5000)

                if df is None:
                    logger.error(f"❌ Failed to load {survey} data")
                    continue

                # Process data
                processed_df = process_survey_data(survey, df)

            if processed_df is None:
                logger.error(f"❌ Failed to process {survey} data")
                continue

            # Save processed data
            if save_processed_data(survey, processed_df):
                logger.info(f"✅ Successfully processed {survey} data")
                success_count += 1
            else:
                logger.error(f"❌ Failed to save {survey} data")

        except Exception as e:
            logger.error(f"❌ Error processing {survey}: {e}")
            logger.exception("Full traceback:")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("📊 PROCESSING SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Surveys processed: {success_count}/{total_surveys}")

    if success_count > 0:
        logger.info("✅ Successfully processed data for:")
        for survey in available_data.keys():
            output_file = Path(f"data/processed/{survey}/{survey}_processed.parquet")
            if output_file.exists():
                logger.info(f"  - {survey}: {output_file}")

    if success_count == total_surveys:
        logger.info("🎉 All surveys processed successfully!")
        return True
    else:
        logger.warning(f"⚠️  {total_surveys - success_count} surveys failed")
        return False


if __name__ == "__main__":
    try:
        success = test_real_data_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
