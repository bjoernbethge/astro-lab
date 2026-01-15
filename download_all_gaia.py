#!/usr/bin/env python3
"""
Download Gaia DR3 partitions (first 100 as test).
This will download approximately 23 GB of data.
"""

import gzip
import logging
import shutil
import time
from pathlib import Path

import pandas as pd
import polars as pl
import requests
from tqdm import tqdm

from astro_lab.config import get_data_paths

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_with_retry(url, file_path, max_retries=3, timeout=300):
    """Download file with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")

            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                # Get file size for progress tracking
                total_size = int(r.headers.get("content-length", 0))

                with open(file_path, "wb") as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress for large files
                            if (
                                total_size > 0 and downloaded % (1024 * 1024) == 0
                            ):  # Every MB
                                progress = (downloaded / total_size) * 100
                                logger.info(
                                    f"Downloaded {downloaded / (1024 * 1024):.1f}MB ({progress:.1f}%)"
                                )

            # Verify file size
            if total_size > 0:
                actual_size = file_path.stat().st_size
                if actual_size != total_size:
                    raise Exception(
                        f"File size mismatch: expected {total_size}, got {actual_size}"
                    )

            logger.info(f"Successfully downloaded {file_path.name}")
            return True

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if file_path.exists():
                file_path.unlink()  # Remove partial file

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed for {url}")
                return False

    return False


def download_gaia_partitions(limit=100):
    """Download and convert Gaia DR3 partitions."""

    # Read partition list
    with open("gaia_partitions.txt", "r") as f:
        partitions = [line.strip() for line in f if line.strip()]

    # Limit for testing
    partitions = partitions[:limit]

    base_url = "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"
    raw_dir = Path(get_data_paths()["raw_dir"]) / "gaia"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Track progress
    total_partitions = len(partitions)
    completed = 0
    failed = []

    logger.info(f"Starting download of {total_partitions} Gaia DR3 partitions...")
    logger.info(f"Estimated size: {total_partitions * 240 / 1024:.1f} GB")

    for partition in tqdm(partitions, desc="Downloading Gaia partitions"):
        try:
            # File paths
            csv_gz_path = raw_dir / partition
            csv_path = raw_dir / partition.replace(".gz", "")
            parquet_path = raw_dir / partition.replace(".csv.gz", ".parquet")

            # Skip if already exists
            if parquet_path.exists():
                completed += 1
                tqdm.write(f"Skipped {partition} (already exists)")
                continue

            # Download with retry logic
            url = base_url + partition
            if not download_with_retry(url, csv_gz_path):
                failed.append(partition)
                continue

            # Extract
            tqdm.write(f"Extracting {partition}...")
            try:
                with gzip.open(csv_gz_path, "rb") as f_in:
                    with open(csv_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                logger.error(f"Failed to extract {partition}: {e}")
                failed.append(partition)
                continue

            # Convert to Parquet
            tqdm.write(f"Converting {partition} to Parquet...")
            try:
                df_pandas = pd.read_csv(csv_path, skiprows=1000, comment="#")
                df = pl.from_pandas(df_pandas)
                df.write_parquet(parquet_path)
            except Exception as e:
                logger.error(f"Failed to convert {partition}: {e}")
                failed.append(partition)
                continue

            # Clean up intermediate files
            csv_path.unlink(missing_ok=True)
            csv_gz_path.unlink(missing_ok=True)

            completed += 1
            tqdm.write(f"Completed {partition} ({completed}/{total_partitions})")

            # Small delay to be nice to the server
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to process {partition}: {e}")
            failed.append(partition)
            continue

    # Summary
    logger.info("Download completed!")
    logger.info(f"Successfully processed: {completed}/{total_partitions}")
    if failed:
        logger.warning(f"Failed partitions: {len(failed)}")
        logger.warning(f"Failed partitions: {failed}")
        with open("failed_partitions.txt", "w") as f:
            for partition in failed:
                f.write(f"{partition}\n")


if __name__ == "__main__":
    download_gaia_partitions(limit=100)  # Start with 100 partitions
