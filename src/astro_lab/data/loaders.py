"""
AstroLab Data Loaders - Unified loading and downloading functions.

This module consolidates all data loading and downloading functionality
that was previously scattered across manager.py, utils.py, and preprocessing.py.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import torch
from astropy.io import fits
from astropy.table import Table

from .config import data_config
from .utils import load_fits_optimized, load_fits_table_optimized

logger = logging.getLogger(__name__)


def load_catalog(catalog_path: Union[str, Path]) -> pl.DataFrame:
    """Load a catalog from various formats (FITS, Parquet, CSV)."""
    path = Path(catalog_path)

    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    logger.info(f"Loading catalog from {path}")

    if path.suffix.lower() == ".fits":
        return load_fits_optimized(path)
    elif path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _find_survey_data_dir(survey: str) -> Path:
    """Find data directory for a survey."""
    # Try raw directory first
    raw_dir = data_config.raw_dir / survey
    if raw_dir.exists():
        return raw_dir

    # Try processed directory
    processed_dir = data_config.processed_dir / survey
    if processed_dir.exists():
        return processed_dir

    # Create raw directory if nothing exists
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def load_survey_catalog(
    survey: str,
    max_samples: Optional[int] = None,
    load_processed: bool = True,
    **kwargs,
) -> pl.DataFrame:
    """
    Load catalog data for a specific survey.

    Parameters
    ----------
    survey : str
        Survey name (gaia, sdss, 2mass, wise, pan_starrs)
    max_samples : int, optional
        Maximum number of samples to load
    load_processed : bool
        Whether to load processed version if available

    Returns
    -------
    pl.DataFrame
        Survey catalog data
    """
    data_dir = _find_survey_data_dir(survey)
    catalog_files = list(data_dir.glob(f"{survey}*.fits")) + list(
        data_dir.glob(f"{survey}*.parquet")
    )

    if not catalog_files:
        raise FileNotFoundError(
            f"No catalog files found for survey '{survey}' in {data_dir}"
        )

    # Use the first available file
    catalog_path = catalog_files[0]
    df = load_catalog(catalog_path)

    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, seed=42)
        logger.info(f"Sampled {max_samples} objects from {len(df)} total")

    return df


def download_survey(survey: str, **kwargs) -> Path:
    """Download data for a specific survey."""
    if survey == "gaia":
        return download_gaia(**kwargs)
    elif survey == "sdss":
        return download_sdss(**kwargs)
    elif survey == "2mass":
        return download_2mass(**kwargs)
    elif survey == "wise":
        return download_wise(**kwargs)
    elif survey == "pan_starrs":
        return download_pan_starrs(**kwargs)
    else:
        raise ValueError(f"Unsupported survey: {survey}")


def download_gaia(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download GAIA catalog."""
    logger.info(f"Downloading GAIA data for region: {region}")
    output_path = data_config.raw_dir / "gaia" / f"gaia_{region}_{magnitude_limit}.fits"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def download_sdss(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download SDSS catalog."""
    logger.info(f"Downloading SDSS data for region: {region}")
    output_path = data_config.raw_dir / "sdss" / f"sdss_{region}_{magnitude_limit}.fits"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def download_2mass(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download 2MASS catalog."""
    logger.info(f"Downloading 2MASS data for region: {region}")
    output_path = (
        data_config.raw_dir / "2mass" / f"2mass_{region}_{magnitude_limit}.fits"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def download_wise(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download WISE catalog."""
    logger.info(f"Downloading WISE data for region: {region}")
    output_path = data_config.raw_dir / "wise" / f"wise_{region}_{magnitude_limit}.fits"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def download_pan_starrs(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download Pan-STARRS catalog."""
    logger.info(f"Downloading Pan-STARRS data for region: {region}")
    output_path = (
        data_config.raw_dir
        / "pan_starrs"
        / f"pan_starrs_{region}_{magnitude_limit}.fits"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def import_fits(fits_file: Union[str, Path], catalog_name: str) -> Path:
    """Import FITS file to processed data directory."""
    fits_path = Path(fits_file)

    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    output_path = data_config.processed_dir / f"{catalog_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and convert to Parquet
    df = load_fits_optimized(fits_path)
    df.write_parquet(output_path)

    logger.info(f"Imported {fits_path} -> {output_path}")
    return output_path


def import_tng50(hdf5_file: Union[str, Path], dataset_name: str = "PartType0") -> Path:
    """Import TNG50 HDF5 file."""
    import h5py

    hdf5_path = Path(hdf5_file)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    output_path = data_config.processed_dir / "tng50" / f"{dataset_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Imported {hdf5_path} -> {output_path}")
    return output_path


def list_available_catalogs(survey: Optional[str] = None) -> pl.DataFrame:
    """List all available catalogs."""
    catalogs = []

    search_dirs = [data_config.raw_dir, data_config.processed_dir]
    if survey:
        search_dirs = [d / survey for d in search_dirs if (d / survey).exists()]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for file_path in search_dir.rglob("*.fits"):
            catalogs.append(
                {
                    "name": file_path.stem,
                    "path": str(file_path),
                    "format": "fits",
                    "size_mb": file_path.stat().st_size / 1024 / 1024,
                    "survey": file_path.parent.name,
                }
            )

        for file_path in search_dir.rglob("*.parquet"):
            catalogs.append(
                {
                    "name": file_path.stem,
                    "path": str(file_path),
                    "format": "parquet",
                    "size_mb": file_path.stat().st_size / 1024 / 1024,
                    "survey": file_path.parent.name,
                }
            )

    return pl.DataFrame(catalogs)
