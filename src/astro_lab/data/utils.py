"""
AstroLab Data Utils - Essential utility functions.

Cleaned up version with only non-redundant utility functions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from astropy.io import fits
from astropy.table import Table

logger = logging.getLogger(__name__)


def check_astroquery_available() -> bool:
    """Check if astroquery is available."""
    try:
        import astroquery

        return True
    except ImportError:
        return False


def get_data_statistics(df: pl.DataFrame) -> Dict[str, Any]:
    """Get comprehensive statistics for a DataFrame."""
    stats = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "memory_usage_mb": df.estimated_size() / (1024 * 1024),
        "columns": df.columns,
        "dtypes": {col: str(dtype) for col, dtype in df.schema.items()},
    }

    # Detect magnitude columns
    mag_cols = _detect_magnitude_columns(df)
    if mag_cols:
        stats["magnitude_columns"] = mag_cols
        for col in mag_cols:
            if col in df.columns:
                col_stats = df[col].describe()
                stats[f"{col}_stats"] = {
                    "mean": col_stats["mean"],
                    "std": col_stats["std"],
                    "min": col_stats["min"],
                    "max": col_stats["max"],
                    "null_count": df[col].null_count(),
                }

    # Detect coordinate columns
    coord_cols = [
        col for col in df.columns if col.lower() in ["ra", "dec", "x", "y", "z"]
    ]
    if coord_cols:
        stats["coordinate_columns"] = coord_cols

    return stats


def _detect_magnitude_columns(df: pl.DataFrame) -> List[str]:
    """Detect magnitude columns in DataFrame."""
    mag_patterns = [
        "mag",
        "magnitude",
        "phot",
        "psf",
        "model",
        "petro",
        "fiber",
        "aper",
    ]

    mag_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in mag_patterns):
            mag_cols.append(col)

    return mag_cols


def load_fits_optimized(
    fits_path: Union[str, Path],
    hdu_index: int = 0,
    memmap: bool = True,
    max_memory_mb: float = 1000.0,
) -> pl.DataFrame:
    """Load FITS file optimized for Polars DataFrame output."""
    try:
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        # Check file size
        file_size_mb = fits_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_memory_mb:
            logger.warning(f"Large FITS file: {file_size_mb:.1f} MB")

        # Load FITS file
        with fits.open(fits_path, memmap=memmap) as hdul:
            hdu = hdul[hdu_index]

            if not hasattr(hdu, "data") or hdu.data is None:
                raise ValueError(f"No data in HDU {hdu_index}")

            # Convert to astropy Table first
            table = Table(hdu.data)

            # Filter out multidimensional columns
            names = [
                name
                for name in table.colnames
                if hasattr(table[name], "shape") and len(table[name].shape) <= 1
            ]

            if len(names) < len(table.colnames):
                filtered_cols = set(table.colnames) - set(names)
                logger.info(
                    f"Filtered out {len(filtered_cols)} multidimensional columns"
                )

            filtered_table = table[names]
            df_pandas = filtered_table.to_pandas()
            return pl.from_pandas(df_pandas)

    except Exception as e:
        logger.error(f"Error loading FITS file: {e}")
        raise


def load_fits_table_optimized(
    fits_path: Union[str, Path],
    hdu_index: int = 1,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> pl.DataFrame:
    """Load FITS table with optimization."""
    df = load_fits_optimized(fits_path, hdu_index)

    if columns:
        available_cols = [col for col in columns if col in df.columns]
        if available_cols:
            df = df.select(available_cols)

    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    return df


def get_fits_info(fits_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about FITS file structure."""
    fits_path = Path(fits_path)

    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    with fits.open(fits_path) as hdul:
        info = {
            "filename": fits_path.name,
            "file_size_mb": fits_path.stat().st_size / (1024 * 1024),
            "n_hdus": len(hdul),
            "hdus": [],
        }

        for i, hdu in enumerate(hdul):
            hdu_info = {
                "index": i,
                "type": type(hdu).__name__,
                "name": hdu.name,
            }

            if hasattr(hdu, "data") and hdu.data is not None:
                if hasattr(hdu.data, "shape"):
                    hdu_info["shape"] = hdu.data.shape
                if hasattr(hdu.data, "dtype"):
                    hdu_info["dtype"] = str(hdu.data.dtype)
                if hasattr(hdu.data, "names") and hdu.data.names:
                    hdu_info["columns"] = list(hdu.data.names)
                    hdu_info["n_columns"] = len(hdu.data.names)

            info["hdus"].append(hdu_info)

    return info


def detect_survey_type(df: pl.DataFrame) -> str:
    """Detect survey type from DataFrame columns."""
    columns = [col.lower() for col in df.columns]

    # GAIA indicators
    gaia_cols = [
        "source_id",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "parallax",
    ]
    if any(col in columns for col in gaia_cols):
        return "gaia"

    # SDSS indicators
    sdss_cols = [
        "objid",
        "modelmag_u",
        "modelmag_g",
        "modelmag_r",
        "modelmag_i",
        "modelmag_z",
    ]
    if any(col in columns for col in sdss_cols):
        return "sdss"

    # NSA indicators
    nsa_cols = ["nsaid", "sersic_n", "sersic_th50"]
    if any(col in columns for col in nsa_cols):
        return "nsa"

    # LINEAR indicators
    linear_cols = ["period", "amplitude", "linear_id"]
    if any(col in columns for col in linear_cols):
        return "linear"

    return "generic"


def save_splits_to_parquet(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    base_path: Union[str, Path],
    dataset_name: str,
) -> Dict[str, Path]:
    """Save train/val/test splits to Parquet files."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        file_path = base_path / f"{dataset_name}_{split_name}.parquet"
        df.write_parquet(file_path)
        paths[split_name] = file_path
        logger.info(f"Saved {split_name} split: {len(df)} rows -> {file_path}")

    return paths


def load_splits_from_parquet(
    base_path: Union[str, Path], dataset_name: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/val/test splits from Parquet files."""
    base_path = Path(base_path)

    train_path = base_path / f"{dataset_name}_train.parquet"
    val_path = base_path / f"{dataset_name}_val.parquet"
    test_path = base_path / f"{dataset_name}_test.parquet"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

    df_train = pl.read_parquet(train_path)
    df_val = pl.read_parquet(val_path)
    df_test = pl.read_parquet(test_path)

    logger.info(
        f"Loaded splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )

    return df_train, df_val, df_test
