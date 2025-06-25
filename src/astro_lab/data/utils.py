"""
Data utilities for astronomical surveys.

This module provides utility functions for data loading, processing, and visualization
of astronomical data from various surveys including Gaia, SDSS, NSA, and TNG50.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from astropy.io import fits
from astropy.table import Table
from matplotlib.colors import LogNorm

from astro_lab.tensors import SurveyTensorDict
from astro_lab.utils.config.surveys import get_survey_config

logger = logging.getLogger(__name__)

# =========================================================================
# 🛠️ UTILITY FUNCTIONS - Data Loading and Processing
# =========================================================================


def check_astroquery_available() -> bool:
    """Check if astroquery is available."""
    try:
        import astroquery

        return True
    except ImportError:
        return False


def get_data_statistics(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a DataFrame.

    Args:
        df: Polars DataFrame

    Returns:
        Dictionary with statistics
    """
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
        # Calculate magnitude statistics
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
    do_not_scale: bool = False,
    section: Optional[Tuple[slice, ...]] = None,
    max_memory_mb: float = 1000.0,
) -> Optional[Union[np.ndarray, Any]]:
    """
    Load FITS file with memory optimization.

    Args:
        fits_path: Path to FITS file
        hdu_index: HDU index to load
        memmap: Use memory mapping for large files
        do_not_scale: Don't scale data
        section: Section to load (for partial loading)
        max_memory_mb: Maximum memory usage in MB

    Returns:
        FITS data as numpy array or astropy Table
    """
    try:
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        # Check file size
        file_size_mb = fits_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_memory_mb:
            print(f"⚠️ Large FITS file: {file_size_mb:.1f} MB")
            if not memmap:
                print("   Consider using memmap=True for large files")

        # Load FITS file
        with fits.open(fits_path, memmap=memmap, do_not_scale=do_not_scale) as hdul:
            hdu = hdul[hdu_index]

            # Check if it's a table or image
            is_image = hasattr(hdu, "is_image") and hdu.is_image
            if is_image:
                # Image data
                if section is not None:
                    data = hdu.data[section] if hasattr(hdu, "data") else None
                else:
                    data = hdu.data if hasattr(hdu, "data") else None
                return data
            else:
                # Table data
                try:
                    table = Table(hdu.data)
                except Exception:
                    return None
                try:
                    names = [
                        name
                        for name in table.colnames
                        if hasattr(table[name], "shape") and len(table[name].shape) <= 1
                    ]
                    if len(names) < len(table.colnames):
                        filtered_cols = set(table.colnames) - set(names)
                        print(
                            f"📋 Filtered out {len(filtered_cols)} multidimensional columns: {list(filtered_cols)[:5]}{'...' if len(filtered_cols) > 5 else ''}"
                        )
                    filtered_table = table[names]
                    df_pandas = filtered_table.to_pandas()  # type: ignore
                    return pl.from_pandas(df_pandas)
                except Exception as e:
                    print(f"⚠️ Error converting table: {e}")
                    return table

    except Exception as e:
        print(f"Error loading FITS file: {e}")
        return None


def load_fits_table_optimized(
    fits_path: Union[str, Path],
    hdu_index: int = 1,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    as_polars: bool = True,
) -> Optional[Union[pl.DataFrame, Any]]:
    """
    Load FITS table with optimization.

    Args:
        fits_path: Path to FITS file
        hdu_index: HDU index (usually 1 for tables)
        columns: Specific columns to load
        max_rows: Maximum number of rows to load
        as_polars: Return as Polars DataFrame

    Returns:
        FITS table as Polars DataFrame or astropy Table
    """
    try:
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        # Load FITS table
        with fits.open(fits_path) as hdul:
            hdu = hdul[hdu_index]

            is_table = hasattr(hdu, "is_table") and hdu.is_table
            if not is_table:
                raise ValueError(f"HDU {hdu_index} is not a table")

            # Load table
            try:
                table = Table(hdu.data)
            except Exception:
                return None

            # Filter columns if specified
            if columns:
                available_cols = [
                    col for col in columns if col in getattr(table, "colnames", [])
                ]
                if available_cols:
                    table = table[available_cols]
                else:
                    raise ValueError(f"No specified columns found: {columns}")

            # Filter rows if specified
            if max_rows and len(table) > max_rows:
                table = table[:max_rows]

            # Convert to Polars if requested
            if as_polars:
                try:
                    # Handle multidimensional columns
                    names = [
                        name
                        for name in getattr(table, "colnames", [])
                        if hasattr(table[name], "shape") and len(table[name].shape) <= 1
                    ]
                    if len(names) < len(getattr(table, "colnames", [])):
                        filtered_cols = set(table.colnames) - set(names)
                        print(
                            f"📋 Filtered out {len(filtered_cols)} multidimensional columns: {list(filtered_cols)[:5]}{'...' if len(filtered_cols) > 5 else ''}"
                        )

                    # Use only 1D columns for Polars conversion
                    filtered_table = table[names]
                    df_pandas = filtered_table.to_pandas()  # type: ignore
                    return pl.from_pandas(df_pandas)
                except Exception as e:
                    print(f"⚠️ Error converting to Polars: {e}")
                    return table
            else:
                return table

    except Exception as e:
        print(f"Error loading FITS table: {e}")
        return None


def get_fits_info(fits_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about FITS file.

    Args:
        fits_path: Path to FITS file

    Returns:
        Dictionary with FITS information
    """
    try:
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        info = {
            "filename": fits_path.name,
            "file_size_mb": fits_path.stat().st_size / (1024 * 1024),
            "hdus": [],
        }

        with fits.open(fits_path) as hdul:
            for i, hdu in enumerate(hdul):
                hdu_info = {
                    "index": i,
                    "name": hdu.name,
                    "type": hdu.header.get("XTENSION", "IMAGE"),
                }

                if hdu.is_image:
                    hdu_info.update(
                        {
                            "shape": hdu.data.shape if hdu.data is not None else None,
                            "dtype": str(hdu.data.dtype)
                            if hdu.data is not None
                            else None,
                        }
                    )
                elif hdu.is_table:
                    hdu_info.update(
                        {
                            "n_rows": len(hdu.data),
                            "n_cols": len(hdu.data.dtype.names)
                            if hdu.data.dtype.names
                            else 0,
                            "columns": list(hdu.data.dtype.names)
                            if hdu.data.dtype.names
                            else [],
                        }
                    )

                info["hdus"].append(hdu_info)

        return info

    except Exception as e:
        return {"error": str(e)}


def create_training_splits(
    df: pl.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_column: Optional[str] = None,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create train/validation/test splits from DataFrame.

    Args:
        df: Input DataFrame
        test_size: Fraction for test set
        val_size: Fraction for validation set
        stratify_column: Column to stratify on
        random_state: Random seed
        shuffle: Whether to shuffle data

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(df)

    if stratify_column and stratify_column in df.columns:
        # Stratified split
        unique_values = df[stratify_column].unique()
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for value in unique_values:
            subset = df.filter(pl.col(stratify_column) == value)
            n_subset = len(subset)

            if n_subset < 3:
                # Too few samples for stratification, add to train
                train_dfs.append(subset)
                continue

            # Calculate split sizes
            n_test = max(1, int(n_subset * test_size))
            n_val = max(1, int(n_subset * val_size))
            n_train = n_subset - n_test - n_val

            # Create indices
            indices = np.arange(n_subset)
            if shuffle:
                np.random.shuffle(indices)

            # Split indices
            train_indices = indices[:n_train]
            val_indices = indices[n_train : n_train + n_val]
            test_indices = indices[n_train + n_val :]

            # Create splits
            train_dfs.append(subset.take(train_indices))
            val_dfs.append(subset.take(val_indices))
            test_dfs.append(subset.take(test_indices))

        # Combine splits
        train_df = pl.concat(train_dfs) if train_dfs else pl.DataFrame()
        val_df = pl.concat(val_dfs) if val_dfs else pl.DataFrame()
        test_df = pl.concat(test_dfs) if test_dfs else pl.DataFrame()

    else:
        # Simple random split
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        train_df = df.take(train_indices)
        val_df = df.take(val_indices)
        test_df = df.take(test_indices)

    return train_df, val_df, test_df


def save_splits_to_parquet(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    base_path: Union[str, Path],
    dataset_name: str,
) -> Dict[str, Path]:
    """
    Save train/val/test splits to parquet files.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        base_path: Base directory for saving
        dataset_name: Name of dataset

    Returns:
        Dictionary with paths to saved files
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save splits
    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        file_path = base_path / f"{dataset_name}_{split_name}.parquet"
        df.write_parquet(file_path)
        paths[split_name] = file_path

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "n_total": len(df_train) + len(df_val) + len(df_test),
        "columns": df_train.columns,
        "created_at": str(pl.datetime.now()),
    }

    metadata_path = base_path / f"{dataset_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    paths["metadata"] = metadata_path
    return paths


def load_splits_from_parquet(
    base_path: Union[str, Path], dataset_name: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load train/val/test splits from parquet files.

    Args:
        base_path: Base directory containing splits
        dataset_name: Name of dataset

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    base_path = Path(base_path)

    train_path = base_path / f"{dataset_name}_train.parquet"
    val_path = base_path / f"{dataset_name}_val.parquet"
    test_path = base_path / f"{dataset_name}_test.parquet"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"Split files not found in {base_path}")

    df_train = pl.read_parquet(train_path)
    df_val = pl.read_parquet(val_path)
    df_test = pl.read_parquet(test_path)

    return df_train, df_val, df_test


def load_nsa_as_tensors(
    data_path: Union[str, Path],
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load NSA data as tensors for training.

    Args:
        data_path: Path to NSA data directory
        split: Data split ('train', 'val', 'test')
        max_samples: Maximum number of samples

    Returns:
        Dictionary with tensors
    """
    data_path = Path(data_path)
    parquet_file = data_path / f"nsa_v1_0_1_{split}.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(f"NSA data file not found: {parquet_file}")

    # Load data
    df = pl.read_parquet(parquet_file)

    # Apply sampling if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, seed=42)

    # Convert to tensors
    tensors = {}

    # Coordinates
    if "ra" in df.columns and "dec" in df.columns:
        coords = torch.tensor(df.select(["ra", "dec"]).to_numpy(), dtype=torch.float32)
        tensors["coordinates"] = coords

    # Magnitudes
    mag_cols = [col for col in df.columns if "mag" in col.lower()]
    if mag_cols:
        mags = torch.tensor(df.select(mag_cols).to_numpy(), dtype=torch.float32)
        tensors["magnitudes"] = mags

    # Redshift
    if "z" in df.columns:
        redshifts = torch.tensor(df["z"].to_numpy(), dtype=torch.float32)
        tensors["redshifts"] = redshifts

    # Create SurveyTensor
    survey_tensor = SurveyTensorDict(
        data=df.to_numpy(),
        survey_name="nsa",
        data_release="v1_0_1",
        filter_system="sdss",
        coordinate_system="icrs",
        photometric_bands=mag_cols,
        n_objects=len(df),
    )

    tensors["survey_tensor"] = survey_tensor

    return tensors


def create_nsa_survey_tensor(
    data_path: Union[str, Path],
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Create NSA SurveyTensor for training.

    Args:
        data_path: Path to NSA data directory
        split: Data split ('train', 'val', 'test')
        max_samples: Maximum number of samples

    Returns:
        SurveyTensor instance
    """
    from astro_lab.tensors import SurveyTensorDict

    data_path = Path(data_path)
    parquet_file = data_path / f"nsa_v1_0_1_{split}.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(f"NSA data file not found: {parquet_file}")

    # Load data
    df = pl.read_parquet(parquet_file)

    # Apply sampling if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, seed=42)

    # Create SurveyTensor
    survey_tensor = SurveyTensorDict(
        data=df.to_numpy(),
        survey_name="nsa",
        data_release="v1_0_1",
        filter_system="sdss",
        coordinate_system="icrs",
        photometric_bands=[col for col in df.columns if "mag" in col.lower()],
        n_objects=len(df),
    )

    return survey_tensor


def test_nsa_tensor_compatibility():
    """Test NSA tensor compatibility."""
    try:
        # Test loading NSA data
        data_path = Path("data/processed/nsa")
        if data_path.exists():
            tensors = load_nsa_as_tensors(data_path, "train", max_samples=100)
            print("✅ NSA tensor loading successful")
            print(f"   Tensors: {list(tensors.keys())}")
            return True
        else:
            print("⚠️ NSA data not found, skipping test")
            return False
    except Exception as e:
        print(f"❌ NSA tensor test failed: {e}")
        return False


def convert_nsa_fits_to_parquet(
    fits_path: Union[str, Path],
    parquet_path: Union[str, Path],
    features: Optional[List[str]] = None,
    max_memory_mb: float = 2000.0,
) -> Path:
    """
    Convert NSA FITS file to Parquet format.

    Args:
        fits_path: Path to NSA FITS file
        parquet_path: Output parquet path
        features: List of features to include
        max_memory_mb: Maximum memory usage

    Returns:
        Path to created parquet file
    """

    fits_path = Path(fits_path)
    parquet_path = Path(parquet_path)

    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    print(f"🔄 Converting NSA FITS to Parquet: {fits_path.name}")

    # Load FITS table
    table = load_fits_table_optimized(fits_path, hdu_index=1, as_polars=False)

    if table is None:
        raise ValueError("Could not load FITS table")

    # Convert to Polars DataFrame
    df = pl.from_pandas(table.to_pandas())

    # Filter features if specified
    if features:
        available_features = [f for f in features if f in df.columns]
        df = df.select(available_features)

    # Write to Parquet with compression
    df.write_parquet(parquet_path, compression="zstd")

    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"✅ Successfully converted to {parquet_path.name} ({file_size_mb:.1f} MB)")

    return parquet_path


def extract_fits_image_data(
    fits_path: Union[str, Path], hdu_index: int = 0
) -> Dict[str, Any]:
    """
    Extract image data from FITS file.

    Args:
        fits_path: Path to FITS file
        hdu_index: HDU index to extract

    Returns:
        Dictionary with image data and metadata
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS

        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        with fits.open(fits_path) as hdul:
            hdu = hdul[hdu_index]

            # Extract image data
            image_data = hdu.data
            if image_data is None:
                raise ValueError(f"No image data found in HDU {hdu_index}")

            # Extract header
            header = hdu.header

            # Try to extract WCS information
            wcs = None
            try:
                wcs = WCS(header)
            except Exception as e:
                logger.warning(f"Could not extract WCS: {e}")

            # Extract useful metadata
            metadata = {
                "filename": fits_path.name,
                "hdu_index": hdu_index,
                "shape": image_data.shape,
                "dtype": str(image_data.dtype),
                "bunit": header.get("BUNIT", "Unknown"),
                "object": header.get("OBJECT", "Unknown"),
                "telescope": header.get("TELESCOP", "Unknown"),
                "instrument": header.get("INSTRUME", "Unknown"),
                "exposure_time": header.get("EXPTIME", None),
                "filter": header.get("FILTER", None),
                "date_obs": header.get("DATE-OBS", None),
            }

            return {
                "image_data": image_data,
                "header": header,
                "wcs": wcs,
                "metadata": metadata,
            }

    except Exception as e:
        logger.error(f"Failed to extract FITS image data: {e}")
        raise


def create_image_tensor_from_fits(
    fits_path: Union[str, Path], normalize: bool = True, **kwargs
) -> torch.Tensor:
    """
    Create PyTorch tensor from FITS image.

    Args:
        fits_path: Path to FITS file
        normalize: Whether to normalize the data
        **kwargs: Additional arguments for extract_fits_image_data

    Returns:
        PyTorch tensor with image data
    """
    try:
        # Extract image data
        fits_data = extract_fits_image_data(fits_path, **kwargs)
        image_data = fits_data["image_data"]

        # Convert to tensor
        tensor = torch.tensor(image_data, dtype=torch.float32)

        # Normalize if requested
        if normalize:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        return tensor

    except Exception as e:
        logger.error(f"Failed to create image tensor from FITS: {e}")
        raise


def visualize_fits_image(
    fits_path: Union[str, Path], backend: str = "matplotlib", **kwargs
) -> Any:
    """
    Visualize FITS image using different backends.

    Args:
        fits_path: Path to FITS file
        backend: Visualization backend ('matplotlib', 'plotly', 'bpy')
        **kwargs: Additional arguments

    Returns:
        Visualization object
    """
    try:
        fits_data = extract_fits_image_data(fits_path)
        image_data = fits_data["image_data"]

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm

            plt.figure(figsize=(10, 8))
            plt.imshow(image_data, cmap="viridis", norm=LogNorm())
            plt.colorbar(label="Intensity")
            plt.title(f"FITS Image: {fits_data['metadata']['filename']}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            return plt.gcf()

        elif backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure(
                data=go.Heatmap(
                    z=image_data,
                    colorscale="Viridis",
                    zmid=np.median(image_data),
                )
            )

            fig.update_layout(
                title=f"FITS Image: {fits_data['metadata']['filename']}",
                xaxis_title="X",
                yaxis_title="Y",
            )

            fig.show()
            return fig

        elif backend == "bpy":
            # Blender visualization (if available)
            try:
                logger.warning(
                    "Blender visualization not available - create_image_mesh function not found"
                )
                return None
            except ImportError:
                logger.warning("Blender visualization not available")
                return None

        else:
            raise ValueError(f"Unknown backend: {backend}")

    except Exception as e:
        logger.error(f"Failed to visualize FITS image: {e}")
        raise
