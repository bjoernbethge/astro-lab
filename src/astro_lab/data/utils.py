"""
AstroLab Data Utilities
=======================

Utility functions for data loading, processing, and analysis.
Includes FITS handling, statistics, and data validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import astroquery
import numpy as np
import polars as pl
import torch
import torch_geometric
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .config import data_config

# Try to import astropy for FITS handling
# Import astropy for FITS data

# Import astroquery for external data


def check_astroquery_available() -> bool:
    """Check if astroquery is available for data downloads."""
    return True


def get_data_statistics(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive statistics for astronomical DataFrame.

    Args:
        df: Input Polars DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "memory_usage_mb": df.estimated_size() / (1024 * 1024),
        "numeric_columns": [],
        "missing_data": {},
    }

    # Analyze each column
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            stats["numeric_columns"].append(col)

        # Count missing values
        null_count = df[col].null_count()
        if null_count > 0:
            stats["missing_data"][col] = {
                "null_count": null_count,
                "null_percentage": (null_count / len(df)) * 100,
            }

    return stats


def _detect_magnitude_columns(df: pl.DataFrame) -> List[str]:
    """Detect likely magnitude columns in a DataFrame."""
    mag_patterns = [
        "mag",
        "magnitude",
        "petromag",
        "fibermag",
        "modelmag",
        "psfmag",
        "phot_",
        "_mag",
    ]

    magnitude_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in mag_patterns):
            # Check if it's actually numeric
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                magnitude_cols.append(col)

    return magnitude_cols


def load_fits_optimized(
    fits_path: Union[str, Path],
    hdu_index: int = 0,
    memmap: bool = True,
    do_not_scale: bool = False,
    section: Optional[Tuple[slice, ...]] = None,
    max_memory_mb: float = 1000.0,
) -> Optional[Union[np.ndarray, Table]]:
    """
    Load FITS data with optimizations for large files.

    Args:
        fits_path: Path to FITS file
        hdu_index: HDU index to load
        memmap: Use memory mapping
        do_not_scale: Disable automatic scaling
        section: Data section to load
        max_memory_mb: Memory limit for automatic memmap

    Returns:
        Loaded data as ndarray or Table
    """
    fits_path = Path(fits_path)
    if not fits_path.exists():
        print(f"FITS file not found: {fits_path}")
        return None

    try:
        # Check file size for auto-memmap
        file_size_mb = fits_path.stat().st_size / (1024 * 1024)
        auto_memmap = memmap or (file_size_mb > max_memory_mb)

        print(f"Loading FITS file: {fits_path.name} ({file_size_mb:.1f} MB)")
        if auto_memmap:
            print("Using memory mapping for efficient loading")

        with fits.open(
            fits_path, memmap=auto_memmap, do_not_scale_image_data=do_not_scale
        ) as hdul:
            if hdu_index >= len(hdul):
                print(f"HDU index {hdu_index} not available (max: {len(hdul) - 1})")
                return None

            hdu = hdul[hdu_index]  # type: ignore

            # Type-safe HDU data access
            if hasattr(hdu, "data") and getattr(hdu, "data", None) is not None:
                data = getattr(hdu, "data")  # type: ignore

                # Apply section if specified
                if section is not None:
                    data = data[section]  # type: ignore

                return data
            else:
                # Header-only HDU - return None for now
                print("No data in HDU - header-only HDUs not supported")
                return None

    except Exception as e:
        print(f"Error loading FITS file: {e}")
        return None


def load_fits_table_optimized(
    fits_path: Union[str, Path],
    hdu_index: int = 1,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    as_polars: bool = True,
) -> Optional[Union[pl.DataFrame, Table]]:
    """
    Load FITS table data with optimizations.

    Args:
        fits_path: Path to FITS file
        hdu_index: HDU index (typically 1 for tables)
        columns: Specific columns to load
        max_rows: Maximum rows to load
        as_polars: Return as Polars DataFrame

    Returns:
        Loaded table as DataFrame or Table
    """
    fits_path = Path(fits_path)
    if not fits_path.exists():
        print(f"FITS file not found: {fits_path}")
        return None

    try:
        with fits.open(fits_path, memmap=True) as hdul:
            if hdu_index >= len(hdul):
                print(f"HDU index {hdu_index} not available")
                return None

            hdu = hdul[hdu_index]  # type: ignore

            # Type-safe HDU data access
            if not hasattr(hdu, "data") or getattr(hdu, "data", None) is None:
                print(f"No table data in HDU {hdu_index}")
                return None

            # Convert to astropy Table
            table = Table(getattr(hdu, "data"))  # type: ignore

            # Select columns if specified
            if columns:
                available_cols = [col for col in columns if col in table.colnames]
                if available_cols:
                    table = table[available_cols]
                else:
                    print(f"None of the requested columns found: {columns}")
                    return None

            # Limit rows if specified
            if max_rows and len(table) > max_rows:
                table = table[:max_rows]

            if as_polars:
                # Convert to Polars DataFrame with proper type handling
                try:
                    # Filter out multidimensional columns as suggested by astropy
                    # This handles the NSA catalog issue with NMGY, ABSMAG, etc.
                    names = [
                        name for name in table.colnames if len(table[name].shape) <= 1
                    ]
                    if len(names) < len(table.colnames):
                        filtered_cols = set(table.colnames) - set(names)
                        print(
                            f"📋 Filtered out {len(filtered_cols)} multidimensional columns: {list(filtered_cols)[:5]}{'...' if len(filtered_cols) > 5 else ''}"
                        )

                    # Use only 1D columns for Polars conversion
                    filtered_table = table[names]
                    df_pandas = filtered_table.to_pandas()  # type: ignore
                    return pl.from_pandas(df_pandas)
                except Exception as e:
                    print(f"Error converting to Polars: {e}")
                    return None
            else:
                return table

    except Exception as e:
        print(f"Error loading FITS table: {e}")
        return None


def get_fits_info(fits_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a FITS file.

    Args:
        fits_path: Path to FITS file

    Returns:
        Dictionary with file information
    """

    fits_path = Path(fits_path)
    if not fits_path.exists():
        return {"error": f"File not found: {fits_path}"}

    try:
        info = {
            "file_path": str(fits_path),
            "file_size_mb": fits_path.stat().st_size / (1024 * 1024),
            "hdus": [],
        }

        with fits.open(fits_path) as hdul:
            for i, hdu in enumerate(hdul):
                hdu_info = {
                    "index": i,
                    "name": hdu.name,
                    "type": type(hdu).__name__,
                }

                if hasattr(hdu, "data") and hdu.data is not None:
                    data = hdu.data
                    hdu_info.update(
                        {
                            "shape": data.shape,
                            "dtype": str(data.dtype),
                            "data_size_mb": data.nbytes / (1024 * 1024),
                        }
                    )

                    # Check for scaling
                    if hasattr(hdu, "header"):
                        header = hdu.header
                        if "BSCALE" in header or "BZERO" in header:
                            hdu_info["scaled"] = True
                            hdu_info["bscale"] = header.get("BSCALE", 1.0)
                            hdu_info["bzero"] = header.get("BZERO", 0.0)
                        else:
                            hdu_info["scaled"] = False

                    # Table information
                    if (
                        hasattr(data, "dtype")
                        and hasattr(data.dtype, "names")
                        and data.dtype.names
                    ):
                        hdu_info["table_rows"] = len(data)
                        hdu_info["table_columns"] = len(data.dtype.names)
                        hdu_info["column_names"] = list(data.dtype.names)

                info["hdus"].append(hdu_info)

        return info

    except Exception as e:
        return {"error": f"Error reading FITS file: {e}"}


def create_training_splits(
    df: pl.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_column: Optional[str] = None,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create train/validation/test splits using native Polars operations.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame to split
    test_size : float, default 0.2
        Proportion of data for test set
    val_size : float, default 0.1
        Proportion of data for validation set
    stratify_column : str, optional
        Column to use for stratified splitting (basic implementation)
    random_state : int, optional
        Random seed for reproducibility
    shuffle : bool, default True
        Whether to shuffle before splitting

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Training, validation, and test DataFrames

    Raises
    ------
    ValueError
        If split sizes are invalid
    """
    # Validate split sizes
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not 0 < val_size < 1:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
    if test_size + val_size >= 1:
        raise ValueError(
            f"test_size + val_size must be < 1, got {test_size + val_size}"
        )

    print(
        f"🔄 Creating splits: train={1 - test_size - val_size:.1%}, val={val_size:.1%}, test={test_size:.1%}"
    )

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    n_total = len(df)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    if shuffle:
        # Create random indices
        indices = np.random.permutation(n_total)
    else:
        indices = np.arange(n_total)

    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    # Create splits using the indices
    df_train = df[train_indices.tolist()]
    df_val = df[val_indices.tolist()]
    df_test = df[test_indices.tolist()]

    print(
        f"✅ Created splits - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
    )

    return df_train, df_val, df_test


def save_splits_to_parquet(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    base_path: Union[str, Path],
    dataset_name: str,
) -> Dict[str, Path]:
    """
    Save train/validation/test splits to Parquet files.

    Parameters
    ----------
    df_train, df_val, df_test : pl.DataFrame
        DataFrames to save
    base_path : Union[str, Path]
        Base directory for saving
    dataset_name : str
        Name of the dataset for file naming

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping split names to file paths
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split_name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        filename = f"{dataset_name}_{split_name}.parquet"
        filepath = base_path / filename

        df.write_parquet(filepath)
        paths[split_name] = filepath
        print(f"💾 Saved {split_name} split to {filepath}")

    return paths


def load_splits_from_parquet(
    base_path: Union[str, Path], dataset_name: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load train/validation/test splits from Parquet files.

    Parameters
    ----------
    base_path : Union[str, Path]
        Base directory containing the split files
    dataset_name : str
        Name of the dataset for file naming

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Training, validation, and test DataFrames
    """
    base_path = Path(base_path)

    df_train = pl.read_parquet(base_path / f"{dataset_name}_train.parquet")
    df_val = pl.read_parquet(base_path / f"{dataset_name}_val.parquet")
    df_test = pl.read_parquet(base_path / f"{dataset_name}_test.parquet")

    print(
        f"📂 Loaded splits - Train: {df_train.height}, Val: {df_val.height}, Test: {df_test.height}"
    )

    return df_train, df_val, df_test


def load_nsa_as_tensors(
    data_path: Union[str, Path],
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load NSA data and convert to tensor format for point cloud models.

    Args:
        data_path: Path to NSA processed data directory
        split: Which split to load ('train', 'val', 'test')
        max_samples: Limit number of samples

    Returns:
        Dictionary with 'pos', 'x', and 'edge_index' tensors
    """
    data_path = Path(data_path)

    # Load parquet data
    parquet_file = data_path / f"nsa_v1_0_1_{split}.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"NSA {split} data not found: {parquet_file}")

    df = pl.read_parquet(parquet_file)

    if max_samples:
        df = df.head(max_samples)

    # Extract coordinates (RA, DEC, Z) -> 3D positions
    ra = df["RA"].to_numpy()
    dec = df["DEC"].to_numpy()
    z = df["Z"].to_numpy()

    # Convert to 3D Cartesian coordinates (simplified)
    c = 299792.458  # km/s
    H0 = 70.0  # km/s/Mpc
    distance = c * z / H0  # Mpc

    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = distance * np.sin(dec_rad)

    pos = torch.tensor(np.column_stack([x, y, z_coord]), dtype=torch.float32)

    # Extract features
    feature_cols = ["RA", "DEC", "Z"]

    # Add photometric features if available
    photo_cols = ["ELPETRO_MASS", "SERSIC_MASS", "MAG"]
    available_photo = [col for col in photo_cols if col in df.columns]
    feature_cols.extend(available_photo)

    features = df.select(feature_cols).to_numpy()
    features = np.nan_to_num(features, nan=0.0)
    x = torch.tensor(features, dtype=torch.float32)

    # Load graph structure if available
    graph_file = data_path / f"graphs_{split}" / f"nsa_v1_0_1_{split}.pt"
    edge_index = None

    if graph_file.exists():
        try:
            # Load with weights_only=False for PyTorch Geometric compatibility
            import torch_geometric

            # Add safe globals for PyTorch Geometric
            torch.serialization.add_safe_globals(
                [
                    torch_geometric.data.Data,
                    torch_geometric.data.data.DataEdgeAttr,
                    torch_geometric.data.data.DataNodeAttr,
                ]
            )

            graph_data = torch.load(graph_file, map_location="cpu", weights_only=False)
            edge_index = graph_data.edge_index
        except Exception as e:
            print(f"Warning: Could not load graph structure: {e}")

    result = {
        "pos": pos,
        "x": x,
    }

    if edge_index is not None:
        result["edge_index"] = edge_index

    return result


def create_nsa_survey_tensor(
    data_path: Union[str, Path],
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Create SurveyTensor from NSA data.

    Args:
        data_path: Path to NSA processed data
        split: Which split to load
        max_samples: Limit number of samples

    Returns:
        SurveyTensor instance
    """
    from astro_lab.tensors import SurveyTensor

    data_path = Path(data_path)
    parquet_file = data_path / f"nsa_v1_0_1_{split}.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(f"NSA {split} data not found: {parquet_file}")

    df = pl.read_parquet(parquet_file)

    if max_samples:
        df = df.head(max_samples)

    # Convert to tensor - handle mixed data types
    numeric_df = df.select(
        [
            col
            for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
    )
    data_tensor = torch.tensor(numeric_df.to_numpy(), dtype=torch.float32)

    # Create column mapping for numeric columns only
    column_mapping = {col: i for i, col in enumerate(numeric_df.columns)}

    # Create survey tensor
    survey_tensor = SurveyTensor(
        data_tensor,
        survey_name="nsa",
        data_release="v1_0_1",
        filter_system="galex_sdss",
        column_mapping=column_mapping,
        survey_metadata={
            "num_galaxies": len(df),
            "redshift_range": [df["Z"].min(), df["Z"].max()],
            "split": split,
        },
    )

    return survey_tensor


def test_nsa_tensor_compatibility():
    """Test NSA data compatibility with tensor system."""
    try:
        # Test loading
        data = load_nsa_as_tensors("data/processed/nsa", "train", max_samples=100)
        print("✅ NSA tensor loading successful")
        print(f"   Positions: {data['pos'].shape}")
        print(f"   Features: {data['x'].shape}")
        if "edge_index" in data:
            print(f"   Edges: {data['edge_index'].shape}")

        # Test survey tensor
        survey_tensor = create_nsa_survey_tensor(
            "data/processed/nsa", "train", max_samples=100
        )
        print("✅ NSA SurveyTensor creation successful")
        print(f"   Survey: {survey_tensor.survey_name}")
        print(f"   Shape: {survey_tensor.shape}")

        return True

    except Exception as e:
        print(f"❌ NSA tensor compatibility test failed: {e}")
        return False


def convert_nsa_fits_to_parquet(
    fits_path: Union[str, Path],
    parquet_path: Union[str, Path],
    features: Optional[List[str]] = None,
    max_memory_mb: float = 2000.0,
) -> Path:
    """
    Convert large NSA FITS catalog to a memory-efficient Parquet file.
    This version now directly uses the robust `load_fits_table_optimized`.
    """
    fits_path = Path(fits_path)
    parquet_path = Path(parquet_path)

    if parquet_path.exists():
        print(f"✅ Parquet file already exists: {parquet_path.name}")
        return parquet_path

    print(f"🔄 Converting NSA FITS to Parquet: {fits_path.name}")

    try:
        # Use the optimized FITS table loader directly to get a Polars DataFrame
        df = load_fits_table_optimized(
            fits_path, as_polars=True, max_rows=None, hdu_index=1
        )

        if df is None:
            raise ValueError("Failed to load NSA FITS table as Polars DataFrame")

        print(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")

        # Select features if specified
        if features:
            available_features = [f for f in features if f in df.columns]
            df = df.select(available_features)

        # Write to Parquet with compression
        df.write_parquet(parquet_path, compression="zstd")
        
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"✅ Successfully converted to {parquet_path.name} ({file_size_mb:.1f} MB)")
        return parquet_path

    except Exception as e:
        print(f"❌ Failed to convert NSA FITS: {e}")
        raise


def extract_fits_image_data(fits_path: Union[str, Path], hdu_index: int = 0) -> Dict[str, Any]:
    """
    Extract image data and metadata from FITS file.
    
    Args:
        fits_path: Path to FITS file
        hdu_index: HDU index to extract (default: 0)
        
    Returns:
        Dictionary containing:
        - image_data: 2D numpy array of image data
        - header: FITS header information
        - wcs: World Coordinate System info (if available)
        - metadata: Extracted metadata
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        import numpy as np
        
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
                'filename': str(fits_path),
                'hdu_index': hdu_index,
                'shape': image_data.shape,
                'dtype': str(image_data.dtype),
                'min_value': float(np.min(image_data)),
                'max_value': float(np.max(image_data)),
                'mean_value': float(np.mean(image_data)),
                'std_value': float(np.std(image_data)),
            }
            
            # Extract common FITS keywords
            common_keys = ['OBJECT', 'TELESCOP', 'INSTRUME', 'FILTER', 'EXPTIME', 
                          'DATE-OBS', 'RA', 'DEC', 'EQUINOX', 'CRVAL1', 'CRVAL2']
            
            for key in common_keys:
                if key in header:
                    try:
                        metadata[key] = header[key]
                    except:
                        pass
            
            return {
                'image_data': image_data,
                'header': header,
                'wcs': wcs,
                'metadata': metadata
            }
            
    except Exception as e:
        logger.error(f"Failed to extract FITS image data: {e}")
        raise


def create_image_tensor_from_fits(fits_path: Union[str, Path], 
                                 normalize: bool = True,
                                 **kwargs) -> torch.Tensor:
    """
    Create a tensor from FITS image data.
    
    Args:
        fits_path: Path to FITS file
        normalize: Whether to normalize the image data
        **kwargs: Additional arguments for extract_fits_image_data
        
    Returns:
        torch.Tensor with image data
    """
    try:
        import torch
        
        # Extract image data
        fits_data = extract_fits_image_data(fits_path, **kwargs)
        image_data = fits_data['image_data']
        
        # Convert to tensor
        tensor = torch.tensor(image_data, dtype=torch.float32)
        
        # Normalize if requested
        if normalize:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        
        return tensor
        
    except Exception as e:
        logger.error(f"Failed to create image tensor from FITS: {e}")
        raise


def visualize_fits_image(fits_path: Union[str, Path], 
                        backend: str = 'matplotlib',
                        **kwargs) -> Any:
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
        image_data = fits_data['image_data']
        
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            plt.figure(figsize=(10, 8))
            plt.imshow(image_data, cmap='viridis', norm=LogNorm())
            plt.colorbar(label='Intensity')
            plt.title(f"FITS Image: {fits_data['metadata']['filename']}")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
            return plt.gcf()
            
        elif backend == 'plotly':
            import plotly.graph_objects as go
            
            fig = go.Figure(data=go.Heatmap(
                z=image_data,
                colorscale='Viridis',
                zmin=image_data.min(),
                zmax=image_data.max()
            ))
            
            fig.update_layout(
                title=f"FITS Image: {fits_data['metadata']['filename']}",
                xaxis_title='X',
                yaxis_title='Y'
            )
            
            fig.show()
            return fig
            
        else:
            raise ValueError(f"Unsupported backend: {backend}")
            
    except Exception as e:
        logger.error(f"Failed to visualize FITS image: {e}")
        raise


if __name__ == "__main__":
    test_nsa_tensor_compatibility()
