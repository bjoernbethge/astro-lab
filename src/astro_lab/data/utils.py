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
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

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


# ============================================================================
# GRAPH UTILITIES
# ============================================================================


def check_graph_consistency(data: Data) -> bool:
    """Check if graph data is consistent."""
    try:
        # Basic checks
        assert data.x.size(0) == data.num_nodes, (
            f"x size {data.x.size(0)} != num_nodes {data.num_nodes}"
        )
        if hasattr(data, "y") and data.y is not None:
            assert data.y.size(0) == data.num_nodes, (
                f"y size {data.y.size(0)} != num_nodes {data.num_nodes}"
            )
        if hasattr(data, "pos") and data.pos is not None:
            assert data.pos.size(0) == data.num_nodes, (
                f"pos size {data.pos.size(0)} != num_nodes {data.num_nodes}"
            )

        # Edge index checks
        if data.edge_index.size(1) > 0:
            max_edge_idx = data.edge_index.max()
            assert max_edge_idx < data.num_nodes, (
                f"edge_index max {max_edge_idx} >= num_nodes {data.num_nodes}"
            )

        logger.info(
            f"✅ Graph consistency check passed: {data.num_nodes} nodes, {data.edge_index.size(1)} edges"
        )
        return True

    except AssertionError as e:
        logger.error(f"❌ Graph consistency check failed: {e}")
        return False


def sample_subgraph_random(data: Data, num_nodes: int, seed: int = 42) -> Data:
    """Sample random subgraph with consistency checks."""
    if data.num_nodes <= num_nodes:
        logger.info(f"Graph already small enough: {data.num_nodes} <= {num_nodes}")
        return data

    logger.info(f"📊 Sampling subgraph: {data.num_nodes} -> {num_nodes} nodes")

    # Random sampling
    torch.manual_seed(seed)
    indices = torch.randperm(data.num_nodes)[:num_nodes]
    indices = torch.sort(indices)[0]  # Sort for better performance

    # Create subgraph
    try:
        device = data.edge_index.device
        indices = indices.to(device)

        # Filter invalid edges first
        if data.edge_index.max() >= data.num_nodes:
            logger.warning(
                f"⚠️ Filtering invalid edges: max={data.edge_index.max()}, nodes={data.num_nodes}"
            )
            valid_edges = (data.edge_index[0] < data.num_nodes) & (
                data.edge_index[1] < data.num_nodes
            )
            edge_index = data.edge_index[:, valid_edges]
            edge_attr = (
                data.edge_attr[valid_edges]
                if hasattr(data, "edge_attr") and data.edge_attr is not None
                else None
            )
        else:
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        # Create subgraph
        sub_edge_index, sub_edge_attr = subgraph(
            indices,
            edge_index,
            edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        # Create new data object
        sub_data = data.__class__()
        sub_data.num_nodes = len(indices)
        sub_data.edge_index = sub_edge_index

        # Copy features safely
        if hasattr(data, "x") and data.x is not None:
            sub_data.x = data.x[indices]

        if hasattr(data, "pos") and data.pos is not None:
            sub_data.pos = data.pos[indices]

        if hasattr(data, "y") and data.y is not None:
            sub_data.y = data.y[indices]

        if sub_edge_attr is not None:
            sub_data.edge_attr = sub_edge_attr

        # Verify consistency
        if not check_graph_consistency(sub_data):
            raise RuntimeError("Subgraph consistency check failed")

        logger.info(
            f"✅ Subgraph created: {sub_data.num_nodes} nodes, {sub_data.edge_index.size(1)} edges"
        )
        return sub_data

    except Exception as e:
        logger.error(f"❌ Subgraph sampling failed: {e}")
        # Fallback: return original data
        logger.warning("⚠️ Falling back to original data")
        return data


def get_graph_statistics(data: Data) -> Dict[str, Any]:
    """Get comprehensive statistics for a graph."""
    stats = {
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.size(1),
        "num_features": data.x.size(1) if hasattr(data, "x") and data.x is not None and data.x.dim() > 1 else 0,
        "has_positions": hasattr(data, "pos") and data.pos is not None,
        "has_labels": hasattr(data, "y") and data.y is not None,
        "is_consistent": check_graph_consistency(data),
    }

    # Handle case where x is 1D (single feature)
    if hasattr(data, "x") and data.x is not None:
        if data.x.dim() == 1:
            stats["num_features"] = 1
        elif data.x.dim() == 2:
            stats["num_features"] = data.x.size(1)

    if hasattr(data, "y") and data.y is not None:
        unique_labels = torch.unique(data.y)
        stats["num_classes"] = len(unique_labels)
        stats["label_distribution"] = {
            int(label): int((data.y == label).sum()) for label in unique_labels
        }

    return stats
