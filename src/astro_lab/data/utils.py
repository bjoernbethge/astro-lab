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
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
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
        "num_features": data.x.size(1)
        if hasattr(data, "x") and data.x is not None and data.x.dim() > 1
        else 0,
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


def pyg_collate_fn(batch):
    """
    Custom collate function for PyTorch Geometric Data objects.

    Args:
        batch: List of Data objects

    Returns:
        Batched Data object or single Data object
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]

    if not batch:
        return None

    # If single item, return as-is
    if len(batch) == 1:
        return batch[0]

    # For multiple items, try to create a Batch
    try:
        return Batch.from_data_list(batch)
    except Exception as e:
        logger.warning(f"Could not batch PyG data objects: {e}, returning first item")
        return batch[0]


class SafePyGDataLoader(DataLoader):
    """
    Safe DataLoader for PyTorch Geometric Data objects that handles pin_memory correctly.

    This DataLoader ensures that pin_memory only applies to CPU tensors and handles
    the proper device transfer of graph data objects.
    """

    def __init__(self, *args, **kwargs):
        # Extract our custom parameters
        self.use_smart_pinning = kwargs.pop("use_smart_pinning", True)
        self.target_device = kwargs.pop("target_device", None)
        self.non_blocking_transfer = kwargs.pop("non_blocking_transfer", True)

        # Set custom collate function for PyG data
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = pyg_collate_fn

        # Disable standard pin_memory for PyG data
        original_pin_memory = kwargs.get("pin_memory", False)
        if original_pin_memory and self.use_smart_pinning:
            logger.info("🔧 Using smart pin_memory for PyTorch Geometric data")
            kwargs["pin_memory"] = False  # Disable standard pinning

        super().__init__(*args, **kwargs)
        self.original_pin_memory = original_pin_memory

    def __iter__(self):
        """Override iterator to handle smart pinning and device transfer."""
        for batch in super().__iter__():
            if self.use_smart_pinning and self.original_pin_memory:
                batch = self._smart_pin_and_transfer(batch)
            elif self.target_device is not None:
                batch = self._transfer_to_device(batch)
            yield batch

    def _smart_pin_and_transfer(
        self, batch: Union[Data, List[Data]]
    ) -> Union[Data, List[Data]]:
        """
        Intelligently pin CPU tensors and transfer to target device.

        Args:
            batch: PyTorch Geometric Data object or list of Data objects

        Returns:
            Processed batch with pinned memory and device transfer
        """
        if isinstance(batch, list):
            return [self._process_single_data(data) for data in batch]
        else:
            return self._process_single_data(batch)

    def _process_single_data(self, data: Data) -> Data:
        """Process a single Data object for smart pinning and transfer."""
        if not isinstance(data, Data):
            return data

        # Pin CPU tensors only
        if hasattr(data, "x") and data.x is not None and data.x.device.type == "cpu":
            try:
                data.x = data.x.pin_memory()
            except Exception as e:
                logger.debug(f"Could not pin x tensor: {e}")

        if (
            hasattr(data, "edge_index")
            and data.edge_index is not None
            and data.edge_index.device.type == "cpu"
        ):
            try:
                data.edge_index = data.edge_index.pin_memory()
            except Exception as e:
                logger.debug(f"Could not pin edge_index tensor: {e}")

        if (
            hasattr(data, "pos")
            and data.pos is not None
            and data.pos.device.type == "cpu"
        ):
            try:
                data.pos = data.pos.pin_memory()
            except Exception as e:
                logger.debug(f"Could not pin pos tensor: {e}")

        if hasattr(data, "y") and data.y is not None and data.y.device.type == "cpu":
            try:
                data.y = data.y.pin_memory()
            except Exception as e:
                logger.debug(f"Could not pin y tensor: {e}")

        # Pin mask tensors if they exist
        for mask_name in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(data, mask_name):
                mask = getattr(data, mask_name)
                if mask is not None and mask.device.type == "cpu":
                    try:
                        setattr(data, mask_name, mask.pin_memory())
                    except Exception as e:
                        logger.debug(f"Could not pin {mask_name} tensor: {e}")

        # Transfer to target device with non-blocking if available
        if self.target_device is not None:
            data = self._transfer_to_device(data)

        return data

    def _transfer_to_device(
        self, batch: Union[Data, List[Data]]
    ) -> Union[Data, List[Data]]:
        """Transfer batch to target device with non-blocking transfer."""
        if self.target_device is None:
            return batch

        try:
            if isinstance(batch, list):
                return [
                    data.to(self.target_device, non_blocking=self.non_blocking_transfer)
                    for data in batch
                ]
            else:
                return batch.to(
                    self.target_device, non_blocking=self.non_blocking_transfer
                )
        except Exception as e:
            logger.warning(
                f"Non-blocking transfer failed, falling back to blocking: {e}"
            )
            if isinstance(batch, list):
                return [data.to(self.target_device) for data in batch]
            else:
                return batch.to(self.target_device)


def create_optimized_dataloader(
    dataset: Any,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    drop_last: bool = False,
    target_device: Optional[torch.device] = None,
    use_smart_pinning: bool = True,
    non_blocking_transfer: bool = True,
    **kwargs,
) -> SafePyGDataLoader:
    """
    Create an optimized DataLoader for PyTorch Geometric data.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory (intelligently applied)
        persistent_workers: Whether to keep workers alive
        prefetch_factor: Number of batches to prefetch
        drop_last: Whether to drop the last incomplete batch
        target_device: Device to transfer data to
        use_smart_pinning: Whether to use smart pinning for PyG data
        non_blocking_transfer: Whether to use non-blocking device transfer
        **kwargs: Additional arguments for DataLoader

    Returns:
        Optimized SafePyGDataLoader instance
    """
    # Auto-detect target device if not specified
    if target_device is None and torch.cuda.is_available():
        target_device = torch.device("cuda")

    # Prepare kwargs
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "target_device": target_device,
        "use_smart_pinning": use_smart_pinning,
        "non_blocking_transfer": non_blocking_transfer,
        **kwargs,
    }

    # Add worker-specific options
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
    else:
        # Remove worker-specific options for single-process loading
        dataloader_kwargs.pop("persistent_workers", None)

    logger.info(
        f"📊 Creating optimized DataLoader: "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, smart_pinning={use_smart_pinning}, "
        f"non_blocking={non_blocking_transfer}, target_device={target_device}"
    )

    return SafePyGDataLoader(dataset, **dataloader_kwargs)


def optimize_memory_transfer(
    data: Union[Data, torch.Tensor],
    target_device: torch.device,
    use_pinning: bool = True,
    non_blocking: bool = True,
) -> Union[Data, torch.Tensor]:
    """
    Optimize memory transfer for single data objects.

    Args:
        data: Data to transfer
        target_device: Target device
        use_pinning: Whether to pin memory before transfer
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Transferred data
    """
    if isinstance(data, torch.Tensor):
        if use_pinning and data.device.type == "cpu":
            try:
                data = data.pin_memory()
            except Exception as e:
                logger.debug(f"Could not pin tensor: {e}")

        return data.to(target_device, non_blocking=non_blocking)

    elif isinstance(data, Data):
        # Pin individual tensors if they're on CPU
        if use_pinning:
            for attr_name in ["x", "edge_index", "pos", "y", "edge_attr"]:
                if hasattr(data, attr_name):
                    attr_value = getattr(data, attr_name)
                    if attr_value is not None and attr_value.device.type == "cpu":
                        try:
                            setattr(data, attr_name, attr_value.pin_memory())
                        except Exception as e:
                            logger.debug(f"Could not pin {attr_name}: {e}")

        return data.to(target_device, non_blocking=non_blocking)

    else:
        return data


class MemoryOptimizedDataModule:
    """
    Mixin class for memory optimization in PyTorch Lightning DataModules.
    """

    def get_optimized_dataloader_kwargs(
        self,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 2,
        drop_last: bool = False,
        target_device: Optional[torch.device] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get optimized dataloader configuration."""

        # Auto-detect optimal settings based on system
        if torch.cuda.is_available():
            if target_device is None:
                target_device = torch.device("cuda")

            # Optimize for GPU systems
            if num_workers == 0:
                num_workers = min(4, torch.get_num_threads())  # Conservative default
        else:
            target_device = torch.device("cpu")
            pin_memory = False  # No point pinning on CPU-only systems

        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers and num_workers > 0,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "drop_last": drop_last,
            "target_device": target_device,
            "use_smart_pinning": True,
            "non_blocking_transfer": True,
            **kwargs,
        }
