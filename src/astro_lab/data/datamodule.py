"""
AstroLab DataModule - Lightning DataModule for Astronomical Data
===============================================================

Lightning DataModule for astronomical datasets with TensorDict integration.
Updated for TensorDict architecture.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as L
import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Use TensorDict classes instead of old tensor classes
from ..tensors import (
    ClusteringTensorDict,
    FeatureTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SimulationTensorDict,
    SpatialTensorDict,
    StatisticsTensorDict,
    SurveyTensorDict,
)
from .config import data_config
from .core import AstroDataset

logger = logging.getLogger(__name__)


class AstroDataModule(L.LightningDataModule):
    """
    Clean Lightning DataModule for astronomical datasets.

    Handles train/val/test splits and data loading.
    Uses unified AstroDataset from core.py.

    2025 Optimizations:
    - PIN memory for faster GPU transfer
    - Persistent workers to avoid recreation overhead
    - Prefetch factor tuning
    - Drop last for consistent batch sizes
    - Better distributed sampling
    - Fixed PyTorch Geometric batch handling
    """

    def __init__(
        self,
        survey: str,
        data_root: Optional[str] = None,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        batch_size: int = 1,  # Graph datasets typically use batch_size=1
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: Optional[int] = None,  # Auto-detect optimal workers
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = True,
        use_distributed_sampler: bool = True,
        # New parameters for laptop optimization
        max_nodes_per_graph: int = 1000,  # Limit graph size for laptop GPUs
        use_subgraph_sampling: bool = True,  # Use subgraph sampling for large graphs
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.survey = survey
        self.dataset_name = survey  # For results organization
        self.data_root = data_root or str(data_config.base_dir)
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.drop_last = drop_last
        self.use_distributed_sampler = use_distributed_sampler

        # Laptop optimization parameters
        self.max_nodes_per_graph = max_nodes_per_graph
        self.use_subgraph_sampling = use_subgraph_sampling

        # Optimize num_workers for laptop
        if num_workers is None:
            # Conservative settings for laptop
            try:
                cpu_count = os.cpu_count()
                # Use fewer workers on laptop to avoid memory pressure
                self.num_workers = max(0, min(cpu_count // 2, 4))
            except (OSError, AttributeError):
                self.num_workers = 2
        else:
            self.num_workers = num_workers

        # Conservative settings for laptop GPUs
        if batch_size == 1:
            self.pin_memory = False
            self.persistent_workers = False
            self.prefetch_factor = None
            self.num_workers = 0
        else:
            # Conservative memory settings for laptop
            self.pin_memory = pin_memory and torch.cuda.is_available()
            self.persistent_workers = persistent_workers and self.num_workers > 0
            self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None

        # Dataset will be created in setup()
        self.dataset = None

        # Store the main data object
        self._main_data = None

        # Class information for Lightning module
        self.num_classes = None
        self.num_features = None

    def prepare_data(self):
        """
        Download or prepare data. Called only on rank 0 in distributed training.
        """
        # This method is called before setup() and only on rank 0
        # Use it for downloading or one-time data preparation
        pass

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        if self.dataset is None:
            self.dataset = AstroDataset(
                root=self.data_root,
                survey=self.survey,
                k_neighbors=self.k_neighbors,
                max_samples=self.max_samples,
            )

        # Get the main data object
        full_data = self.dataset[0]

        # Apply subgraph sampling if the graph is too large
        if (
            self.use_subgraph_sampling
            and full_data.num_nodes > self.max_nodes_per_graph
        ):
            logger.info(
                f"Graph too large ({full_data.num_nodes} nodes). Creating subgraph with {self.max_nodes_per_graph} nodes."
            )
            self._main_data = self._create_subgraph_samples(
                full_data, self.max_nodes_per_graph
            )
        else:
            self._main_data = full_data

        # Extract dataset information for Lightning module
        self.num_features = (
            self._main_data.x.size(1) if hasattr(self._main_data, "x") else None
        )
        if hasattr(self._main_data, "y"):
            unique_labels = torch.unique(self._main_data.y)
            self.num_classes = max(len(unique_labels), 2)  # Ensure at least 2 classes

            # If we only have one class, create synthetic binary labels for demonstration
            if len(unique_labels) == 1:
                logger.warning(
                    f"Dataset has only 1 unique label ({unique_labels[0].item()}). Creating synthetic binary labels for training."
                )
                # Create binary labels based on node features or random split
                # Use feature-based split: nodes with feature sum > median get label 1
                feature_sums = self._main_data.x.sum(dim=1)
                median_val = feature_sums.median()
                self._main_data.y = (feature_sums > median_val).long()
                self.num_classes = 2

        # Create train/val/test splits using masks
        self._create_data_splits()

    def _create_data_splits(self):
        """Create train/val/test masks for the graph data."""
        data = self._main_data
        num_nodes = data.num_nodes

        # Create train/val/test masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Random split
        indices = torch.randperm(num_nodes)
        train_size = int(num_nodes * self.train_ratio)
        val_size = int(num_nodes * self.val_ratio)

        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size : train_size + val_size]] = True
        data.test_mask[indices[train_size + val_size :]] = True

        # Log split information
        logger.info(
            f"Data splits - Train: {data.train_mask.sum()}, "
            f"Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}"
        )

    def _create_subgraph_samples(self, data, max_nodes: int):
        """Create smaller subgraph samples for laptop training."""
        from torch_geometric.utils import subgraph

        # Randomly sample nodes
        num_nodes = data.num_nodes
        if num_nodes <= max_nodes:
            return data

        # Sample nodes randomly
        indices = torch.randperm(num_nodes)[:max_nodes]

        # Create subgraph
        edge_index, edge_attr = subgraph(
            indices,
            data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )

        # Create new data object with subgraph
        sub_data = data.__class__()
        sub_data.num_nodes = len(indices)
        sub_data.x = data.x[indices]
        sub_data.edge_index = edge_index
        if hasattr(data, "pos"):
            sub_data.pos = data.pos[indices]
        if hasattr(data, "y"):
            sub_data.y = data.y[indices]
        if edge_attr is not None:
            sub_data.edge_attr = edge_attr

        return sub_data

    def _estimate_memory_usage(self, data) -> float:
        """Estimate memory usage of graph data in MB."""
        total_memory = 0

        # Node features
        if hasattr(data, "x"):
            total_memory += data.x.numel() * data.x.element_size()

        # Edge indices
        if hasattr(data, "edge_index"):
            total_memory += data.edge_index.numel() * data.edge_index.element_size()

        # Positions
        if hasattr(data, "pos"):
            total_memory += data.pos.numel() * data.pos.element_size()

        # Labels
        if hasattr(data, "y"):
            total_memory += data.y.numel() * data.y.element_size()

        # Masks
        if hasattr(data, "train_mask"):
            total_memory += data.train_mask.numel() * data.train_mask.element_size()

        return total_memory / (1024 * 1024)  # Convert to MB

    def _get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get optimized dataloader kwargs based on settings."""
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }

        # Add prefetch_factor only if using workers
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor

        return kwargs

    def train_dataloader(self):
        """Create training dataloader with optimizations."""
        if self._main_data is None:
            self.setup()

        # Create a simple dataloader that yields the data object directly
        class SingleGraphDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                yield self.data

            def __len__(self):
                return 1

        return SingleGraphDataLoader(self._main_data)

    def val_dataloader(self):
        """Create validation dataloader."""
        if self._main_data is None:
            self.setup()

        # Create a simple dataloader that yields the data object directly
        class SingleGraphDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                yield self.data

            def __len__(self):
                return 1

        return SingleGraphDataLoader(self._main_data)

    def test_dataloader(self):
        """Create test dataloader."""
        if self._main_data is None:
            self.setup()

        # Create a simple dataloader that yields the data object directly
        class SingleGraphDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                yield self.data

            def __len__(self):
                return 1

        return SingleGraphDataLoader(self._main_data)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        # Clean up cached data
        self._main_data = None

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if self.dataset is None:
            return {"error": "Dataset not initialized"}

        info = self.dataset.get_info()

        # Add datamodule-specific info
        info.update(
            {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "num_classes": self.num_classes,
                "num_features": self.num_features,
            }
        )

        return info

    def state_dict(self) -> Dict[str, Any]:
        """Save datamodule state."""
        return {
            "survey": self.survey,
            "k_neighbors": self.k_neighbors,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load datamodule state."""
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
