"""
AstroLab DataModule - Lightning DataModule for Astronomical Data
===============================================================

Lightning DataModule for astronomical datasets with TensorDict integration.
Updated for TensorDict architecture.
"""

import logging
import os
import tempfile
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
from .datasets import SurveyGraphDataset
from .utils import (
    MemoryOptimizedDataModule,
    SafePyGDataLoader,
    check_graph_consistency,
    create_optimized_dataloader,
    get_graph_statistics,
    sample_subgraph_random,
)

logger = logging.getLogger(__name__)


class AstroDataModule(L.LightningDataModule, MemoryOptimizedDataModule):
    """
    Lightning DataModule for astronomical survey data with graph neural networks.

    Optimized for laptop GPUs with intelligent memory management and pinned memory.
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
        # Memory optimization parameters
        use_smart_pinning: bool = True,  # Use intelligent pinning for PyG data
        non_blocking_transfer: bool = True,  # Use non-blocking device transfers
        target_device: Optional[str] = None,  # Target device for data transfer
        **kwargs,
    ):
        super().__init__()

        # Core parameters
        self.survey = survey
        self.data_root = data_root or tempfile.gettempdir()
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

        # Memory optimization parameters
        self.use_smart_pinning = use_smart_pinning
        self.non_blocking_transfer = non_blocking_transfer

        # Parse target device
        if target_device is None:
            self.target_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.target_device = torch.device(target_device)

        # Auto-detect optimal worker configuration
        if num_workers is None:
            if torch.cuda.is_available():
                self.num_workers = min(
                    4, torch.get_num_threads()
                )  # Conservative for GPUs
            else:
                self.num_workers = 2  # Conservative for CPU-only
        else:
            self.num_workers = max(0, num_workers)

        # Intelligent pin_memory configuration based on PyTorch best practices
        if self.batch_size == 1:
            # For single graph training (typical for GNNs)
            self.pin_memory = (
                pin_memory and torch.cuda.is_available() and self.use_smart_pinning
            )
            self.persistent_workers = True if self.num_workers > 0 else False
            self.prefetch_factor = 2 if self.num_workers > 0 else None
        else:
            # For multi-graph batching
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

        logger.info(
            f"🔧 AstroDataModule initialized: "
            f"survey={survey}, batch_size={batch_size}, num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, smart_pinning={self.use_smart_pinning}, "
            f"target_device={self.target_device}"
        )

    def prepare_data(self):
        """
        Download or prepare data. Called only on rank 0 in distributed training.
        """
        # This method is called before setup() and only on rank 0
        # Use it for downloading or one-time data preparation
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training/validation/testing.
        Checks dataset size and logs batch information for robust training.
        """
        if self.dataset is None:
            self.dataset = SurveyGraphDataset(
                root=self.data_root,
                survey=self.survey,
                k_neighbors=self.k_neighbors,
                max_samples=self.max_samples,
            )

        # Get the main data object from the InMemoryDataset
        if len(self.dataset) == 0:
            raise RuntimeError("Dataset is empty")

        # Check if dataset contains multiple graphs or just one
        if hasattr(self.dataset, "__len__"):
            num_graphs = len(self.dataset)
        else:
            num_graphs = 1
        if num_graphs == 1:
            logger.warning(
                "Dataset contains only one graph. Training will use only one batch per epoch. This may cause scheduler warnings and suboptimal training."
            )
        else:
            logger.info(f"Dataset contains {num_graphs} graphs.")

        # Calculate number of batches per epoch
        num_batches = max(1, num_graphs // self.batch_size)
        logger.info(f"Batch size: {self.batch_size}, Batches per epoch: {num_batches}")
        if num_batches == 1:
            logger.warning(
                "Only one batch per epoch will be used. Consider reducing batch_size or increasing dataset size for better training."
            )

        full_data = self.dataset._data
        if not isinstance(full_data, Data):
            raise TypeError(f"Expected Data object, got {type(full_data)}")

        # Apply subgraph sampling if the graph is too large
        if (
            self.use_subgraph_sampling
            and hasattr(full_data, "num_nodes")
            and full_data.num_nodes is not None
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

        # Verify graph consistency after setup
        if not check_graph_consistency(self._main_data):
            logger.error("❌ Graph consistency check failed after setup")
            raise RuntimeError("Graph data is inconsistent after setup")

        # Log graph statistics
        stats = get_graph_statistics(self._main_data)
        logger.info(f"📊 Graph statistics: {stats}")

        # Extract dataset information for Lightning module
        self.num_features = None
        if hasattr(self._main_data, "x") and self._main_data.x is not None:
            if self._main_data.x.dim() >= 2:
                self.num_features = self._main_data.x.size(1)
            else:
                self.num_features = 1

        if hasattr(self._main_data, "y") and self._main_data.y is not None:
            unique_labels = torch.unique(self._main_data.y)
            self.num_classes = max(len(unique_labels), 2)
            if len(unique_labels) == 1:
                logger.warning(
                    f"Dataset has only 1 unique label ({unique_labels[0].item()}). Creating synthetic binary labels for training."
                )
                if self._main_data.x.dim() >= 2:
                    feature_sums = self._main_data.x.sum(dim=1)
                else:
                    feature_sums = self._main_data.x
                median_val = feature_sums.median()
                self._main_data.y = (feature_sums > median_val).long()
                self.num_classes = 2
        else:
            logger.warning(
                "No labels found in dataset. Creating synthetic binary labels for training."
            )
            if hasattr(self._main_data, "x") and self._main_data.x is not None:
                if self._main_data.x.dim() >= 2:
                    feature_sums = self._main_data.x.sum(dim=1)
                else:
                    feature_sums = self._main_data.x
                median_val = feature_sums.median()
                self._main_data.y = (feature_sums > median_val).long()
                self.num_classes = 2
            else:
                logger.warning("No features found, creating random binary labels")
                num_nodes = (
                    self._main_data.num_nodes
                    if hasattr(self._main_data, "num_nodes")
                    else 100
                )
                self._main_data.y = torch.randint(0, 2, (num_nodes,))
                self.num_classes = 2

        self._create_data_splits()

        # Keep small datasets on CPU to avoid unnecessary transfers
        if (
            hasattr(self._main_data, "num_nodes")
            and self._main_data.num_nodes is not None
            and self._main_data.num_nodes <= 10
        ):
            self._main_data = self._main_data.cpu()
            logger.info("Small dataset detected, keeping on CPU")

    def _create_data_splits(self):
        """Create train/val/test masks for the graph data."""
        data = self._main_data

        # Get number of nodes safely
        num_nodes: int
        if hasattr(data, "num_nodes") and data.num_nodes is not None:
            num_nodes = int(data.num_nodes)
        elif hasattr(data, "x") and data.x is not None:
            num_nodes = int(data.x.size(0))
        else:
            raise RuntimeError("Cannot determine number of nodes in graph data")

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
        """Create smaller subgraph samples using utility function."""
        return sample_subgraph_random(data, max_nodes, seed=42)

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

    def train_dataloader(self):
        """Create training dataloader with optimized memory management."""
        if self._main_data is None:
            self.setup()

        train_data = self._main_data
        if train_data is None:
            raise RuntimeError("Training data is None")

        return create_optimized_dataloader(
            dataset=[train_data],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            target_device=self.target_device,
            use_smart_pinning=self.use_smart_pinning,
            non_blocking_transfer=self.non_blocking_transfer,
        )

    def val_dataloader(self):
        """Create validation dataloader with optimized memory management."""
        if self._main_data is None:
            self.setup()

        val_data = self._main_data
        if val_data is None:
            raise RuntimeError("Validation data is None")

        return create_optimized_dataloader(
            dataset=[val_data],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            target_device=self.target_device,
            use_smart_pinning=self.use_smart_pinning,
            non_blocking_transfer=self.non_blocking_transfer,
        )

    def test_dataloader(self):
        """Create test dataloader with optimized memory management."""
        if self._main_data is None:
            self.setup()

        test_data = self._main_data
        if test_data is None:
            raise RuntimeError("Test data is None")

        return create_optimized_dataloader(
            dataset=[test_data],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
            target_device=self.target_device,
            use_smart_pinning=self.use_smart_pinning,
            non_blocking_transfer=self.non_blocking_transfer,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        # Clean up cached data
        self._main_data = None

        # Zentrales Memory-Management nutzen
        from astro_lab.memory import force_comprehensive_cleanup

        force_comprehensive_cleanup()

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
                "use_smart_pinning": self.use_smart_pinning,
                "non_blocking_transfer": self.non_blocking_transfer,
                "target_device": str(self.target_device),
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
