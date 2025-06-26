"""
AstroLab DataModule - Lightning DataModule for Astronomical Data
===============================================================

Lightning DataModule for astronomical datasets with TensorDict integration.
Updated for TensorDict architecture.
"""

import logging
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as L
import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.loader import DataLoader

# Removed MemoryOptimizedDataModule import
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


class AstroDataModule(L.LightningDataModule):
    """
    Lightning DataModule for astronomical survey data with graph neural networks.

    Uses PyTorch Geometric's LightningNodeData for single-graph node classification.
    """

    def __init__(
        self,
        survey: str,
        data_root: Optional[str] = None,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        batch_size: int = 1,  # Not used for node classification
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: int = 0,  # Keep simple for now
        **kwargs,
    ):
        super().__init__()

        # Core parameters
        self.survey = survey
        self.data_root = data_root or tempfile.gettempdir()
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers

        # Will be created in setup()
        self.lightning_data = None
        self.dataset = None
        self._main_data = None

        logger.info(
            f"🔧 AstroDataModule initialized: survey={survey}, "
            f"train_ratio={train_ratio}, val_ratio={val_ratio}"
        )

    def setup(self, stage: Optional[str] = None):
        """Setup datasets using PyTorch Geometric's LightningNodeData."""
        if self.lightning_data is None:
            # Create the survey dataset
            self.dataset = SurveyGraphDataset(
                root=self.data_root,
                survey=self.survey,
                k_neighbors=self.k_neighbors,
                max_samples=self.max_samples,
            )

            if len(self.dataset) == 0:
                raise RuntimeError("Dataset is empty")

            # Get the main data object
            self._main_data = self.dataset._data
            if not isinstance(self._main_data, Data):
                raise TypeError(f"Expected Data object, got {type(self._main_data)}")

            # Create synthetic labels if needed
            self._ensure_labels()

            # Create train/val/test masks
            self._create_node_splits()

            # Use LightningNodeData for node classification
            self.lightning_data = LightningNodeData(
                self._main_data,
                batch_size=1024,  # Batch size for node sampling
                num_workers=self.num_workers,
                num_neighbors=[10, 10],  # 2-hop neighborhood sampling
            )

            logger.info(f"✅ Setup complete with {self._main_data.num_nodes} nodes")

    def _ensure_labels(self):
        """Ensure the data has labels for node classification."""
        data = self._main_data

        if not hasattr(data, "y") or data.y is None:
            logger.warning("No labels found. Creating synthetic binary labels.")
            if hasattr(data, "x") and data.x is not None:
                # Use feature sums to create meaningful labels
                if data.x.dim() >= 2:
                    feature_sums = data.x.sum(dim=1)
                else:
                    feature_sums = data.x
                median_val = feature_sums.median()
                data.y = (feature_sums > median_val).long()
            else:
                # Random binary labels as fallback
                num_nodes = data.num_nodes if hasattr(data, "num_nodes") else 100
                data.y = torch.randint(0, 2, (num_nodes,))

    def _create_node_splits(self):
        """Create train/val/test node masks."""
        data = self._main_data
        num_nodes = data.num_nodes

        # Create masks
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

        logger.info(
            f"Node splits - Train: {data.train_mask.sum()}, "
            f"Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}"
        )

    def train_dataloader(self):
        """Get training dataloader from LightningNodeData."""
        if self.lightning_data is None:
            self.setup()
        return self.lightning_data.train_dataloader()

    def val_dataloader(self):
        """Get validation dataloader from LightningNodeData."""
        if self.lightning_data is None:
            self.setup()
        return self.lightning_data.val_dataloader()

    def test_dataloader(self):
        """Get test dataloader from LightningNodeData."""
        if self.lightning_data is None:
            self.setup()
        return self.lightning_data.test_dataloader()

    def predict_dataloader(self):
        """Get prediction dataloader from LightningNodeData."""
        if self.lightning_data is None:
            self.setup()
        return self.lightning_data.predict_dataloader()

    @property
    def num_classes(self) -> int:
        """Number of classes for node classification."""
        if self._main_data is not None and hasattr(self._main_data, "y"):
            return len(torch.unique(self._main_data.y))
        return 2  # Default binary classification

    @property
    def num_features(self) -> int:
        """Number of node features."""
        if self._main_data is not None and hasattr(self._main_data, "x"):
            if self._main_data.x.dim() >= 2:
                logger.info(
                    f"[AstroDataModule] num_features: {self._main_data.x.size(1)} (x.shape={self._main_data.x.shape})"
                )
                return self._main_data.x.size(1)
            else:
                logger.info(
                    f"[AstroDataModule] num_features: 1 (x.shape={self._main_data.x.shape})"
                )
                return 1
        logger.info("[AstroDataModule] num_features: 1 (no features found)")
        return 1

    def prepare_data(self):
        """
        Download or prepare data. Called only on rank 0 in distributed training.
        """
        # This method is called before setup() and only on rank 0
        # Use it for downloading or one-time data preparation
        pass

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
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load datamodule state."""
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _create_synthetic_binary_labels(self):
        """Create synthetic binary labels for the dataset."""
        data = self._main_data

        if hasattr(data, "x") and data.x is not None:
            # Use feature sums to create meaningful binary labels
            if data.x.dim() >= 2:
                feature_sums = data.x.sum(dim=1)
            else:
                feature_sums = data.x
            median_val = feature_sums.median()
            data.y = (feature_sums > median_val).long()
        else:
            # No features available - create random binary labels
            logger.warning("No features found, creating random binary labels")
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") else 100
            data.y = torch.randint(0, 2, (num_nodes,))

        self.num_classes = 2
