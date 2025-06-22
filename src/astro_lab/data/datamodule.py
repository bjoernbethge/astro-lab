"""
AstroDataModule - Lightning DataModule Implementation
====================================================

Clean Lightning DataModule for astronomical data.
Uses the unified AstroDataset from core.py.
"""

import logging
from typing import Optional

import lightning as L
import torch
from torch_geometric.loader import DataLoader

from .core import AstroDataset

logger = logging.getLogger(__name__)


class AstroDataModule(L.LightningDataModule):
    """
    Clean Lightning DataModule for astronomical datasets.

    Handles train/val/test splits and data loading.
    Uses unified AstroDataset from core.py.
    """

    def __init__(
        self,
        survey: str,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        batch_size: int = 1,  # Graph datasets typically use batch_size=1
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: int = 0,  # Disable multiprocessing for stability
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.survey = survey
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers

        # Dataset will be created in setup()
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup dataset and create train/val/test splits."""
        if self.dataset is None:
            logger.info(f"📊 Setting up dataset for survey: {self.survey}")
            self.dataset = AstroDataset(
                survey=self.survey,
                k_neighbors=self.k_neighbors,
                max_samples=self.max_samples,
            )

        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")

        # Get the single graph data object
        data = self.dataset[0]
        if data is None:
            raise ValueError("Dataset contains None values")

        num_nodes = data.num_nodes

        # Create random node splits
        indices = torch.randperm(num_nodes)

        train_size = int(self.train_ratio * num_nodes)
        val_size = int(self.val_ratio * num_nodes)

        # Create masks for node classification
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size : train_size + val_size]] = True
        data.test_mask[indices[train_size + val_size :]] = True

        logger.info(
            f"📊 Split {self.survey}: "
            f"Train={data.train_mask.sum()}, "
            f"Val={data.val_mask.sum()}, "
            f"Test={data.test_mask.sum()}"
        )

    def train_dataloader(self):
        """Create training dataloader."""
        if self.dataset is None:
            self.setup()

        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")

        data = self.dataset[0]
        if data is None:
            raise ValueError("Dataset contains None values")

        return DataLoader(
            [data],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        if self.dataset is None:
            self.setup()

        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")

        data = self.dataset[0]
        if data is None:
            raise ValueError("Dataset contains None values")

        return DataLoader(
            [data],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        if self.dataset is None:
            self.setup()

        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")

        data = self.dataset[0]
        if data is None:
            raise ValueError("Dataset contains None values")

        return DataLoader(
            [data],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def get_info(self):
        """Get dataset information."""
        if self.dataset is None:
            return {"error": "Dataset not initialized"}
        return self.dataset.get_info()
