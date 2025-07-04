"""Minimal universal data module for astronomical machine learning.

Handles only splits and delegates DataLoader creation to the sampler. All data access, preprocessing, and sampling logic is handled by the respective modules.
"""

from typing import Optional

import pytorch_lightning as pl_lightning
import torch


class AstroLabDataModule(pl_lightning.LightningDataModule):
    """
    Universal data module for all astronomical data types (spatial, photometric, temporal, heterogeneous).
    Handles only splits and delegates DataLoader creation to the sampler.
    """

    def __init__(
        self,
        dataset,
        sampler,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset and create train/val/test splits."""
        dataset_size = max(1, len(self.dataset))  # Ensure at least 1 sample

        # For graph classification, split on graph level
        indices = torch.randperm(dataset_size).tolist()
        train_size = int(self.train_ratio * dataset_size)
        val_size = int(self.val_ratio * dataset_size)

        # Ensure at least one sample in each split for small datasets
        if dataset_size == 1:
            train_size = 1
            val_size = 1  # Use same sample for validation
            # Use same sample for all splits
            self.train_indices = [0]
            self.val_indices = [0]
            self.test_indices = [0]
        else:
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size : train_size + val_size]
            self.test_indices = indices[train_size + val_size :]

        # Create DataLoaders explicitly and store them
        train_subset = torch.utils.data.Subset(self.dataset, self.train_indices)
        val_subset = torch.utils.data.Subset(self.dataset, self.val_indices)
        test_subset = torch.utils.data.Subset(self.dataset, self.test_indices)

        # Use the dataset's get_loader method if available, otherwise use sampler
        if hasattr(self.dataset, "get_loader"):
            self._train_dataloader = self.dataset.get_loader(
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
            self._val_dataloader = self.dataset.get_loader(
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
            self._test_dataloader = self.dataset.get_loader(
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
        else:
            self._train_dataloader = self.sampler.create_dataloader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
            self._val_dataloader = self.sampler.create_dataloader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
            self._test_dataloader = self.sampler.create_dataloader(
                test_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

        print(
            f"[DEBUG] Created DataLoaders: train={len(self._train_dataloader)}, val={len(self._val_dataloader)}, test={len(self._test_dataloader)}"
        )

    def train_dataloader(self):
        """Return the training DataLoader from the sampler."""
        return self._train_dataloader

    def val_dataloader(self):
        """Return the validation DataLoader from the sampler."""
        return self._val_dataloader

    def test_dataloader(self):
        """Return the test DataLoader from the sampler."""
        return self._test_dataloader
