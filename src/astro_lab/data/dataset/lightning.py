"""Minimal universal data module for astronomical machine learning with TensorDict optimization.

Handles only splits and delegates DataLoader creation to the dataset's sampler.
Now with optimized batching using TensorDict.lazy_stack for memory efficiency.
"""

from typing import Any, List, Optional

import lightning.pytorch as pl_lightning
import torch
from tensordict import TensorDict


def tensordict_collate_fn(batch: List) -> Any:
    """Collate function that uses TensorDict.lazy_stack for efficient batching.

    This is the key optimization - lazy stacking avoids memory copies and
    enables efficient batch operations.
    """
    # Check if batch contains TensorDicts
    if batch and isinstance(batch[0], TensorDict):
        # Use lazy_stack for memory efficiency
        return TensorDict.lazy_stack(batch)
    else:
        # Fallback to standard PyG collation
        from torch_geometric.loader.dataloader import Collater

        collater = Collater(None, None)
        return collater(batch)


class AstroLabDataModule(pl_lightning.LightningDataModule):
    """
    Universal data module for all astronomical data types.
    Handles only splits and delegates DataLoader creation to the dataset's sampler.

    Now with TensorDict optimizations:
    - Efficient batching with lazy_stack
    - Pin memory support for async GPU transfer
    - Consolidation option for faster device transfers
    """

    def __init__(
        self,
        dataset,
        sampler=None,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        enable_dynamic_batching: bool = False,
        neighbor_sizes: Optional[List[int]] = None,
        use_tensordict_optimization: bool = True,  # Enable TensorDict optimizations
        pin_memory: bool = True,  # Pin memory for async GPU transfer
        persistent_workers: bool = True,  # Keep workers alive between epochs
    ):
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler or getattr(dataset, "sampler", None)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.enable_dynamic_batching = enable_dynamic_batching
        self.neighbor_sizes = neighbor_sizes
        self.use_tensordict_optimization = use_tensordict_optimization
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_indices: list = []
        self.val_indices: list = []
        self.test_indices: list = []
        self.save_hyperparameters(ignore=["dataset", "sampler"])

    def setup(self, stage: Optional[str] = None):
        """Setup the dataset and create train/val/test splits."""
        # Check if dataset is properly loaded
        if not hasattr(self.dataset, "_data") or self.dataset._data is None:
            survey_name = getattr(self.dataset, "survey_name", "unknown")
            raise RuntimeError(
                f"\n{'=' * 60}\n"
                f"ERROR: No data found for survey '{survey_name}'!\n"
                f"{'=' * 60}\n\n"
                f"The dataset has not been downloaded and preprocessed yet.\n"
                f"Please run the following commands:\n\n"
                f"1. Download the raw data:\n"
                f"   astro-lab download {survey_name}\n\n"
                f"2. Preprocess the data:\n"
                f"   astro-lab preprocess --surveys {survey_name}\n\n"
                f"3. (Optional) Process with cosmic web features:\n"
                f"   astro-lab process --surveys {survey_name}\n\n"
                f"For more options, run:\n"
                f"   astro-lab --help\n"
                f"{'=' * 60}\n"
            )

        dataset_size = len(self.dataset)

        if dataset_size == 0:
            survey_name = getattr(self.dataset, "survey_name", "unknown")
            raise ValueError(
                f"\n{'=' * 60}\n"
                f"ERROR: Dataset '{survey_name}' is empty!\n"
                f"{'=' * 60}\n\n"
                f"This can happen if:\n"
                f"1. The preprocessing failed\n"
                f"2. The data files are corrupted\n"
                f"3. The survey name is incorrect\n\n"
                f"Please check:\n"
                f"- data/processed/{survey_name}/{survey_name}.parquet exists\n"
                f"- The file contains valid data\n\n"
                f"Try re-running:\n"
                f"   astro-lab preprocess --surveys {survey_name} --force\n"
                f"{'=' * 60}\n"
            )

        # Create splits
        indices = torch.randperm(dataset_size).tolist()
        train_size = max(1, int(self.train_ratio * dataset_size))
        val_size = max(1, int(self.val_ratio * dataset_size))
        max(1, dataset_size - train_size - val_size)

        # Ensure at least one sample in each split for small datasets
        if dataset_size == 1:
            self.train_indices = [0]
            self.val_indices = [0]
            self.test_indices = [0]
        elif dataset_size == 2:
            self.train_indices = [0]
            self.val_indices = [1]
            self.test_indices = [1]
        else:
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size : train_size + val_size]
            self.test_indices = indices[train_size + val_size :]

        # Determine collate function based on optimization settings
        collate_fn = tensordict_collate_fn if self.use_tensordict_optimization else None

        # If sampler is available, use it to create dataloaders
        if self.sampler and hasattr(self.sampler, "create_dataloader"):
            # Pass optimization settings to sampler
            sampler_kwargs = {
                "indices": self.train_indices,
                "batch_size": self.batch_size,
                "shuffle": True,
                "num_workers": self.num_workers,
                "collate_fn": collate_fn,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
            }
            self._train_dataloader = self.sampler.create_dataloader(
                self.dataset, **sampler_kwargs
            )

            sampler_kwargs["indices"] = self.val_indices
            sampler_kwargs["shuffle"] = False
            self._val_dataloader = self.sampler.create_dataloader(
                self.dataset, **sampler_kwargs
            )

            sampler_kwargs["indices"] = self.test_indices
            self._test_dataloader = self.sampler.create_dataloader(
                self.dataset, **sampler_kwargs
            )
        else:
            # Handle neighbor loader case or default PyG loader
            if self.neighbor_sizes is not None:
                # Use NeighborLoader for multi-hop sampling
                from torch_geometric.loader import NeighborLoader

                # NeighborLoader doesn't support custom collate_fn directly
                # but we can wrap the output
                loader_kwargs = {
                    "num_neighbors": self.neighbor_sizes,
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers,
                    "pin_memory": self.pin_memory,
                    "persistent_workers": self.persistent_workers,
                }

                # Create node indices for each split
                self._train_dataloader = NeighborLoader(
                    self.dataset[0],  # Assuming single graph
                    input_nodes=torch.tensor(self.train_indices),
                    shuffle=True,
                    **loader_kwargs,
                )
                self._val_dataloader = NeighborLoader(
                    self.dataset[0],
                    input_nodes=torch.tensor(self.val_indices),
                    shuffle=False,
                    **loader_kwargs,
                )
                self._test_dataloader = NeighborLoader(
                    self.dataset[0],
                    input_nodes=torch.tensor(self.test_indices),
                    shuffle=False,
                    **loader_kwargs,
                )
            else:
                # Use standard PyG DataLoader with optimization
                from torch_geometric.loader import DataLoader

                train_dataset = [self.dataset[i] for i in self.train_indices]
                val_dataset = [self.dataset[i] for i in self.val_indices]
                test_dataset = [self.dataset[i] for i in self.test_indices]

                loader_kwargs = {
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers,
                    "pin_memory": self.pin_memory,
                    "persistent_workers": self.persistent_workers,
                }

                # Add custom collate function for TensorDict optimization
                if self.use_tensordict_optimization:
                    loader_kwargs["collate_fn"] = tensordict_collate_fn

                self._train_dataloader = DataLoader(
                    train_dataset, shuffle=True, **loader_kwargs
                )
                self._val_dataloader = DataLoader(
                    val_dataset, shuffle=False, **loader_kwargs
                )
                self._test_dataloader = DataLoader(
                    test_dataset, shuffle=False, **loader_kwargs
                )

        print(
            f"Dataset splits: train={len(self.train_indices)}, val={len(self.val_indices)}, test={len(self.test_indices)}"
        )

        if self.use_tensordict_optimization:
            print(
                "TensorDict optimizations enabled: lazy_stack batching, pin_memory, persistent_workers"
            )

        # Optionally create memory-mapped cache
        if (
            hasattr(self.dataset, "create_memmap_cache")
            and self.use_tensordict_optimization
            and stage == "fit"
        ):
            print("Creating memory-mapped cache for faster data loading...")
            self.dataset.create_memmap_cache(num_workers=self.num_workers)

    def train_dataloader(self):
        """Return the training DataLoader from the dataset's sampler."""
        if not hasattr(self, "_train_dataloader"):
            self.setup()
        return self._train_dataloader

    def val_dataloader(self):
        """Return the validation DataLoader from the dataset's sampler."""
        if not hasattr(self, "_val_dataloader"):
            self.setup()
        return self._val_dataloader

    def test_dataloader(self):
        """Return the test DataLoader from the dataset's sampler."""
        if not hasattr(self, "_test_dataloader"):
            self.setup()
        return self._test_dataloader

    def get_info(self):
        """Get dataset information for model initialization."""
        if hasattr(self.dataset, "get_info"):
            info = self.dataset.get_info()
            # Add optimization info
            info["use_tensordict_optimization"] = self.use_tensordict_optimization
            info["pin_memory"] = self.pin_memory
            return info
        else:
            raise AttributeError(
                "Dataset does not have get_info() method. "
                "Please ensure you're using AstroLabInMemoryDataset."
            )
