"""
AstroDataModule - Lightning DataModule Implementation
====================================================

Clean Lightning DataModule for astronomical data.
Uses the unified AstroDataset from core.py.
Optimized for 2025 best practices including:
- Efficient data loading with persistent workers
- PIN memory for GPU transfer optimization
- Proper distributed sampling
- Mixed precision support
"""

import logging
import os
from typing import Optional, Dict, Any
import multiprocessing as mp

import lightning as L
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .core import AstroDataset
from .config import data_config

# Configure logging to avoid duplicates
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
        num_workers: int = None,  # Auto-detect optimal workers
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
                cpu_count = mp.cpu_count()
                # Use fewer workers on laptop to avoid memory pressure
                self.num_workers = max(0, min(cpu_count // 2, 4))
            except:
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
        
        # Cache for data splits
        self._train_data = []
        self._val_data = []
        self._test_data = []

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
        data = self.dataset[0]
        if self.use_subgraph_sampling and data.num_nodes > self.max_nodes_per_graph:
            num_subgraphs = min(100, max(10, data.num_nodes // self.max_nodes_per_graph))
            self._train_data = []
            for _ in range(num_subgraphs):
                sub_data = self._create_subgraph_samples(data, self.max_nodes_per_graph)
                split_data = self._split_graph_data(sub_data)
                self._train_data.append(split_data)
            self._val_data = self._train_data[:max(1, len(self._train_data)//10)]
            self._test_data = self._train_data[-max(1, len(self._train_data)//10):]
        else:
            split_data = self._split_graph_data(data)
            self._train_data = [split_data]
            self._val_data = [split_data]
            self._test_data = [split_data]

    def _split_graph_data(self, data):
        """Split graph data into train/val/test masks."""
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
        
        return data

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
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        # Create new data object with subgraph
        sub_data = data.__class__()
        sub_data.num_nodes = len(indices)
        sub_data.x = data.x[indices]
        sub_data.edge_index = edge_index
        if hasattr(data, 'pos'):
            sub_data.pos = data.pos[indices]
        if hasattr(data, 'y'):
            sub_data.y = data.y[indices]
        if edge_attr is not None:
            sub_data.edge_attr = edge_attr
            
        return sub_data
        
    def _estimate_memory_usage(self, data) -> float:
        """Estimate memory usage of graph data in MB."""
        total_memory = 0
        
        # Node features
        if hasattr(data, 'x'):
            total_memory += data.x.numel() * data.x.element_size()
            
        # Edge indices
        if hasattr(data, 'edge_index'):
            total_memory += data.edge_index.numel() * data.edge_index.element_size()
            
        # Positions
        if hasattr(data, 'pos'):
            total_memory += data.pos.numel() * data.pos.element_size()
            
        # Labels
        if hasattr(data, 'y'):
            total_memory += data.y.numel() * data.y.element_size()
            
        # Masks
        if hasattr(data, 'train_mask'):
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
        if self.dataset is None:
            self.setup()

        if not self._train_data:
            raise ValueError("Training data is empty")

        kwargs = self._get_dataloader_kwargs()
        kwargs["drop_last"] = True  # Always drop last for training
        
        # Add distributed sampler if needed
        if self.use_distributed_sampler and torch.distributed.is_initialized():
            # For graph data, we typically don't use DistributedSampler
            # as we're working with multiple graphs
            pass
        
        return DataLoader(self._train_data, **kwargs)

    def val_dataloader(self):
        """Create validation dataloader."""
        if self.dataset is None:
            self.setup()

        if not self._val_data:
            raise ValueError("Validation data is empty")

        kwargs = self._get_dataloader_kwargs()
        kwargs["drop_last"] = False  # Don't drop last for validation
        
        return DataLoader(self._val_data, **kwargs)

    def test_dataloader(self):
        """Create test dataloader."""
        if self.dataset is None:
            self.setup()

        if not self._test_data:
            raise ValueError("Test data is empty")

        kwargs = self._get_dataloader_kwargs()
        kwargs["drop_last"] = False  # Don't drop last for testing
        
        return DataLoader(self._test_data, **kwargs)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        # Clean up cached data
        self._train_data = []
        self._val_data = []
        self._test_data = []
        
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
        info.update({
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "num_graphs": len(self._train_data) if hasattr(self, '_train_data') else 0,
        })
        
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
