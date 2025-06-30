"""
Fast Point Cloud Dataset with Memory Optimization
================================================

State-of-the-art optimized dataset implementation using:
- Memory-mapped files for efficient data access
- GPU-accelerated processing
- Memory pinning for fast CPU-GPU transfers
- Persistent workers and prefetching
"""

import logging
import mmap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import gc

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import KNNGraph
from torch_geometric.nn.pool import fps
import torch_cluster

from astro_lab.config import get_data_config, get_survey_config
from astro_lab.data.preprocessors import get_preprocessor

logger = logging.getLogger(__name__)


class FastPointCloudDataset(Dataset):
    """
    Ultra-fast point cloud dataset with memory optimization.
    
    Key optimizations:
    - Memory-mapped data files for zero-copy access
    - Pre-computed graph structures
    - GPU-resident data when possible
    - Efficient batch loading
    """
    
    def __init__(
        self,
        root: str,
        survey: str,
        k_neighbors: int = 20,
        num_subgraphs: int = 1000,
        points_per_subgraph: int = 500,
        use_memory_map: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        use_gpu_cache: bool = True,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
        force_reload: bool = False,
        **kwargs,
    ):
        """
        Initialize fast dataset.
        
        Args:
            use_memory_map: Use memory-