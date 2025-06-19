"""
Base Dataset Classes and Utilities
==================================

Common utilities and base classes for all astronomical datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

import numpy as np
import polars as pl
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, InMemoryDataset

# Import utilities from parent module
from astro_lab.data.utils import _detect_magnitude_columns, get_data_statistics

# Note: Tensor imports are handled in individual dataset files to avoid circular imports

# Check if tensors are available at runtime
try:
    import astro_lab.tensors
    ASTRO_LAB_TENSORS_AVAILABLE = True
except ImportError:
    ASTRO_LAB_TENSORS_AVAILABLE = False


def get_device():
    """Get the best available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(tensor, device=None):
    """Move tensor to device with memory optimization."""
    if device is None:
        device = get_device()
    return tensor.to(device, non_blocking=True)


def gpu_knn_graph(coords, k_neighbors=8, max_distance=None, device=None):
    """Create k-NN graph using GPU acceleration when available."""
    if device is None:
        device = get_device()
    
    # For very large datasets, use CPU sklearn (more memory efficient)
    if len(coords) > 100000 or not torch.cuda.is_available():
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords))
        distances, indices = nbrs.kneighbors(np.radians(coords))
        return distances, indices
    
    # GPU-accelerated k-NN for smaller datasets
    try:
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        coords_rad = torch.deg2rad(coords_tensor)
        
        # Compute pairwise haversine distances on GPU
        batch_size = min(1000, len(coords))  # Process in batches to save memory
        all_distances = []
        all_indices = []
        
        for i in range(0, len(coords), batch_size):
            batch_coords = coords_rad[i:i+batch_size]
            
            # Haversine distance computation
            lat1, lon1 = batch_coords[:, 1:2], batch_coords[:, 0:1]
            lat2, lon2 = coords_rad[:, 1:2].T, coords_rad[:, 0:1].T
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
            distances = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
            
            # Get k nearest neighbors
            topk_distances, topk_indices = torch.topk(distances, k_neighbors + 1, dim=1, largest=False)
            
            all_distances.append(topk_distances.cpu().numpy())
            all_indices.append(topk_indices.cpu().numpy())
        
        distances = np.vstack(all_distances)
        indices = np.vstack(all_indices)
        
        return distances, indices
        
    except Exception as e:
        print(f"   GPU k-NN failed ({e}), falling back to CPU")
        # Fallback to CPU
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords))
        distances, indices = nbrs.kneighbors(np.radians(coords))
        return distances, indices


class AstroLabDataset(InMemoryDataset):
    """
    General astronomical dataset using PyTorch Geometric's InMemoryDataset.

    Provides easy access to astronomical catalogs for machine learning.
    Creates tensor-based datasets from magnitude columns.
    """

    def __init__(
        self,
        catalog_file: Union[str, Path],
        root: Optional[str] = None,
        magnitude_columns: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize AstroLab dataset.

        Parameters
        ----------
        catalog_file : str or Path
            Path to catalog file (parquet, CSV, or FITS)
        root : str, optional
            Root directory for processed files
        magnitude_columns : list of str, optional
            Specific magnitude columns to use
        max_samples : int, optional
            Maximum number of samples to load
        """
        self.catalog_file = Path(catalog_file)
        self.magnitude_columns = magnitude_columns
        self.max_samples = max_samples

        if root is None:
            root = f"data/processed/astrolab_{self.catalog_file.stem}"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return [self.catalog_file.name]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        suffix = f"_max{self.max_samples}" if self.max_samples else ""
        return [f"astrolab_dataset{suffix}.pt"]

    def download(self):
        """Download not needed - using local catalog file."""
        if not self.catalog_file.exists():
            raise FileNotFoundError(f"Catalog file not found: {self.catalog_file}")

    def process(self):
        """Process catalog into graph format."""
        print(f"🔄 Processing catalog: {self.catalog_file.name}")

        # Load catalog
        if self.catalog_file.suffix == ".parquet":
            df = pl.read_parquet(self.catalog_file)
        elif self.catalog_file.suffix == ".csv":
            df = pl.read_csv(self.catalog_file)
        else:
            raise ValueError(f"Unsupported file format: {self.catalog_file.suffix}")

        # Limit samples if requested
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(self.max_samples, seed=42)
            print(f"   Sampled {self.max_samples} objects from {len(df)} total")

        # Detect magnitude columns if not specified
        if self.magnitude_columns is None:
            self.magnitude_columns = _detect_magnitude_columns(df.columns)
            print(f"   Detected magnitude columns: {self.magnitude_columns}")

        # Convert to numpy for processing
        df_np = df.to_numpy()
        
        # Create simple graph with all objects as nodes
        data_list = []
        
        # Extract features (magnitudes)
        mag_indices = [df.columns.index(col) for col in self.magnitude_columns if col in df.columns]
        
        if len(mag_indices) == 0:
            raise ValueError("No valid magnitude columns found")
            
        features = df_np[:, mag_indices].astype(np.float32)
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=99.0)  # Use 99.0 for missing magnitudes
        
        # Create single large graph
        x = torch.tensor(features, dtype=torch.float32)
        
        # Create simple edge connections (each node connects to next few nodes)
        num_nodes = len(features)
        edge_index = []
        
        for i in range(num_nodes):
            for j in range(max(0, i-2), min(num_nodes, i+3)):
                if i != j:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data object
        data = Data(x=x, edge_index=edge_index)
        data.num_objects = num_nodes
        data.magnitude_columns = self.magnitude_columns
        
        data_list.append(data)

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed {len(data_list)} graphs with {num_nodes} total nodes")

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        if len(self) == 0:
            return {"error": "Dataset not loaded"}
            
        data = self[0]
        return {
            "num_graphs": len(self),
            "num_nodes_total": data.num_objects,
            "num_features": data.x.shape[1],
            "magnitude_columns": data.magnitude_columns,
            "feature_stats": {
                "mean": data.x.mean(dim=0).tolist(),
                "std": data.x.std(dim=0).tolist(),
                "min": data.x.min(dim=0)[0].tolist(),
                "max": data.x.max(dim=0)[0].tolist(),
            }
        } 