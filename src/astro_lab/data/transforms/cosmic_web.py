"""
Cosmic Web Transforms
====================

Transforms for cosmic web analysis and large-scale structure detection.
"""

import logging
from typing import Optional

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix
from sklearn.neighbors import KernelDensity, NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import fps
from torch_geometric.transforms import BaseTransform

logger = logging.getLogger(__name__)


class CosmicWebClassification(BaseTransform):
    """Classify cosmic web structures using density and velocity fields."""

    def __init__(
        self,
        method: str = "density",
        smoothing_scale: float = 2.0,
        threshold_params: Optional[dict] = None,
    ):
        """
        Initialize cosmic web classification.

        Args:
            method: Classification method ('density', 'tidal', 'velocity')
            smoothing_scale: Smoothing scale in Mpc
            threshold_params: Parameters for classification thresholds
        """
        self.method = method
        self.smoothing_scale = smoothing_scale
        self.threshold_params = threshold_params or {
            "void": 0.2,
            "sheet": 0.5,
            "filament": 1.0,
            "node": 2.0,
        }

    def __call__(self, data: Data) -> Data:
        """Apply cosmic web classification."""
        if not hasattr(data, "pos") or data.pos is None:
            return data

        # First compute density if not available
        if not hasattr(data, "density"):
            density_transform = DensityFieldEstimation()
            data = density_transform(data)

        # Classify based on density thresholds
        density = data.density
        n_nodes = density.shape[0]
        web_class = torch.zeros(n_nodes, dtype=torch.long)

        # Simple threshold-based classification
        web_class[density < self.threshold_params["void"]] = 0  # Void
        web_class[
            (density >= self.threshold_params["void"])
            & (density < self.threshold_params["sheet"])
        ] = 1  # Sheet
        web_class[
            (density >= self.threshold_params["sheet"])
            & (density < self.threshold_params["filament"])
        ] = 2  # Filament
        web_class[density >= self.threshold_params["node"]] = 3  # Node/Cluster

        data.cosmic_web_class = web_class

        # Add as node feature
        if hasattr(data, "x") and data.x is not None:
            # One-hot encode the classification
            one_hot = torch.nn.functional.one_hot(web_class, num_classes=4).float()
            data.x = torch.cat([data.x, one_hot], dim=1)

        logger.debug(
            f"Classified cosmic web structures: "
            f"{(web_class == 0).sum()} voids, "
            f"{(web_class == 1).sum()} sheets, "
            f"{(web_class == 2).sum()} filaments, "
            f"{(web_class == 3).sum()} nodes"
        )

        return data


class FilamentDetection(BaseTransform):
    """Detect filamentary structures using Minimum Spanning Tree."""

    def __init__(
        self,
        max_length: float = 50.0,
        min_density: float = 0.5,
        n_neighbors: int = 20,
    ):
        """
        Initialize filament detection.

        Args:
            max_length: Maximum edge length in MST (Mpc)
            min_density: Minimum density for filament points
            n_neighbors: Number of neighbors for local analysis
        """
        self.max_length = max_length
        self.min_density = min_density
        self.n_neighbors = n_neighbors

    def __call__(self, data: Data) -> Data:
        """Detect filaments using MST approach."""
        if not hasattr(data, "pos") or data.pos is None:
            return data

        try:
            # Use imported functions
            pass
        except ImportError:
            logger.warning("scipy required for filament detection")
            return data

        pos = data.pos.cpu().numpy()
        n_points = pos.shape[0]

        # For large datasets, work with a subset
        if n_points > 5000:
            # Use FPS to get representative sample
            sample_idx = fps(data.pos, ratio=5000 / n_points)
            subset_pos = pos[sample_idx.cpu().numpy()]
        else:
            subset_pos = pos
            sample_idx = np.arange(n_points)

        # Compute distance matrix
        dist_matrix = distance_matrix(subset_pos, subset_pos)

        # Build MST
        mst = minimum_spanning_tree(dist_matrix)

        # Extract edges shorter than max_length
        mst_coo = mst.tocoo()
        valid_edges = mst_coo.data < self.max_length

        # Create edge list
        edge_list = []
        for i, j, weight in zip(
            mst_coo.row[valid_edges],
            mst_coo.col[valid_edges],
            mst_coo.data[valid_edges],
        ):
            # Map back to original indices if sampled
            if n_points > 5000:
                i = sample_idx[i]
                j = sample_idx[j]
            edge_list.append([i, j])
            edge_list.append([j, i])  # Undirected

        if edge_list:
            data.filament_edges = torch.tensor(edge_list, dtype=torch.long).t()
            logger.info(f"Detected {len(edge_list) // 2} filament edges")

        return data


class DensityFieldEstimation(BaseTransform):
    """Estimate 3D density field using kernel density estimation."""

    def __init__(
        self,
        method: str = "grid",
        grid_size: int = 50,
        bandwidth: Optional[float] = None,
    ):
        """
        Initialize density field estimation.

        Args:
            method: Estimation method ('grid', 'kde', 'sph')
            grid_size: Grid resolution for binning
            bandwidth: Kernel bandwidth (auto if None)
        """
        self.method = method
        self.grid_size = grid_size
        self.bandwidth = bandwidth

    def __call__(self, data: Data) -> Data:
        """Estimate density field."""
        if not hasattr(data, "pos") or data.pos is None:
            return data

        pos = data.pos.cpu().numpy()

        if self.method == "grid":
            # Simple grid-based density estimation
            density = self._grid_density(pos)
        elif self.method == "kde":
            # Kernel density estimation
            density = self._kde_density(pos)
        else:
            # SPH-like density
            density = self._sph_density(pos)

        data.density = torch.tensor(density, dtype=torch.float32)

        # Normalize density
        data.density = data.density / data.density.mean()

        # Add as node feature
        if hasattr(data, "x") and data.x is not None:
            data.x = torch.cat([data.x, data.density.unsqueeze(1)], dim=1)

        logger.debug(f"Computed density field using {self.method} method")

        return data

    def _grid_density(self, pos: np.ndarray) -> np.ndarray:
        """Grid-based density estimation."""
        # Create 3D histogram
        H, edges = np.histogramdd(pos, bins=self.grid_size)

        # Get bin centers
        centers = []
        for edge in edges:
            centers.append((edge[:-1] + edge[1:]) / 2)

        # Interpolate back to particle positions

        interpolator = RegularGridInterpolator(
            centers, H, bounds_error=False, fill_value=0
        )

        return interpolator(pos)

    def _kde_density(self, pos: np.ndarray) -> np.ndarray:
        """Kernel density estimation."""
        try:
            # Use imported KernelDensity
            pass
        except ImportError:
            logger.warning("sklearn required for KDE, falling back to grid")
            return self._grid_density(pos)

        # Estimate bandwidth if not provided
        if self.bandwidth is None:
            # Scott's rule
            n = pos.shape[0]
            d = pos.shape[1]
            self.bandwidth = n ** (-1.0 / (d + 4))

        kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
        kde.fit(pos)

        # Evaluate at particle positions
        log_density = kde.score_samples(pos)
        return np.exp(log_density)

    def _sph_density(self, pos: np.ndarray) -> np.ndarray:
        """SPH-like density estimation using nearest neighbors."""

        # Find k nearest neighbors
        k = min(32, pos.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(pos)

        distances, indices = nbrs.kneighbors(pos)

        # Use distance to kth neighbor as smoothing length
        h = distances[:, -1]

        # Simple density estimate
        volume = (4 / 3) * np.pi * h**3
        density = k / volume

        return density


class VoidDetection(BaseTransform):
    """Detect cosmic voids in the data."""

    def __init__(self, density_threshold: float = 0.2, min_radius: float = 10.0):
        """
        Initialize void detection.

        Args:
            density_threshold: Threshold for void identification
            min_radius: Minimum void radius
        """
        self.density_threshold = density_threshold
        self.min_radius = min_radius

    def __call__(self, data: Data) -> Data:
        """Detect voids."""
        if not hasattr(data, "density"):
            logger.warning("No density field found, computing it first")
            density_transform = DensityFieldEstimation()
            data = density_transform(data)

        # Identify low-density regions
        void_mask = data.density < self.density_threshold
        data.void_mask = void_mask

        # Could add more sophisticated void finding algorithms here

        logger.info(f"Identified {void_mask.sum().item()} nodes in voids")
        return data


class HaloIdentification(BaseTransform):
    """Identify dark matter halos using Friends-of-Friends or similar algorithms."""

    def __init__(self, linking_length: float = 0.2, min_group_size: int = 10):
        """
        Initialize halo identification.

        Args:
            linking_length: Linking length for FoF algorithm
            min_group_size: Minimum number of particles in a halo
        """
        self.linking_length = linking_length
        self.min_group_size = min_group_size

    def __call__(self, data: Data) -> Data:
        """Identify halos."""
        if not hasattr(data, "pos") or data.pos is None:
            return data

        # Placeholder for FoF or other halo finding algorithms
        # Real implementation would use specialized halo finders
        logger.debug(f"Identifying halos with linking length {self.linking_length}")

        # Add halo membership as node feature
        # data.halo_id = torch.zeros(data.num_nodes, dtype=torch.long)

        return data
