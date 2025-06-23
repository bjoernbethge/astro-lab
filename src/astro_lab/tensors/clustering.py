"""
Clustering Tensor for Astronomical Clustering
=============================================

Specialized tensor for astronomical clustering operations including
galaxy clusters, stellar associations, and large-scale structure.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors

from .base import AstroTensorBase


class ClusteringTensor(AstroTensorBase):
    """
    A tensor specialized for clustering tasks on spatial and feature data.
    """

    def __init__(self, data: Any, **kwargs: Any):
        """Initializes the ClusteringTensor."""
        kwargs.setdefault("tensor_type", "clustering")
        # Default spatial dimensions if not provided
        kwargs.setdefault("n_spatial_dims", data.shape[1] if hasattr(data, 'shape') else 0)
        super().__init__(data=data, **kwargs)

    @property
    def n_spatial_dims(self) -> int:
        return self.meta.get("n_spatial_dims", 0)

    @property
    def positions(self) -> torch.Tensor:
        """Position data."""
        if self.n_spatial_dims > 0:
            return self._data[:, : self.n_spatial_dims]
        raise ValueError("Spatial dimensions not defined.")

    @property
    def features(self) -> Optional[torch.Tensor]:
        """Feature data."""
        if self._data.shape[1] > self.n_spatial_dims:
            return self._data[:, self.n_spatial_dims :]
        return None

    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> torch.Tensor:
        """Performs DBSCAN clustering."""
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.positions.cpu().numpy())
        labels = torch.from_numpy(db.labels_).to(self._data.device)
        self.update_metadata(cluster_labels=labels)
        return labels

    def hierarchical_clustering(self, n_clusters: int) -> torch.Tensor:
        """Performs hierarchical clustering."""
        model = AgglomerativeClustering(n_clusters=n_clusters).fit(self.positions.cpu().numpy())
        labels = torch.from_numpy(model.labels_).to(self._data.device)
        self.update_metadata(cluster_labels=labels)
        return labels
        
    def friends_of_friends(self, linking_length: float) -> torch.Tensor:
        # This is a placeholder for a real friends-of-friends implementation
        # For now, we use DBSCAN as a proxy.
        return self.dbscan_clustering(eps=linking_length)

    def _estimate_eps(self, min_samples: int = 5) -> float:
        """Estimates a good value for eps for DBSCAN."""
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(self.positions.cpu().numpy())
        distances, _ = neighbors_fit.kneighbors(self.positions.cpu().numpy())
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        # Heuristic to find the "elbow"
        # This can be improved with more sophisticated methods
        return float(np.percentile(distances, 95))


    def to_dict(self) -> Dict[str, Any]:
        """Converts the tensor and its metadata to a dictionary."""
        return super().to_dict()

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "ClusteringTensor":
        """Creates a ClusteringTensor from a dictionary."""
        return cls(**data_dict)

    def __repr__(self) -> str:
        """Provides a string representation of the ClusteringTensor."""
        shape_str = f"shape={list(self.shape)}"
        coord_sys = self.meta.get("coordinate_system", "unknown")
        n_clusters = len(torch.unique(self.meta.get("cluster_labels", torch.tensor([])))) -1
        return f"ClusteringTensor(objects={self.shape[0]}, {shape_str}, coord_sys='{coord_sys}', n_clusters={n_clusters})"
