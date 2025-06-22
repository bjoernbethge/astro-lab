"""
Clustering Tensor for Astronomical Clustering
=============================================

Specialized tensor for astronomical clustering operations including
galaxy clusters, stellar associations, and large-scale structure.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance as distance
import torch
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans

# Optional dependencies
from sklearn.neighbors import BallTree, NearestNeighbors

from .base import AstroTensorBase


class ClusteringTensor(AstroTensorBase):
    """
    Tensor for astronomical clustering operations.

    Provides specialized clustering algorithms for:
    - Galaxy cluster detection
    - Stellar association identification
    - Large-scale structure analysis
    - Friends-of-friends linking
    - Hierarchical clustering with astronomical metrics
    """

    _metadata_fields = [
        "coordinate_system",
        "clustering_algorithms",
        "cluster_labels",
        "cluster_properties",
        "distance_metrics",
        "clustering_parameters",
        "spatial_scales",
        "astronomical_context",
    ]

    def __init__(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        coordinate_system: str = "cartesian",
        astronomical_context: str = "general",
        **kwargs,
    ):
        """
        Initialize clustering tensor.

        Args:
            positions: Spatial positions [N, 2] or [N, 3]
            features: Additional features for clustering [N, F]
            coordinate_system: 'cartesian', 'spherical', 'sky'
            astronomical_context: 'galaxies', 'stars', 'lss', 'general'
        """
        # Validate positions
        pos_tensor = torch.as_tensor(positions, dtype=torch.float32)
        if pos_tensor.dim() != 2 or pos_tensor.shape[1] not in [2, 3]:
            raise ValueError(
                f"Positions must have shape [N, 2] or [N, 3], got {pos_tensor.shape}"
            )

        # Store features if provided
        if features is not None:
            feat_tensor = torch.as_tensor(features, dtype=torch.float32)
            if feat_tensor.shape[0] != pos_tensor.shape[0]:
                raise ValueError(
                    "Features and positions must have same number of objects"
                )
            # Combine positions and features
            data = torch.cat([pos_tensor, feat_tensor], dim=1)
        else:
            data = pos_tensor

        # Initialize metadata
        metadata = {
            "coordinate_system": coordinate_system,
            "astronomical_context": astronomical_context,
            "clustering_algorithms": {},
            "cluster_labels": {},
            "cluster_properties": {},
            "distance_metrics": {},
            "clustering_parameters": {},
            "spatial_scales": {},
            "n_spatial_dims": pos_tensor.shape[1],
            "n_features": feat_tensor.shape[1] if features is not None else 0,
            "tensor_type": "clustering",
        }
        metadata.update(kwargs)

        super().__init__(data, **metadata)

    @property
    def positions(self) -> torch.Tensor:
        """Get spatial positions."""
        n_spatial = self.get_metadata("n_spatial_dims", 3)
        return self._data[:, :n_spatial]

    @property
    def features(self) -> Optional[torch.Tensor]:
        """Get additional features."""
        n_spatial = self.get_metadata("n_spatial_dims", 3)
        n_features = self.get_metadata("n_features", 0)
        if n_features > 0:
            return self._data[:, n_spatial:]
        return None

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return self._data.shape[0]

    def dbscan_clustering(
        self,
        eps: Optional[float] = None,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm_name: str = "dbscan",
    ) -> torch.Tensor:
        """
        Perform DBSCAN clustering with astronomical parameters.

        Args:
            eps: Maximum distance between samples (auto-estimated if None)
            min_samples: Minimum samples in neighborhood
            metric: Distance metric
            algorithm_name: Name for storing results

        Returns:
            Cluster labels (-1 for noise)
        """

        # Auto-estimate eps if not provided
        if eps is None:
            eps = self._estimate_eps(min_samples)

        # Use positions for clustering
        positions = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            positions = self._preprocess_spherical_coordinates(positions)

        # Perform DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(positions)

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric,
                "n_clusters": len(np.unique(labels[labels >= 0])),
                "n_noise": (labels == -1).sum(),
            },
        )

        return labels_tensor

    def hdbscan_clustering(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        algorithm_name: str = "hdbscan",
    ) -> torch.Tensor:
        """
        Perform HDBSCAN clustering for hierarchical density-based clustering.

        Args:
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples in neighborhood
            algorithm_name: Name for storing results

        Returns:
            Cluster labels (-1 for noise)
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan package required for HDBSCAN clustering")

        positions = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            positions = self._preprocess_spherical_coordinates(positions)

        # Perform HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        labels = clusterer.fit_predict(positions)

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": len(np.unique(labels[labels >= 0])),
                "n_noise": (labels == -1).sum(),
                "cluster_persistence": clusterer.cluster_persistence_
                if hasattr(clusterer, "cluster_persistence_")
                else None,
            },
        )

        return labels_tensor

    def friends_of_friends(
        self,
        linking_length: float,
        min_group_size: int = 2,
        algorithm_name: str = "fof",
    ) -> torch.Tensor:
        """
        Friends-of-Friends clustering for large-scale structure.

        Args:
            linking_length: Maximum linking distance
            min_group_size: Minimum group size
            algorithm_name: Name for storing results

        Returns:
            Group labels (-1 for isolated objects)
        """

        positions = self.positions.cpu().numpy()

        # Build neighbor graph
        nbrs = NearestNeighbors(radius=linking_length)
        nbrs.fit(positions)

        # Find all pairs within linking length
        distances, indices = nbrs.radius_neighbors(positions)

        # Build connected components (friends-of-friends groups)
        n_objects = len(positions)
        labels = np.full(n_objects, -1, dtype=int)
        current_label = 0

        for i in range(n_objects):
            if labels[i] == -1:  # Unassigned object
                # Start new group
                group = set([i])
                queue = [i]

                while queue:
                    current = queue.pop(0)
                    neighbors = indices[current]

                    for neighbor in neighbors:
                        if neighbor != current and labels[neighbor] == -1:
                            labels[neighbor] = current_label
                            group.add(neighbor)
                            queue.append(neighbor)

                # Assign label if group is large enough
                if len(group) >= min_group_size:
                    for obj in group:
                        labels[obj] = current_label
                    current_label += 1
                else:
                    # Mark as noise
                    for obj in group:
                        labels[obj] = -1

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "linking_length": linking_length,
                "min_group_size": min_group_size,
                "n_groups": len(np.unique(labels[labels >= 0])),
                "n_isolated": (labels == -1).sum(),
            },
        )

        return labels_tensor

    def hierarchical_clustering(
        self,
        n_clusters: int,
        linkage: str = "ward",
        distance_threshold: Optional[float] = None,
        algorithm_name: str = "hierarchical",
    ) -> torch.Tensor:
        """
        Hierarchical clustering with astronomical distance metrics.

        Args:
            n_clusters: Number of clusters (ignored if distance_threshold set)
            linkage: Linkage criterion
            distance_threshold: Distance threshold for automatic cluster number
            algorithm_name: Name for storing results

        Returns:
            Cluster labels
        """

        # Use both positions and features if available
        if self.features is not None:
            data = self._data.cpu().numpy()
        else:
            data = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            pos_data = self.positions.cpu().numpy()
            pos_data = self._preprocess_spherical_coordinates(pos_data)
            if self.features is not None:
                feat_data = self.features.cpu().numpy()
                data = np.concatenate([pos_data, feat_data], axis=1)
            else:
                data = pos_data

        # Perform hierarchical clustering
        if distance_threshold is not None:
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold, linkage=linkage
            )
        else:
            from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

        labels = clustering.fit_predict(data)

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "n_clusters": len(np.unique(labels)),
                "linkage": linkage,
                "distance_threshold": distance_threshold,
            },
        )

        return labels_tensor

    def galaxy_cluster_detection(
        self,
        richness_threshold: int = 10,
        radius_mpc: float = 2.0,
        algorithm_name: str = "galaxy_clusters",
    ) -> torch.Tensor:
        """
        Detect galaxy clusters using astronomical criteria.

        Args:
            richness_threshold: Minimum number of galaxies in cluster
            radius_mpc: Search radius in Mpc
            algorithm_name: Name for storing results

        Returns:
            Cluster labels
        """
        # Use DBSCAN with astronomical parameters
        eps = radius_mpc  # Mpc
        min_samples = richness_threshold

        labels = self.dbscan_clustering(
            eps=eps, min_samples=min_samples, algorithm_name=algorithm_name
        )

        # Calculate cluster properties
        cluster_props = self._calculate_cluster_properties(labels)

        # Update metadata with cluster properties using keyword arguments
        metadata_key = f"{algorithm_name}_properties"
        metadata_update = {metadata_key: cluster_props}
        self.update_metadata(**metadata_update)

        return labels

    def stellar_association_detection(
        self,
        max_separation_pc: float = 50.0,
        min_members: int = 5,
        algorithm_name: str = "stellar_associations",
    ) -> torch.Tensor:
        """
        Detect stellar associations and moving groups.

        Args:
            max_separation_pc: Maximum separation in parsecs
            min_members: Minimum number of members
            algorithm_name: Name for storing results

        Returns:
            Association labels
        """
        # Convert separation to appropriate units
        if self.get_metadata("coordinate_system") == "spherical":
            # Assume positions are in degrees, convert pc to degrees at typical distance
            typical_distance_pc = 100.0  # pc
            eps_deg = np.degrees(max_separation_pc / typical_distance_pc)
            eps = eps_deg
        else:
            eps = max_separation_pc

        labels = self.dbscan_clustering(
            eps=eps, min_samples=min_members, algorithm_name=algorithm_name
        )

        return labels

    def get_cluster_statistics(self, algorithm_name: str) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        if algorithm_name not in self.get_metadata("cluster_labels", {}):
            raise ValueError(f"No clustering results for algorithm: {algorithm_name}")

        labels = self.get_metadata("cluster_labels")[algorithm_name]
        params = self.get_metadata("clustering_parameters")[algorithm_name]

        unique_labels = torch.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = (labels == -1).sum().item()

        stats = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": n_noise / len(labels),
            "parameters": params,
        }

        # Cluster size distribution
        if n_clusters > 0:
            cluster_sizes = []
            for label in unique_labels[unique_labels >= 0]:
                cluster_sizes.append((labels == label).sum().item())

            stats.update(
                {
                    "cluster_sizes": cluster_sizes,
                    "mean_cluster_size": np.mean(cluster_sizes),
                    "std_cluster_size": np.std(cluster_sizes),
                    "min_cluster_size": min(cluster_sizes),
                    "max_cluster_size": max(cluster_sizes),
                }
            )

        return stats

    # Helper methods
    def _estimate_eps(self, min_samples: int) -> float:
        """Estimate optimal eps parameter for DBSCAN."""
        positions = self.positions.cpu().numpy()

        # Use k-distance graph method
        nbrs = NearestNeighbors(n_neighbors=min_samples)
        nbrs.fit(positions)
        distances, indices = nbrs.kneighbors(positions)

        # Sort distances to k-th nearest neighbor
        k_distances = np.sort(distances[:, min_samples - 1])

        # Find elbow point (simplified)
        eps = np.percentile(k_distances, 90)  # Use 90th percentile as heuristic

        return float(eps)

    def _preprocess_spherical_coordinates(self, positions: np.ndarray) -> np.ndarray:
        """Preprocess spherical coordinates for clustering."""
        if positions.shape[1] == 2:  # RA, Dec
            ra, dec = positions[:, 0], positions[:, 1]
            # Convert to Cartesian on unit sphere
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)
            return np.column_stack([x, y, z])
        elif positions.shape[1] == 3:  # RA, Dec, Distance
            ra, dec, dist = positions[:, 0], positions[:, 1], positions[:, 2]
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            x = dist * np.cos(dec_rad) * np.cos(ra_rad)
            y = dist * np.cos(dec_rad) * np.sin(ra_rad)
            z = dist * np.sin(dec_rad)
            return np.column_stack([x, y, z])
        else:
            return positions

    def _store_clustering_results(
        self, algorithm_name: str, labels: torch.Tensor, parameters: Dict[str, Any]
    ):
        """Store clustering results in metadata."""
        cluster_labels = self.get_metadata("cluster_labels", {})
        cluster_labels[algorithm_name] = labels

        clustering_params = self.get_metadata("clustering_parameters", {})
        clustering_params[algorithm_name] = parameters

        self.update_metadata(
            cluster_labels=cluster_labels, clustering_parameters=clustering_params
        )

    def _calculate_cluster_properties(
        self, labels: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """Calculate properties for each cluster."""
        properties = {}
        positions = self.positions

        unique_labels = torch.unique(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            mask = labels == label
            cluster_positions = positions[mask]

            if len(cluster_positions) > 0:
                center = cluster_positions.mean(dim=0)
                if len(cluster_positions) > 1:
                    distances = torch.norm(cluster_positions - center, dim=1)
                    radius = distances.max()
                    radius_rms = distances.std()
                else:
                    radius = 0.0
                    radius_rms = 0.0

                properties[int(label)] = {
                    "size": int(mask.sum()),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "center_z": float(center[2]) if len(center) > 2 else 0.0,
                    "radius": float(radius),
                    "radius_rms": float(radius_rms),
                }

        return properties

    def __repr__(self) -> str:
        n_algorithms = len(self.get_metadata("cluster_labels", {}))
        return (
            f"ClusteringTensor(objects={self.n_objects}, "
            f"coord_system={self.get_metadata('coordinate_system')}, "
            f"algorithms={n_algorithms})"
        )
