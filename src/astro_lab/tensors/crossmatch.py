"""
CrossMatch Tensor for Catalog Cross-Matching
===========================================

Specialized tensor for cross-matching astronomical catalogs with
proper handling of uncertainties and multi-survey matching.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance as distance
import torch
from sklearn.cluster import DBSCAN, KMeans

# GPU-accelerated imports
import torch_cluster
from torch_geometric.nn import radius_graph

from .base import AstroTensorBase


class CrossMatchTensor(AstroTensorBase):
    """
    Tensor for cross-matching astronomical catalogs.

    Provides specialized cross-matching algorithms for:
    - Sky coordinate matching with uncertainties
    - Multi-survey cross-matching
    - Proper motion matching for moving objects
    - Bayesian matching with source probabilities
    - Time-dependent matching for variable sources
    """

    _metadata_fields = [
        "catalog_info",
        "matching_results",
        "matching_parameters",
        "uncertainty_models",
        "match_statistics",
        "coordinate_systems",
        "proper_motion_data",
        "time_epochs",
    ]

    def __init__(
        self,
        coords_a: Union[torch.Tensor, np.ndarray],
        coords_b: Union[torch.Tensor, np.ndarray],
        catalog_names: Tuple[str, str] = ("catalog_a", "catalog_b"),
        coordinate_system: str = "icrs",
        **kwargs,
    ):
        """
        Initialize cross-match tensor.

        Args:
            coords_a: Coordinates from first catalog [N, D]
            coords_b: Coordinates from second catalog [M, D]
            catalog_names: Names of the catalogs
            coordinate_system: Coordinate system
            **kwargs: Additional metadata
        """
        # Convert to tensors
        coords_a_tensor = torch.as_tensor(coords_a, dtype=torch.float32)
        coords_b_tensor = torch.as_tensor(coords_b, dtype=torch.float32)

        # Stack coordinates for storage
        data = torch.cat([coords_a_tensor, coords_b_tensor], dim=0)

        # Initialize base class
        super().__init__(data, **kwargs)

        # Store catalog information
        self._catalog_a_size = len(coords_a_tensor)
        self._catalog_b_size = len(coords_b_tensor)

        # Set metadata
        self._metadata.update({
            "catalog_names": catalog_names,
            "coordinate_system": coordinate_system,
            "catalog_sizes": (self._catalog_a_size, self._catalog_b_size),
            "matching_results": {},
            "match_statistics": {},
        })

    @property
    def coords_a(self) -> torch.Tensor:
        """Coordinates from first catalog."""
        return self._data[:self._catalog_a_size]

    @property
    def coords_b(self) -> torch.Tensor:
        """Coordinates from second catalog."""
        return self._data[self._catalog_a_size:]

    @property
    def catalog_names(self) -> Tuple[str, str]:
        """Names of the catalogs."""
        return self._metadata.get("catalog_names", ("catalog_a", "catalog_b"))

    def nearest_neighbor_match(
        self,
        tolerance: float = 1.0,
        max_matches: Optional[int] = None,
        algorithm_name: str = "nearest_neighbor",
    ) -> List[Dict[str, Any]]:
        """
        Find nearest neighbor matches using GPU acceleration.

        Args:
            tolerance: Maximum distance for matching
            max_matches: Maximum number of matches per source
            algorithm_name: Name for storing results

        Returns:
            List of match dictionaries
        """
        # Use GPU-accelerated k-NN search
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coords_a_gpu = self.coords_a.to(device)
        coords_b_gpu = self.coords_b.to(device)

        # Find k nearest neighbors (k=1 for nearest neighbor)
        k = 1 if max_matches is None else min(max_matches, len(coords_b_gpu))
        
        # Use torch_cluster for GPU-accelerated nearest neighbor search
        edge_index = torch_cluster.knn_graph(
            x=coords_b_gpu,
            k=k,
            loop=False,
            flow='source_to_target'
        )

        # Calculate distances
        distances = torch.norm(
            coords_b_gpu[edge_index[0]] - coords_b_gpu[edge_index[1]], 
            dim=1
        )

        # Filter by tolerance
        within_tolerance = distances <= tolerance
        filtered_edges = edge_index[:, within_tolerance]
        filtered_distances = distances[within_tolerance]

        # Convert to match format
        matches = []
        for i in range(filtered_edges.shape[1]):
            src, tgt = filtered_edges[:, i]
            matches.append({
                "index_a": int(src),
                "index_b": int(tgt),
                "distance": float(filtered_distances[i]),
            })

        # Store results
        self._store_matching_results(
            algorithm_name,
            matches,
            {
                "tolerance": tolerance,
                "max_matches": max_matches,
                "n_matches": len(matches),
                "algorithm": "gpu_nearest_neighbor",
            },
        )

        return matches

    def radius_match(
        self,
        radius: float = 1.0,
        algorithm_name: str = "radius_match",
    ) -> List[Dict[str, Any]]:
        """
        Find all matches within a radius using GPU acceleration.

        Args:
            radius: Search radius
            algorithm_name: Name for storing results

        Returns:
            List of match dictionaries
        """
        # Use GPU-accelerated radius search
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coords_a_gpu = self.coords_a.to(device)
        coords_b_gpu = self.coords_b.to(device)

        # Create radius graph
        edge_index = radius_graph(
            x=coords_b_gpu,
            r=radius,
            loop=False,
            flow='source_to_target'
        )

        # Calculate distances
        distances = torch.norm(
            coords_b_gpu[edge_index[0]] - coords_b_gpu[edge_index[1]], 
            dim=1
        )

        # Convert to match format
        matches = []
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[:, i]
            matches.append({
                "index_a": int(src),
                "index_b": int(tgt),
                "distance": float(distances[i]),
            })

        # Store results
        self._store_matching_results(
            algorithm_name,
            matches,
            {
                "radius": radius,
                "n_matches": len(matches),
                "algorithm": "gpu_radius_search",
            },
        )

        return matches

    def probabilistic_match(
        self,
        uncertainty_a: Optional[torch.Tensor] = None,
        uncertainty_b: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.95,
        algorithm_name: str = "probabilistic",
    ) -> List[Dict[str, Any]]:
        """
        Probabilistic matching with uncertainties.

        Args:
            uncertainty_a: Positional uncertainties for catalog A
            uncertainty_b: Positional uncertainties for catalog B
            confidence_threshold: Minimum confidence for matches
            algorithm_name: Name for storing results

        Returns:
            List of match dictionaries with probabilities
        """
        # Use nearest neighbor match as base
        base_matches = self.nearest_neighbor_match(
            tolerance=float('inf'),
            algorithm_name=f"{algorithm_name}_base"
        )

        # Calculate match probabilities
        matches_with_prob = []
        for match in base_matches:
            idx_a, idx_b = match["index_a"], match["index_b"]
            distance = match["distance"]

            # Get uncertainties (default to 1.0 if not provided)
            unc_a = uncertainty_a[idx_a] if uncertainty_a is not None else 1.0
            unc_b = uncertainty_b[idx_b] if uncertainty_b is not None else 1.0

            # Combined uncertainty
            combined_unc = torch.sqrt(unc_a**2 + unc_b**2)

            # Calculate probability (simplified Gaussian model)
            probability = torch.exp(-0.5 * (distance / combined_unc)**2)

            if probability >= confidence_threshold:
                match["probability"] = float(probability)
                match["uncertainty_a"] = float(unc_a)
                match["uncertainty_b"] = float(unc_b)
                matches_with_prob.append(match)

        # Store results
        self._store_matching_results(
            algorithm_name,
            matches_with_prob,
            {
                "confidence_threshold": confidence_threshold,
                "n_matches": len(matches_with_prob),
                "algorithm": "probabilistic",
            },
        )

        return matches_with_prob

    def get_match_statistics(self, algorithm_name: str) -> Dict[str, Any]:
        """Get statistics for a specific matching algorithm."""
        return self._metadata.get("match_statistics", {}).get(algorithm_name, {})

    def get_matches(self, algorithm_name: str) -> List[Dict[str, Any]]:
        """Get matches for a specific algorithm."""
        return self._metadata.get("matching_results", {}).get(algorithm_name, [])

    def list_matching_algorithms(self) -> List[str]:
        """List available matching algorithms."""
        return list(self._metadata.get("matching_results", {}).keys())

    def _store_matching_results(
        self, 
        algorithm_name: str, 
        matches: List[Dict[str, Any]], 
        statistics: Dict[str, Any]
    ) -> None:
        """Store matching results in metadata."""
        if "matching_results" not in self._metadata:
            self._metadata["matching_results"] = {}
        if "match_statistics" not in self._metadata:
            self._metadata["match_statistics"] = {}

        self._metadata["matching_results"][algorithm_name] = matches
        self._metadata["match_statistics"][algorithm_name] = statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "catalog_names": self.catalog_names,
            "catalog_sizes": (self._catalog_a_size, self._catalog_b_size),
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossMatchTensor":
        """Create from dictionary representation."""
        tensor_data = data["data"]
        catalog_sizes = data.get("catalog_sizes", (len(tensor_data) // 2, len(tensor_data) // 2))
        
        coords_a = tensor_data[:catalog_sizes[0]]
        coords_b = tensor_data[catalog_sizes[0]:]
        
        return cls(
            coords_a=coords_a,
            coords_b=coords_b,
            catalog_names=data.get("catalog_names", ("catalog_a", "catalog_b")),
            metadata=data.get("metadata", {}),
        )
