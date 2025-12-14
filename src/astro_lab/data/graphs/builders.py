"""
Concrete Graph Builders
======================

Implementation of specific graph building strategies with state-of-the-art methods.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from astro_lab.tensors import SurveyTensorDict

from .base import BaseGraphBuilder, GraphConfig


class KNNGraphBuilder(BaseGraphBuilder):
    """K-Nearest Neighbors graph builder with optimizations."""

    def __init__(self, k_neighbors: int = 16, **kwargs):
        config = GraphConfig(method="knn", k_neighbors=k_neighbors, **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build optimized KNN graph."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Build graph based on device and size
        if coords.shape[0] > 100000 and not self.config.use_gpu_construction:
            # Use CPU-optimized method for large graphs
            edge_index = self._build_knn_cpu_optimized(coords)
        else:
            # Use PyG's GPU-accelerated method
            edge_index = knn_graph(
                coords,
                k=self.config.k_neighbors,
                batch=None,
                loop=self.config.self_loops,
                flow="source_to_target",
                cosine=False,
                num_workers=self.config.num_workers,
            )

        # Make undirected if requested
        if not self.config.directed:
            edge_index = to_undirected(edge_index, num_nodes=coords.shape[0])

        # Create PyG Data object
        data = self.create_data_object(features, edge_index, coords)

        # Add metadata
        data.graph_type = "knn"
        data.k_neighbors = self.config.k_neighbors

        return data

    def _build_knn_cpu_optimized(self, coords: torch.Tensor) -> torch.Tensor:
        """CPU-optimized KNN using sklearn for large graphs."""
        coords_numpy = coords.cpu().numpy()
        
        # Use sklearn's optimized implementation
        nbrs = NearestNeighbors(
            n_neighbors=self.config.k_neighbors + 1,  # +1 for self
            algorithm='auto',  # Automatically choose best algorithm
            metric='euclidean',
            n_jobs=self.config.num_workers if self.config.num_workers > 0 else -1
        )
        
        nbrs.fit(coords_numpy)
        distances, indices = nbrs.kneighbors(coords_numpy)
        
        # Remove self-connections if not needed
        if not self.config.self_loops:
            indices = indices[:, 1:]  # Skip first neighbor (self)
        
        # Convert to edge_index format
        n_nodes = coords.shape[0]
        k = indices.shape[1]
        
        source_nodes = torch.arange(n_nodes).unsqueeze(1).expand(-1, k).flatten()
        target_nodes = torch.from_numpy(indices).flatten()
        
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        
        return edge_index.to(self.device)


class RadiusGraphBuilder(BaseGraphBuilder):
    """Radius-based graph builder with adaptive radius."""

    def __init__(self, radius: float = 1.0, **kwargs):
        config = GraphConfig(method="radius", radius=radius, **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build radius graph with adaptive features."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Adaptive radius based on local density
        if hasattr(self.config, "adaptive_radius") and self.config.adaptive_radius:
            radius = self._compute_adaptive_radius(coords)
        else:
            radius = self.config.radius

        # Create radius graph
        edge_index = radius_graph(
            coords,
            r=radius,
            batch=None,
            loop=self.config.self_loops,
            max_num_neighbors=self.config.k_neighbors * 2,  # Limit for efficiency
            flow="source_to_target",
            num_workers=self.config.num_workers,
        )

        # Make undirected if requested
        if not self.config.directed:
            edge_index = to_undirected(edge_index, num_nodes=coords.shape[0])

        # Create PyG Data object
        data = self.create_data_object(features, edge_index, coords)

        # Add metadata
        data.graph_type = "radius"
        data.radius = radius if isinstance(radius, float) else radius.mean().item()

        return data

    def _compute_adaptive_radius(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute adaptive radius based on local density (memory efficient)."""
        # Find k-th nearest neighbor distance for each point
        k = min(10, coords.shape[0] - 1)
        
        # For large datasets, use chunked computation to avoid memory issues
        n_points = coords.shape[0]
        chunk_size = min(1000, n_points)  # Process 1000 points at a time
        
        if n_points > 10000:
            # Chunked computation for large datasets
            kth_dists_list = []
            for i in range(0, n_points, chunk_size):
                end_i = min(i + chunk_size, n_points)
                chunk_coords = coords[i:end_i]
                
                # Compute distances only for this chunk
                chunk_dists = torch.cdist(chunk_coords, coords)
                chunk_kth_dists, _ = torch.kthvalue(chunk_dists, k + 1, dim=1)
                kth_dists_list.append(chunk_kth_dists)
                
            kth_dists = torch.cat(kth_dists_list)
        else:
            # For small datasets, compute directly
            dists = torch.cdist(coords, coords)
            kth_dists, _ = torch.kthvalue(dists, k + 1, dim=1)  # +1 to skip self
        
        # Use median of k-th distances as base radius
        median_dist = kth_dists.median()
        
        # Scale by configuration
        radius = median_dist * self.config.radius
        
        self.logger.info(f"Adaptive radius: {radius:.4f} (median k-NN dist: {median_dist:.4f})")
        
        return radius


class AstronomicalGraphBuilder(BaseGraphBuilder):
    """Astronomical graph builder with domain-specific optimizations."""

    def __init__(self, **kwargs):
        config = GraphConfig(method="astronomical", **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build astronomical graph with specialized logic."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Determine graph construction method based on data
        if self._is_survey_data(survey_tensor):
            edge_index = self._build_survey_graph(coords, survey_tensor)
        elif self._is_simulation_data(survey_tensor):
            edge_index = self._build_simulation_graph(coords, survey_tensor)
        else:
            # Default to optimized KNN
            edge_index = self._build_astronomical_knn(coords)

        # Make undirected if requested
        if not self.config.directed:
            edge_index = to_undirected(edge_index, num_nodes=coords.shape[0])

        # Create PyG Data object
        data = self.create_data_object(features, edge_index, coords)

        # Add astronomical metadata
        data.graph_type = "astronomical"
        data.coordinate_system = self.config.coordinate_system
        data.use_3d = self.config.use_3d_coordinates

        # Add survey metadata if available
        if hasattr(survey_tensor, "meta") and survey_tensor.meta:
            data.survey_name = survey_tensor.meta.get("survey_name", "unknown")
            data.data_release = survey_tensor.meta.get("data_release", "unknown")

        return data

    def _is_survey_data(self, survey_tensor: SurveyTensorDict) -> bool:
        """Check if data is from astronomical survey."""
        if hasattr(survey_tensor, "survey_name"):
            return survey_tensor.survey_name in ["gaia", "sdss", "lsst", "euclid"]
        return False

    def _is_simulation_data(self, survey_tensor: SurveyTensorDict) -> bool:
        """Check if data is from simulation."""
        if hasattr(survey_tensor, "survey_name"):
            return survey_tensor.survey_name in ["tng50", "eagle", "illustris"]
        return False

    def _build_survey_graph(
        self, coords: torch.Tensor, survey_tensor: SurveyTensorDict
    ) -> torch.Tensor:
        """Build graph optimized for survey data."""
        survey_name = getattr(survey_tensor, "survey_name", "unknown")
        
        if survey_name == "gaia" and coords.shape[1] >= 2:
            # Use angular distance for Gaia
            return self._build_angular_graph(coords)
        elif survey_name in ["sdss", "lsst"] and coords.shape[1] >= 3:
            # Use 3D distance with redshift
            return self._build_cosmological_graph(coords)
        else:
            # Default KNN
            return knn_graph(
                coords,
                k=self.config.k_neighbors,
                batch=None,
                loop=self.config.self_loops,
            )

    def _build_simulation_graph(
        self, coords: torch.Tensor, survey_tensor: SurveyTensorDict
    ) -> torch.Tensor:
        """Build graph optimized for simulation data."""
        # For simulations, use adaptive neighbor search
        return self._build_adaptive_knn(coords)

    def _build_astronomical_knn(self, coords: torch.Tensor) -> torch.Tensor:
        """Build KNN graph with astronomical optimizations."""
        # Use appropriate distance metric
        if self.config.coordinate_system == "spherical":
            return self._build_angular_graph(coords)
        else:
            return knn_graph(
                coords,
                k=self.config.k_neighbors,
                batch=None,
                loop=self.config.self_loops,
            )

    def _build_angular_graph(self, coords: torch.Tensor) -> torch.Tensor:
        """Build graph using angular distance (optimized)."""
        n_nodes = coords.shape[0]
        k = self.config.k_neighbors
        
        # Convert to unit vectors for efficient computation
        if coords.shape[1] == 2:  # RA, Dec
            ra, dec = coords[:, 0], coords[:, 1]
            x = torch.cos(dec) * torch.cos(ra)
            y = torch.cos(dec) * torch.sin(ra)
            z = torch.sin(dec)
            unit_vectors = torch.stack([x, y, z], dim=1)
        else:  # Already 3D
            norms = torch.norm(coords, dim=1, keepdim=True)
            unit_vectors = coords / (norms + 1e-8)
        
        # Compute cosine similarity (dot product of unit vectors)
        cos_sim = torch.mm(unit_vectors, unit_vectors.t())
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # Angular distance
        angular_dist = torch.acos(cos_sim)
        
        # Find k nearest neighbors
        _, indices = torch.topk(angular_dist, k + 1, dim=1, largest=False)
        
        if not self.config.self_loops:
            indices = indices[:, 1:]  # Remove self
        
        # Create edge index
        source_nodes = torch.arange(n_nodes).unsqueeze(1).expand(-1, indices.shape[1])
        edge_index = torch.stack([source_nodes.flatten(), indices.flatten()])
        
        return edge_index.to(self.device)

    def _build_cosmological_graph(self, coords: torch.Tensor) -> torch.Tensor:
        """Build graph considering cosmological distances."""
        # For now, use standard KNN with 3D coordinates
        # TODO: Implement proper cosmological distance metrics
        return knn_graph(
            coords,
            k=self.config.k_neighbors,
            batch=None,
            loop=self.config.self_loops,
        )

    def _build_adaptive_knn(self, coords: torch.Tensor) -> torch.Tensor:
        """Build KNN with adaptive k based on local density (memory efficient)."""
        n_nodes = coords.shape[0]
        
        # Find distance to 10th nearest neighbor as density proxy
        k_density = min(10, n_nodes - 1)
        max_k = getattr(self.config, 'k_max', self.config.k_neighbors * 2)
        
        # For large datasets, compute in chunks to save memory
        chunk_size = min(1000, n_nodes)
        
        if n_nodes > 10000:
            # Chunked computation
            density_dists_list = []
            for i in range(0, n_nodes, chunk_size):
                end_i = min(i + chunk_size, n_nodes)
                chunk_coords = coords[i:end_i]
                chunk_dists = torch.cdist(chunk_coords, coords)
                chunk_density_dists, _ = torch.kthvalue(chunk_dists, k_density + 1, dim=1)
                density_dists_list.append(chunk_density_dists)
            density_dists = torch.cat(density_dists_list)
        else:
            # Compute full distance matrix for small datasets
            dists = torch.cdist(coords, coords)
            density_dists, _ = torch.kthvalue(dists, k_density + 1, dim=1)
        
        # Normalize densities
        density_scores = 1.0 / (density_dists + 1e-8)
        density_scores = density_scores / density_scores.max()
        
        # Adaptive k: more neighbors in dense regions
        k_min = getattr(self.config, 'k_min', self.config.k_neighbors // 2)
        adaptive_k = (
            k_min + 
            (max_k - k_min) * density_scores
        ).long()
        
        # Build graph with varying k - vectorized approach
        # Use maximum k for all nodes, then filter
        max_k_actual = adaptive_k.max().item()
        
        # Compute distances for all nodes in chunks
        edge_list = []
        for i in range(0, n_nodes, chunk_size):
            end_i = min(i + chunk_size, n_nodes)
            chunk_coords = coords[i:end_i]
            chunk_dists = torch.cdist(chunk_coords, coords)
            
            for j, (dist_row, k_i) in enumerate(zip(chunk_dists, adaptive_k[i:end_i])):
                node_idx = i + j
                k_i_val = k_i.item()
                _, neighbors = torch.topk(dist_row, k_i_val + 1, largest=False)
                
                if not self.config.self_loops:
                    neighbors = neighbors[1:]  # Skip self
                
                sources = torch.full((len(neighbors),), node_idx, device=coords.device)
                edge_list.append(torch.stack([sources, neighbors]))
        
        edge_index = torch.cat(edge_list, dim=1)
        
        return edge_index.to(self.device)


class MultiScaleGraphBuilder(BaseGraphBuilder):
    """Multi-scale graph builder for hierarchical representations."""

    def __init__(self, scales: List[int] = None, **kwargs):
        scales = scales or [8, 16, 32]
        config = GraphConfig(
            method="multiscale",
            scales=scales,
            use_multiscale=True,
            **kwargs
        )
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build multi-scale graph."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Build graphs at multiple scales
        all_edges = []
        edge_attrs = []
        
        for scale_idx, k in enumerate(self.config.scales):
            # Build graph at this scale
            edge_index_scale = knn_graph(
                coords,
                k=k,
                batch=None,
                loop=False,
                flow="source_to_target",
            )
            
            # Add scale attribute
            scale_attr = torch.full(
                (edge_index_scale.shape[1],),
                scale_idx,
                dtype=torch.long,
                device=self.device
            )
            
            all_edges.append(edge_index_scale)
            edge_attrs.append(scale_attr)
        
        # Combine all scales
        edge_index = torch.cat(all_edges, dim=1)
        edge_attr = torch.cat(edge_attrs)
        
        # Remove duplicate edges, keeping lowest scale
        edge_index, edge_attr = self._remove_duplicate_edges(edge_index, edge_attr)
        
        # Make undirected if requested
        if not self.config.directed:
            edge_index = to_undirected(edge_index, num_nodes=coords.shape[0])
            # Duplicate edge attributes for undirected edges
            edge_attr = torch.cat([edge_attr, edge_attr])
        
        # Create PyG Data object
        data = self.create_data_object(
            features, edge_index, coords,
            edge_attr=edge_attr
        )
        
        # Add metadata
        data.graph_type = "multiscale"
        data.scales = self.config.scales
        data.num_scales = len(self.config.scales)
        
        return data

    def _remove_duplicate_edges(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove duplicate edges, keeping those with lowest scale."""
        # Create unique edge identifier
        num_nodes = edge_index.max() + 1
        edge_ids = edge_index[0] * num_nodes + edge_index[1]
        
        # Sort by edge_id and scale
        sort_idx = torch.argsort(edge_ids * len(self.config.scales) + edge_attr)
        edge_index = edge_index[:, sort_idx]
        edge_attr = edge_attr[sort_idx]
        edge_ids = edge_ids[sort_idx]
        
        # Find unique edges
        unique_mask = torch.cat([
            torch.tensor([True], device=edge_ids.device),
            edge_ids[1:] != edge_ids[:-1]
        ])
        
        return edge_index[:, unique_mask], edge_attr[unique_mask]


class AdaptiveGraphBuilder(BaseGraphBuilder):
    """Adaptive graph builder that adjusts connectivity based on local structure."""

    def __init__(self, **kwargs):
        config = GraphConfig(method="adaptive", **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build adaptive graph based on local density and features."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Compute local structure metrics
        density_scores = self._compute_local_density(coords)
        feature_similarity = self._compute_feature_similarity(features)
        
        # Adaptive connectivity
        edge_index = self._build_adaptive_edges(
            coords, features, density_scores, feature_similarity
        )
        
        # Create PyG Data object
        data = self.create_data_object(features, edge_index, coords)
        
        # Add metadata
        data.graph_type = "adaptive"
        data.density_scores = density_scores
        
        return data

    def _compute_local_density(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute local density for each point (memory efficient)."""
        k = min(20, coords.shape[0] - 1)
        n_nodes = coords.shape[0]
        chunk_size = min(1000, n_nodes)
        
        if n_nodes > 10000:
            # Chunked computation for large datasets
            kth_dists_list = []
            for i in range(0, n_nodes, chunk_size):
                end_i = min(i + chunk_size, n_nodes)
                chunk_coords = coords[i:end_i]
                chunk_dists = torch.cdist(chunk_coords, coords)
                chunk_kth_dists, _ = torch.kthvalue(chunk_dists, k + 1, dim=1)
                kth_dists_list.append(chunk_kth_dists)
            kth_dists = torch.cat(kth_dists_list)
        else:
            dists = torch.cdist(coords, coords)
            kth_dists, _ = torch.kthvalue(dists, k + 1, dim=1)
        
        # Density is inverse of average distance to k neighbors
        density = 1.0 / (kth_dists + 1e-8)
        
        # Normalize
        density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        
        return density

    def _compute_feature_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute feature-based similarity matrix."""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Cosine similarity
        similarity = torch.mm(features_norm, features_norm.t())
        
        return similarity

    def _build_adaptive_edges(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        density_scores: torch.Tensor,
        feature_similarity: torch.Tensor,
    ) -> torch.Tensor:
        """Build edges adaptively based on multiple criteria (memory efficient)."""
        n_nodes = coords.shape[0]
        chunk_size = min(1000, n_nodes)
        
        # Adaptive k based on density
        k_base = self.config.k_neighbors
        k_min = getattr(self.config, 'k_min', k_base // 2)
        k_max = getattr(self.config, 'k_max', k_base * 2)
        adaptive_k = (k_base * (0.5 + density_scores)).long()
        adaptive_k = torch.clamp(adaptive_k, k_min, k_max)
        
        # Combined distance metric weights
        alpha = 0.7  # Weight for spatial distance
        beta = 0.3   # Weight for feature similarity
        
        # Build edges in chunks to avoid full distance matrix
        edge_list = []
        for i in range(0, n_nodes, chunk_size):
            end_i = min(i + chunk_size, n_nodes)
            chunk_coords = coords[i:end_i]
            
            # Compute spatial distances for chunk
            spatial_dists = torch.cdist(chunk_coords, coords)
            
            # Get feature similarity for chunk
            chunk_similarity = feature_similarity[i:end_i, :]
            
            # Combined distance metric
            combined_dists = alpha * spatial_dists - beta * chunk_similarity
            
            # Build edges for each node in chunk
            for j, (dist_row, k_i) in enumerate(zip(combined_dists, adaptive_k[i:end_i])):
                node_idx = i + j
                k_i_val = k_i.item()
                _, neighbors = torch.topk(dist_row, k_i_val + 1, largest=False)
                
                if not self.config.self_loops:
                    neighbors = neighbors[1:]
                
                sources = torch.full((len(neighbors),), node_idx, device=coords.device)
                edge_list.append(torch.stack([sources, neighbors]))
        
        edge_index = torch.cat(edge_list, dim=1)
        
        return edge_index.to(self.device)


class HeterogeneousGraphBuilder(BaseGraphBuilder):
    """Builder for heterogeneous graphs with multiple node and edge types."""

    def __init__(self, **kwargs):
        config = GraphConfig(method="heterogeneous", use_hetero=True, **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> HeteroData:
        """Build heterogeneous graph."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Classify objects into types
        node_types, node_masks = self._classify_objects(survey_tensor, features)
        
        # Split features by node type
        node_features = {}
        node_coords = {}
        
        for node_type, mask in node_masks.items():
            if mask.any():
                node_features[node_type] = features[mask]
                node_coords[node_type] = coords[mask]
        
        # Build edges between different node types
        edge_indices = self._build_hetero_edges(node_masks, coords)
        
        # Create heterogeneous data object
        data = self.create_hetero_data_object(
            node_features, edge_indices, node_coords
        )
        
        # Add metadata
        data.graph_type = "heterogeneous"
        data.node_types = list(node_types)
        
        return data

    def _classify_objects(
        self, survey_tensor: SurveyTensorDict, features: torch.Tensor
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        """Classify objects into different types."""
        n_objects = features.shape[0]
        
        # Simple classification based on survey type
        # TODO: Implement proper classification based on features
        
        if hasattr(survey_tensor, "survey_name"):
            if survey_tensor.survey_name == "gaia":
                # Classify stars vs other objects
                # Placeholder: random classification
                star_mask = torch.rand(n_objects) > 0.2
                
                node_types = ["star", "other"]
                node_masks = {
                    "star": star_mask,
                    "other": ~star_mask
                }
            
            elif survey_tensor.survey_name in ["sdss", "nsa"]:
                # Classify galaxies by type
                # Placeholder: random classification
                elliptical_mask = torch.rand(n_objects) < 0.3
                spiral_mask = torch.rand(n_objects) < 0.5
                spiral_mask &= ~elliptical_mask
                irregular_mask = ~(elliptical_mask | spiral_mask)
                
                node_types = ["elliptical", "spiral", "irregular"]
                node_masks = {
                    "elliptical": elliptical_mask,
                    "spiral": spiral_mask,
                    "irregular": irregular_mask
                }
            
            else:
                # Default: single type
                node_types = ["object"]
                node_masks = {"object": torch.ones(n_objects, dtype=torch.bool)}
        
        else:
            # Default: single type
            node_types = ["object"]
            node_masks = {"object": torch.ones(n_objects, dtype=torch.bool)}
        
        return node_types, node_masks

    def _build_hetero_edges(
        self, node_masks: Dict[str, torch.Tensor], coords: torch.Tensor
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Build edges between different node types."""
        edge_indices = {}
        
        # Build edges within each node type
        for node_type, mask in node_masks.items():
            if mask.sum() > 1:  # Need at least 2 nodes
                type_coords = coords[mask]
                type_indices = torch.where(mask)[0]
                
                # KNN within type
                edge_index_local = knn_graph(
                    type_coords,
                    k=min(self.config.k_neighbors, len(type_coords) - 1),
                    batch=None,
                    loop=self.config.self_loops,
                )
                
                # Map back to global indices
                edge_index_global = type_indices[edge_index_local]
                
                edge_type = (node_type, "similar", node_type)
                edge_indices[edge_type] = edge_index_global
        
        # Build edges between different node types
        # TODO: Implement cross-type connections based on spatial proximity
        
        return edge_indices


# Convenience functions for easy usage
def create_knn_graph(
    survey_tensor: SurveyTensorDict, k_neighbors: int = 16, **kwargs
) -> Data:
    """Create optimized KNN graph from SurveyTensorDict."""
    builder = KNNGraphBuilder(k_neighbors=k_neighbors, **kwargs)
    return builder.build(survey_tensor)


def create_radius_graph(
    survey_tensor: SurveyTensorDict, radius: float = 1.0, **kwargs
) -> Data:
    """Create radius graph with adaptive features."""
    builder = RadiusGraphBuilder(radius=radius, **kwargs)
    return builder.build(survey_tensor)


def create_astronomical_graph(survey_tensor: SurveyTensorDict, **kwargs) -> Data:
    """Create astronomical graph with domain-specific optimizations."""
    builder = AstronomicalGraphBuilder(**kwargs)
    return builder.build(survey_tensor)


def create_multiscale_graph(
    survey_tensor: SurveyTensorDict, scales: List[int] = None, **kwargs
) -> Data:
    """Create multi-scale graph for hierarchical analysis."""
    builder = MultiScaleGraphBuilder(scales=scales, **kwargs)
    return builder.build(survey_tensor)


def create_adaptive_graph(survey_tensor: SurveyTensorDict, **kwargs) -> Data:
    """Create adaptive graph based on local structure."""
    builder = AdaptiveGraphBuilder(**kwargs)
    return builder.build(survey_tensor)


def create_heterogeneous_graph(
    survey_tensor: SurveyTensorDict, **kwargs
) -> HeteroData:
    """Create heterogeneous graph with multiple node types."""
    builder = HeterogeneousGraphBuilder(**kwargs)
    return builder.build(survey_tensor)
