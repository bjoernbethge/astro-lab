"""
Point Cloud Graph Builder
========================

Specialized graph builder for astronomical point cloud data.
Consolidated from astro_gnn.surveys.preprocessing.
"""

from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from astro_lab.tensors import SurveyTensorDict

from .base import BaseGraphBuilder, GraphConfig


class PointCloudGraphBuilder(BaseGraphBuilder):
    """
    Graph builder optimized for 3D astronomical point clouds.
    
    Consolidates functionality from:
    - astro_gnn.surveys.preprocessing.create_survey_pointcloud_graph
    - Various point cloud processing methods
    """

    def __init__(
        self, 
        k_neighbors: int = 16,
        use_gpu_construction: bool = True,
        normalize_positions: bool = True,
        add_self_loops: bool = False,
        **kwargs
    ):
        config = GraphConfig(
            method="pointcloud",
            k_neighbors=k_neighbors,
            use_gpu_construction=use_gpu_construction,
            self_loops=add_self_loops,
            **kwargs
        )
        super().__init__(config)
        self.normalize_positions = normalize_positions

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build optimized point cloud graph."""
        self.validate_input(survey_tensor)

        # Extract 3D positions
        positions = self.extract_coordinates(survey_tensor)
        
        # Normalize to unit sphere if requested
        if self.normalize_positions:
            positions = self._normalize_to_unit_sphere(positions)
        
        # Extract features
        features = self.extract_features(survey_tensor)
        
        # Build spatial graph using positions
        edge_index = self._build_spatial_graph(positions)
        
        # Optional: Add feature-based edges
        if hasattr(self.config, "use_feature_edges") and self.config.use_feature_edges:
            feature_edges = self._build_feature_graph(features)
            edge_index = torch.cat([edge_index, feature_edges], dim=1)
            edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates
        
        # Create data object
        data = self.create_data_object(features, edge_index, positions)
        
        # Add point cloud specific metadata
        data.graph_type = "pointcloud"
        data.normalized_positions = self.normalize_positions
        data.original_positions = self.extract_coordinates(survey_tensor)  # Keep original
        
        return data

    def _normalize_to_unit_sphere(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize positions to unit sphere."""
        # Center positions
        center = positions.mean(dim=0, keepdim=True)
        positions_centered = positions - center
        
        # Scale to unit sphere
        max_radius = torch.norm(positions_centered, dim=1).max()
        if max_radius > 0:
            positions_normalized = positions_centered / max_radius
        else:
            positions_normalized = positions_centered
        
        return positions_normalized

    def _build_spatial_graph(self, positions: torch.Tensor) -> torch.Tensor:
        """Build graph based on spatial proximity."""
        # Use GPU-accelerated KNN if available
        if self.config.use_gpu_construction and positions.is_cuda:
            edge_index = knn_graph(
                positions,
                k=self.config.k_neighbors,
                batch=None,
                loop=self.config.self_loops,
                flow="source_to_target",
                cosine=False,
                num_workers=1,
            )
        else:
            # CPU fallback for large datasets
            edge_index = self._build_knn_cpu(positions)
        
        return edge_index

    def _build_knn_cpu(self, positions: torch.Tensor) -> torch.Tensor:
        """CPU-optimized KNN construction."""
        from sklearn.neighbors import NearestNeighbors
        
        positions_numpy = positions.cpu().numpy()
        
        nbrs = NearestNeighbors(
            n_neighbors=self.config.k_neighbors + 1,
            algorithm='ball_tree',  # Better for 3D data
            metric='euclidean',
            n_jobs=-1
        )
        
        nbrs.fit(positions_numpy)
        distances, indices = nbrs.kneighbors(positions_numpy)
        
        # Remove self-connections if not needed
        if not self.config.self_loops:
            indices = indices[:, 1:]
        
        # Convert to edge_index format
        n_nodes = positions.shape[0]
        k = indices.shape[1]
        
        source_nodes = torch.arange(n_nodes).unsqueeze(1).expand(-1, k).flatten()
        target_nodes = torch.from_numpy(indices).flatten()
        
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        
        return edge_index.to(self.device)

    def _build_feature_graph(self, features: torch.Tensor) -> torch.Tensor:
        """Build additional edges based on feature similarity."""
        # Normalize features
        features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # Compute feature similarity
        similarity = torch.mm(features_norm, features_norm.t())
        
        # Find high similarity pairs (excluding self)
        similarity.fill_diagonal_(-1)
        threshold = 0.9  # High similarity threshold
        
        high_sim_pairs = (similarity > threshold).nonzero().t()
        
        # Limit number of feature edges
        max_feature_edges = features.shape[0] * 2
        if high_sim_pairs.shape[1] > max_feature_edges:
            perm = torch.randperm(high_sim_pairs.shape[1])[:max_feature_edges]
            high_sim_pairs = high_sim_pairs[:, perm]
        
        return high_sim_pairs


class AdaptivePointCloudGraphBuilder(PointCloudGraphBuilder):
    """
    Advanced point cloud graph builder with adaptive connectivity.
    
    Adjusts graph structure based on local point density and
    astronomical properties.
    """

    def __init__(
        self,
        k_min: int = 8,
        k_max: int = 32,
        density_adaptive: bool = True,
        **kwargs
    ):
        super().__init__(k_neighbors=k_min, **kwargs)
        self.k_min = k_min
        self.k_max = k_max
        self.density_adaptive = density_adaptive

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build adaptive point cloud graph."""
        self.validate_input(survey_tensor)

        # Extract positions and features
        positions = self.extract_coordinates(survey_tensor)
        if self.normalize_positions:
            positions = self._normalize_to_unit_sphere(positions)
        
        features = self.extract_features(survey_tensor)
        
        # Compute local density
        if self.density_adaptive:
            density_scores = self._compute_local_density(positions)
            edge_index = self._build_adaptive_graph(positions, density_scores)
        else:
            edge_index = self._build_spatial_graph(positions)
        
        # Create data object
        data = self.create_data_object(features, edge_index, positions)
        
        # Add metadata
        data.graph_type = "adaptive_pointcloud"
        if self.density_adaptive:
            data.density_scores = density_scores
        
        return data

    def _compute_local_density(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute local point density for each node."""
        # Find distance to k-th nearest neighbor as density proxy
        k_density = min(20, positions.shape[0] - 1)
        
        # Compute pairwise distances
        dists = torch.cdist(positions, positions)
        
        # Get k-th smallest distance (excluding self)
        kth_dists, _ = torch.kthvalue(dists, k_density + 1, dim=1)
        
        # Density is inverse of k-th distance
        density = 1.0 / (kth_dists + 1e-8)
        
        # Normalize to [0, 1]
        density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        
        return density

    def _build_adaptive_graph(
        self, 
        positions: torch.Tensor, 
        density_scores: torch.Tensor
    ) -> torch.Tensor:
        """Build graph with adaptive connectivity based on density."""
        n_nodes = positions.shape[0]
        
        # Adaptive k: more neighbors in dense regions
        adaptive_k = (
            self.k_min + 
            (self.k_max - self.k_min) * density_scores
        ).long()
        
        # Build edges with varying k
        edge_list = []
        
        # Compute pairwise distances once
        dists = torch.cdist(positions, positions)
        
        for i in range(n_nodes):
            k_i = adaptive_k[i].item()
            
            # Find k nearest neighbors
            _, neighbors = torch.topk(dists[i], k_i + 1, largest=False)
            
            if not self.config.self_loops:
                neighbors = neighbors[1:]  # Skip self
            
            # Add edges
            sources = torch.full((len(neighbors),), i, device=self.device)
            edge_list.append(torch.stack([sources, neighbors]))
        
        edge_index = torch.cat(edge_list, dim=1)
        
        return edge_index


# Convenience functions
def create_pointcloud_graph(
    survey_tensor: SurveyTensorDict,
    k_neighbors: int = 16,
    normalize_positions: bool = True,
    **kwargs
) -> Data:
    """Create point cloud graph from survey data."""
    builder = PointCloudGraphBuilder(
        k_neighbors=k_neighbors,
        normalize_positions=normalize_positions,
        **kwargs
    )
    return builder.build(survey_tensor)


def create_adaptive_pointcloud_graph(
    survey_tensor: SurveyTensorDict,
    k_min: int = 8,
    k_max: int = 32,
    density_adaptive: bool = True,
    **kwargs
) -> Data:
    """Create adaptive point cloud graph from survey data."""
    builder = AdaptivePointCloudGraphBuilder(
        k_min=k_min,
        k_max=k_max,
        density_adaptive=density_adaptive,
        **kwargs
    )
    return builder.build(survey_tensor)
