"""
Advanced Graph Building Methods
==============================

State-of-the-art graph construction techniques for astronomical data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, KMeans
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.utils import (
    add_remaining_self_loops,
    coalesce,
    remove_isolated_nodes,
    to_undirected,
)

from astro_lab.tensors import SurveyTensorDict

from .base import BaseGraphBuilder, GraphConfig


class DynamicGraphBuilder(BaseGraphBuilder):
    """Dynamic graph builder that adapts structure during training."""

    def __init__(self, initial_k: int = 16, **kwargs):
        config = GraphConfig(method="dynamic", k_neighbors=initial_k, **kwargs)
        super().__init__(config)
        self.edge_weights = None
        self.learned_edges = None

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build initial graph structure."""
        self.validate_input(survey_tensor)

        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Build initial KNN graph
        edge_index = knn_graph(
            coords,
            k=self.config.k_neighbors,
            batch=None,
            loop=self.config.self_loops,
        )

        # Initialize learnable edge weights
        n_edges = edge_index.shape[1]
        edge_weight = torch.ones(n_edges, device=self.device)

        # Create data object
        data = self.create_data_object(
            features, edge_index, coords,
            edge_weight=edge_weight
        )

        data.graph_type = "dynamic"
        data.supports_edge_learning = True

        return data

    def refine_graph(
        self, data: Data, node_embeddings: torch.Tensor, temperature: float = 1.0
    ) -> Data:
        """Refine graph structure based on learned embeddings."""
        # Compute similarity in embedding space
        embeddings_norm = F.normalize(node_embeddings, p=2, dim=1)
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())

        # Get current edges
        edge_index = data.edge_index
        n_nodes = data.num_nodes

        # Compute edge scores based on similarity
        edge_scores = similarity[edge_index[0], edge_index[1]]

        # Apply temperature scaling
        edge_probs = torch.sigmoid(edge_scores / temperature)

        # Prune low-probability edges
        keep_mask = edge_probs > 0.5
        refined_edge_index = edge_index[:, keep_mask]
        refined_edge_weight = edge_probs[keep_mask]

        # Add high-similarity edges not in original graph
        similarity_threshold = 0.8
        high_sim_mask = similarity > similarity_threshold

        # Remove self-connections and existing edges
        high_sim_mask.fill_diagonal_(False)
        for i in range(edge_index.shape[1]):
            high_sim_mask[edge_index[0, i], edge_index[1, i]] = False

        # Add new edges
        new_edges = high_sim_mask.nonzero().t()
        if new_edges.shape[1] > 0:
            # Limit number of new edges
            max_new_edges = n_nodes * 2
            if new_edges.shape[1] > max_new_edges:
                perm = torch.randperm(new_edges.shape[1])[:max_new_edges]
                new_edges = new_edges[:, perm]

            # Combine with refined edges
            refined_edge_index = torch.cat([refined_edge_index, new_edges], dim=1)
            new_weights = similarity[new_edges[0], new_edges[1]]
            refined_edge_weight = torch.cat([refined_edge_weight, new_weights])

        # Update data
        data.edge_index = refined_edge_index
        data.edge_weight = refined_edge_weight

        return data


class GraphOfGraphsBuilder(BaseGraphBuilder):
    """Build hierarchical graph-of-graphs structure."""

    def __init__(
        self,
        cluster_method: str = "kmeans",
        n_clusters: int = 10,
        **kwargs
    ):
        config = GraphConfig(method="graph_of_graphs", **kwargs)
        super().__init__(config)
        self.cluster_method = cluster_method
        self.n_clusters = n_clusters

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build hierarchical graph structure."""
        self.validate_input(survey_tensor)

        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Cluster objects
        cluster_labels = self._cluster_objects(coords, features)

        # Build subgraphs for each cluster
        subgraphs = []
        cluster_features = []

        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if mask.sum() == 0:
                continue

            # Extract cluster data
            cluster_coords = coords[mask]
            cluster_feats = features[mask]

            # Build subgraph
            if mask.sum() > 1:
                subgraph_edges = knn_graph(
                    cluster_coords,
                    k=min(self.config.k_neighbors, mask.sum() - 1),
                    batch=None,
                )
            else:
                subgraph_edges = torch.empty((2, 0), dtype=torch.long)

            subgraphs.append({
                'node_indices': torch.where(mask)[0],
                'edge_index': subgraph_edges,
                'features': cluster_feats,
                'coords': cluster_coords,
            })

            # Compute cluster-level features
            cluster_feat = self._compute_cluster_features(cluster_feats, cluster_coords)
            cluster_features.append(cluster_feat)

        # Build inter-cluster graph
        cluster_features = torch.stack(cluster_features)
        cluster_graph = knn_graph(
            cluster_features,
            k=min(5, len(cluster_features) - 1),
            batch=None,
        )

        # Combine all edges
        all_edges = []
        node_to_cluster = torch.zeros(coords.shape[0], dtype=torch.long)

        for i, subgraph in enumerate(subgraphs):
            # Map local to global indices
            global_indices = subgraph['node_indices']
            local_edges = subgraph['edge_index']

            if local_edges.shape[1] > 0:
                global_edges = global_indices[local_edges]
                all_edges.append(global_edges)

            # Store cluster assignment
            node_to_cluster[global_indices] = i

        # Add inter-cluster edges
        inter_edges = self._create_inter_cluster_edges(
            coords, node_to_cluster, cluster_graph
        )
        if inter_edges.shape[1] > 0:
            all_edges.append(inter_edges)

        # Combine all edges
        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create data object
        data = self.create_data_object(features, edge_index, coords)
        
        # Add hierarchical information
        data.cluster_labels = cluster_labels
        data.n_clusters = self.n_clusters
        data.cluster_features = cluster_features
        data.cluster_graph = cluster_graph
        data.graph_type = "hierarchical"

        return data

    def _cluster_objects(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Cluster objects using specified method."""
        # Combine spatial and feature information
        combined = torch.cat([
            coords / coords.std(dim=0, keepdim=True),
            features / features.std(dim=0, keepdim=True)
        ], dim=1).cpu().numpy()

        if self.cluster_method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels = kmeans.fit_predict(combined)
        elif self.cluster_method == "dbscan":
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(combined)
            # Handle noise points
            labels[labels == -1] = labels.max() + 1
            self.n_clusters = len(np.unique(labels))
        else:
            raise ValueError(f"Unknown cluster method: {self.cluster_method}")

        return torch.tensor(labels, device=self.device)

    def _compute_cluster_features(
        self, features: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute aggregated features for a cluster."""
        # Statistical features
        feat_mean = features.mean(dim=0)
        feat_std = features.std(dim=0)
        feat_max = features.max(dim=0)[0]
        feat_min = features.min(dim=0)[0]

        # Spatial features
        coord_mean = coords.mean(dim=0)
        coord_std = coords.std(dim=0)

        # Size feature
        size = torch.tensor([features.shape[0]], device=features.device)

        # Combine all
        cluster_feat = torch.cat([
            feat_mean, feat_std, feat_max, feat_min,
            coord_mean, coord_std, size
        ])

        return cluster_feat

    def _create_inter_cluster_edges(
        self,
        coords: torch.Tensor,
        node_to_cluster: torch.Tensor,
        cluster_graph: torch.Tensor
    ) -> torch.Tensor:
        """Create edges between clusters."""
        inter_edges = []

        # For each edge in cluster graph
        for i in range(cluster_graph.shape[1]):
            c1, c2 = cluster_graph[0, i], cluster_graph[1, i]

            # Find boundary nodes (closest pairs between clusters)
            mask1 = node_to_cluster == c1
            mask2 = node_to_cluster == c2

            if mask1.sum() > 0 and mask2.sum() > 0:
                coords1 = coords[mask1]
                coords2 = coords[mask2]
                idx1 = torch.where(mask1)[0]
                idx2 = torch.where(mask2)[0]

                # Compute pairwise distances
                dists = torch.cdist(coords1, coords2)

                # Connect k closest pairs
                k_connect = min(3, dists.shape[0], dists.shape[1])
                for _ in range(k_connect):
                    min_idx = dists.argmin()
                    i1 = min_idx // dists.shape[1]
                    i2 = min_idx % dists.shape[1]

                    inter_edges.append([idx1[i1], idx2[i2]])
                    inter_edges.append([idx2[i2], idx1[i1]])  # Bidirectional

                    # Set distance to inf to avoid reselection
                    dists[i1, i2] = float('inf')

        if inter_edges:
            return torch.tensor(inter_edges, device=self.device).t()
        else:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)


class TemporalGraphBuilder(BaseGraphBuilder):
    """Build temporal graphs for time-series astronomical data."""

    def __init__(
        self,
        temporal_window: int = 5,
        temporal_stride: int = 1,
        **kwargs
    ):
        config = GraphConfig(method="temporal", use_temporal=True, **kwargs)
        super().__init__(config)
        self.temporal_window = temporal_window
        self.temporal_stride = temporal_stride

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build temporal graph from time-series data."""
        self.validate_input(survey_tensor)

        # Check for temporal data
        if "temporal" not in survey_tensor:
            raise ValueError("SurveyTensorDict must contain 'temporal' data")

        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)
        temporal_data = survey_tensor["temporal"]

        # Build spatial edges
        spatial_edges = knn_graph(
            coords,
            k=self.config.k_neighbors,
            batch=None,
            loop=False,
        )

        # Build temporal edges
        temporal_edges = self._build_temporal_edges(
            temporal_data, coords.shape[0]
        )

        # Combine edges
        edge_index = torch.cat([spatial_edges, temporal_edges], dim=1)

        # Edge types: 0 for spatial, 1 for temporal
        edge_type = torch.cat([
            torch.zeros(spatial_edges.shape[1], dtype=torch.long),
            torch.ones(temporal_edges.shape[1], dtype=torch.long)
        ])

        # Create data object
        data = self.create_data_object(
            features, edge_index, coords,
            edge_type=edge_type
        )

        # Add temporal information
        if hasattr(temporal_data, "timestamps"):
            data.timestamps = temporal_data.timestamps
        if hasattr(temporal_data, "time_features"):
            data.time_features = temporal_data.time_features

        data.graph_type = "temporal"
        data.temporal_window = self.temporal_window

        return data

    def _build_temporal_edges(
        self, temporal_data: Any, n_nodes: int
    ) -> torch.Tensor:
        """Build edges connecting same object across time."""
        temporal_edges = []

        # If we have time stamps for each observation
        if hasattr(temporal_data, "node_timestamps"):
            timestamps = temporal_data.node_timestamps
            unique_times = torch.unique(timestamps, sorted=True)

            # Connect observations of same object across time
            if hasattr(temporal_data, "object_ids"):
                object_ids = temporal_data.object_ids

                for t_idx in range(len(unique_times) - 1):
                    t1 = unique_times[t_idx]
                    t2 = unique_times[t_idx + 1]

                    mask1 = timestamps == t1
                    mask2 = timestamps == t2

                    # Find matching objects
                    for obj_id in torch.unique(object_ids):
                        obj_mask1 = mask1 & (object_ids == obj_id)
                        obj_mask2 = mask2 & (object_ids == obj_id)

                        if obj_mask1.any() and obj_mask2.any():
                            idx1 = torch.where(obj_mask1)[0]
                            idx2 = torch.where(obj_mask2)[0]

                            # Connect all pairs
                            for i1 in idx1:
                                for i2 in idx2:
                                    temporal_edges.append([i1, i2])

        if temporal_edges:
            return torch.tensor(temporal_edges, device=self.device).t()
        else:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)


class GeometricPriorGraphBuilder(BaseGraphBuilder):
    """Build graphs using astronomical geometric priors."""

    def __init__(self, prior_type: str = "filament", **kwargs):
        config = GraphConfig(method="geometric_prior", **kwargs)
        super().__init__(config)
        self.prior_type = prior_type

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build graph with geometric priors."""
        self.validate_input(survey_tensor)

        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        if self.prior_type == "filament":
            edge_index = self._build_filament_graph(coords, features)
        elif self.prior_type == "cluster":
            edge_index = self._build_cluster_aware_graph(coords, features)
        elif self.prior_type == "void":
            edge_index = self._build_void_aware_graph(coords, features)
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")

        # Create data object
        data = self.create_data_object(features, edge_index, coords)
        data.graph_type = f"geometric_{self.prior_type}"

        return data

    def _build_filament_graph(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Build graph emphasizing filamentary structures."""
        # Compute local principal directions
        k_local = 20
        edge_index = knn_graph(coords, k=k_local, batch=None)

        # For each node, compute local PCA
        filament_edges = []
        
        for i in range(coords.shape[0]):
            # Get neighbors
            neighbors_mask = edge_index[0] == i
            neighbor_indices = edge_index[1, neighbors_mask]
            
            if len(neighbor_indices) < 3:
                continue
                
            # Local coordinates
            local_coords = coords[neighbor_indices] - coords[i]
            
            # PCA to find principal direction
            U, S, V = torch.svd(local_coords.t() @ local_coords)
            principal_dir = V[:, 0]
            
            # Connect to neighbors aligned with principal direction
            alignments = torch.abs((local_coords @ principal_dir))
            aligned_mask = alignments > alignments.mean()
            
            aligned_neighbors = neighbor_indices[aligned_mask]
            for j in aligned_neighbors:
                filament_edges.append([i, j])

        if filament_edges:
            edge_index = torch.tensor(filament_edges, device=self.device).t()
        else:
            # Fallback to regular KNN
            edge_index = knn_graph(coords, k=self.config.k_neighbors, batch=None)

        return edge_index

    def _build_cluster_aware_graph(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Build graph aware of cluster structures."""
        # Detect high-density regions
        k_density = 10
        dists = torch.cdist(coords, coords)
        kth_dists, _ = torch.kthvalue(dists, k_density + 1, dim=1)
        
        density = 1.0 / (kth_dists + 1e-8)
        density_threshold = density.quantile(0.7)
        
        high_density_mask = density > density_threshold
        
        # Different connectivity for high/low density
        edge_list = []
        
        # High density: more connections
        if high_density_mask.any():
            high_density_coords = coords[high_density_mask]
            high_density_idx = torch.where(high_density_mask)[0]
            
            k_high = min(self.config.k_neighbors * 2, len(high_density_coords) - 1)
            if k_high > 0:
                edges_high = knn_graph(high_density_coords, k=k_high, batch=None)
                edges_high_global = high_density_idx[edges_high]
                edge_list.append(edges_high_global)
        
        # Low density: fewer connections
        low_density_mask = ~high_density_mask
        if low_density_mask.any():
            low_density_coords = coords[low_density_mask]
            low_density_idx = torch.where(low_density_mask)[0]
            
            k_low = min(self.config.k_neighbors // 2, len(low_density_coords) - 1)
            if k_low > 0:
                edges_low = knn_graph(low_density_coords, k=k_low, batch=None)
                edges_low_global = low_density_idx[edges_low]
                edge_list.append(edges_low_global)
        
        # Inter-density connections
        if high_density_mask.any() and low_density_mask.any():
            inter_edges = self._connect_density_regions(
                coords, high_density_mask, low_density_mask
            )
            if inter_edges.shape[1] > 0:
                edge_list.append(inter_edges)
        
        if edge_list:
            edge_index = torch.cat(edge_list, dim=1)
        else:
            edge_index = knn_graph(coords, k=self.config.k_neighbors, batch=None)
            
        return edge_index

    def _build_void_aware_graph(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Build graph aware of void structures."""
        # Similar to cluster-aware but inverted
        # Emphasize connections around voids
        return self._build_cluster_aware_graph(coords, features)

    def _connect_density_regions(
        self,
        coords: torch.Tensor,
        high_mask: torch.Tensor,
        low_mask: torch.Tensor
    ) -> torch.Tensor:
        """Connect high and low density regions."""
        high_idx = torch.where(high_mask)[0]
        low_idx = torch.where(low_mask)[0]
        
        # Sample connections to avoid too many edges
        n_connections = min(len(high_idx), len(low_idx), 100)
        
        if n_connections > 0:
            high_sample = high_idx[torch.randperm(len(high_idx))[:n_connections]]
            low_sample = low_idx[torch.randperm(len(low_idx))[:n_connections]]
            
            edges = torch.stack([
                torch.cat([high_sample, low_sample]),
                torch.cat([low_sample, high_sample])
            ])
            
            return edges
        else:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)


# Convenience functions
def create_dynamic_graph(
    survey_tensor: SurveyTensorDict, initial_k: int = 16, **kwargs
) -> Data:
    """Create dynamic graph with learnable structure."""
    builder = DynamicGraphBuilder(initial_k=initial_k, **kwargs)
    return builder.build(survey_tensor)


def create_hierarchical_graph(
    survey_tensor: SurveyTensorDict,
    cluster_method: str = "kmeans",
    n_clusters: int = 10,
    **kwargs
) -> Data:
    """Create hierarchical graph-of-graphs."""
    builder = GraphOfGraphsBuilder(
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        **kwargs
    )
    return builder.build(survey_tensor)


def create_temporal_graph(
    survey_tensor: SurveyTensorDict,
    temporal_window: int = 5,
    **kwargs
) -> Data:
    """Create temporal graph for time-series data."""
    builder = TemporalGraphBuilder(
        temporal_window=temporal_window,
        **kwargs
    )
    return builder.build(survey_tensor)


def create_geometric_prior_graph(
    survey_tensor: SurveyTensorDict,
    prior_type: str = "filament",
    **kwargs
) -> Data:
    """Create graph with astronomical geometric priors."""
    builder = GeometricPriorGraphBuilder(
        prior_type=prior_type,
        **kwargs
    )
    return builder.build(survey_tensor)
