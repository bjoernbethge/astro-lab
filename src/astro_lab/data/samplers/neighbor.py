"""Neighbor-based samplers with integrated graph construction.

Provides k-NN and radius-based graph construction with astronomical features.
"""

from typing import Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph

from .base import AstroLabSampler, AstronomicalSamplerMixin, SpatialSamplerMixin


class KNNSampler(AstroLabSampler, SpatialSamplerMixin, AstronomicalSamplerMixin):
    """k-Nearest Neighbors graph sampler with astronomical features."""

    def __init__(self, k: int = 8, loop: bool = False, **kwargs):
        """Initialize k-NN sampler.

        Args:
            k: Number of nearest neighbors
            loop: Include self-loops
            **kwargs: Additional config
        """
        super().__init__({"k": k, "loop": loop, **kwargs})
        self.k = k
        self.loop = loop

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Union[Data, HeteroData]:
        """Create k-NN graph from astronomical data.

        Args:
            coordinates: 3D positions [N, 3]
            features: Node features [N, F]
            **kwargs: Additional data (y, train_mask, node_type_labels, etc.)

        Returns:
            PyG Data or HeteroData object with k-NN edges
        """
        self.validate_inputs(coordinates, features)

        # Ensure features are 2D
        if features.dim() == 1:
            features = features.unsqueeze(1)

        # Check if we should create heterogeneous graph
        node_type_labels = kwargs.get("node_type_labels", None)
        kwargs.get("survey_name", "gaia")

        if node_type_labels is not None and len(torch.unique(node_type_labels)) > 1:
            # Create heterogeneous graph
            # Remove node_type_labels from kwargs to avoid duplicate argument
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("node_type_labels", None)
            return self._create_heterogeneous_graph(
                coordinates, features, node_type_labels, **kwargs_copy
            )
        else:
            # Create homogeneous graph (original behavior)
            return self._create_homogeneous_graph(coordinates, features, **kwargs)

    def _create_homogeneous_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create homogeneous k-NN graph."""
        # Create k-NN graph

        edge_index = knn_graph(coordinates, k=self.k, loop=self.loop)

        # Calculate edge features
        edge_attr = self.calculate_edge_features(coordinates, edge_index)

        # Create Data object
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Add any additional attributes
        for key, value in kwargs.items():
            # Skip attributes that we already set
            if key not in ["x", "pos", "edge_index", "edge_attr"]:
                setattr(data, key, value)

        # Update stats
        self.sampling_stats.update(
            {
                "num_nodes": coordinates.size(0),
                "num_edges": edge_index.size(1),
                "avg_degree": edge_index.size(1) / coordinates.size(0),
                "k": self.k,
                "graph_type": "homogeneous",
            }
        )

        return data

    def _create_heterogeneous_graph(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        node_type_labels: torch.Tensor,
        **kwargs,
    ) -> HeteroData:
        """Create heterogeneous k-NN graph with multiple node types."""
        # Define node types based on survey
        survey_name = kwargs.get("survey_name", "gaia")
        if survey_name == "gaia":
            node_types = ["star", "binary", "variable"]
        elif survey_name == "sdss":
            node_types = ["galaxy", "quasar", "star"]
        elif survey_name == "nsa":
            node_types = ["galaxy", "cluster"]
        else:
            node_types = ["object"]

        # Create HeteroData object
        data = HeteroData()

        # Split data by node type
        for i, node_type in enumerate(node_types):
            mask = node_type_labels == i
            if mask.sum() > 0:
                # Add node features and positions for this type
                data[node_type].x = features[mask]
                data[node_type].pos = coordinates[mask]

                # Add any additional node attributes
                for key, value in kwargs.items():
                    if key not in [
                        "x",
                        "pos",
                        "edge_index",
                        "edge_attr",
                        "survey_name",
                    ]:
                        if isinstance(value, torch.Tensor) and value.size(
                            0
                        ) == coordinates.size(0):
                            setattr(data[node_type], key, value[mask])

        # Create edges within each node type (homogeneous edges)
        for node_type in node_types:
            if node_type in data.node_types:
                # Get coordinates for this node type
                mask = node_type_labels == node_types.index(node_type)
                if mask.sum() > 1:  # Need at least 2 nodes for edges
                    node_coords = coordinates[mask]

                    # Create k-NN edges within this node type
                    edge_index = knn_graph(node_coords, k=self.k, loop=self.loop)

                    # Map back to original indices
                    node_indices = torch.where(mask)[0]
                    mapped_edge_index = node_indices[edge_index]

                    # Add edge index
                    data[node_type, "to", node_type].edge_index = mapped_edge_index

                    # Calculate edge features
                    edge_attr = self.calculate_edge_features(
                        coordinates, mapped_edge_index
                    )
                    data[node_type, "to", node_type].edge_attr = edge_attr

        # Update stats
        total_edges = sum(
            data[edge_type].edge_index.size(1) for edge_type in data.edge_types
        )
        self.sampling_stats.update(
            {
                "num_nodes": coordinates.size(0),
                "node_types": data.node_types,
                "edge_types": data.edge_types,
                "total_edges": total_edges,
                "k": self.k,
                "graph_type": "heterogeneous",
            }
        )

        return data

    def create_dataloader(
        self,
        dataset,
        indices=None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ):
        """Create DataLoader for graph classification tasks."""

        if indices is not None:
            # Create subset from indices
            subset = [dataset[i] for i in indices]
        else:
            # Use full dataset
            subset = dataset

        return DataLoader(
            subset,
            batch_size=min(batch_size, len(subset)),
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            **kwargs,
        )
