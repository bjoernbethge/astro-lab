"""Neighbor-based samplers with integrated graph construction.

Provides k-NN and radius-based graph construction with astronomical features.
"""

import warnings
from typing import List, Optional, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader, HGTLoader, NeighborLoader

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
        survey_name = kwargs.get("survey_name", "gaia")

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
        edge_index = self.knn_graph(coordinates, k=self.k, loop=self.loop)

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
            if key not in ['x', 'pos', 'edge_index', 'edge_attr']:
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
                    if key not in ['x', 'pos', 'edge_index', 'edge_attr', 'survey_name']:
                        if isinstance(value, torch.Tensor) and value.size(0) == coordinates.size(0):
                            setattr(data[node_type], key, value[mask])

        # Create edges between node types
        edge_types = [(nt, "to", nt) for nt in node_types]
        for src_type, relation, dst_type in edge_types:
            if src_type in data.node_types and dst_type in data.node_types:
                # Get coordinates for source and destination types
                src_mask = node_type_labels == node_types.index(src_type)
                dst_mask = node_type_labels == node_types.index(dst_type)

                if src_mask.sum() > 0 and dst_mask.sum() > 0:
                    src_coords = coordinates[src_mask]
                    dst_coords = coordinates[dst_mask]

                    # Create k-NN edges from src to dst
                    edge_index = self.knn_graph(src_coords, k=self.k, loop=self.loop)

                    # Map back to original indices
                    src_indices = torch.where(src_mask)[0]
                    dst_indices = torch.where(dst_mask)[0]

                    mapped_edge_index = torch.stack(
                        [src_indices[edge_index[0]], dst_indices[edge_index[1]]]
                    )

                    # Add edge index
                    data[src_type, relation, dst_type].edge_index = mapped_edge_index

                    # Calculate edge features
                    edge_attr = self.calculate_edge_features(
                        coordinates, mapped_edge_index
                    )
                    data[src_type, relation, dst_type].edge_attr = edge_attr

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
        data: Union[Data, HeteroData, List[Data], torch.utils.data.Subset],
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader for graph classification: always returns DataLoader over list of Data objects or Subset."""
        from torch_geometric.loader import DataLoader as PyGDataLoader

        if isinstance(data, torch.utils.data.Subset):
            # For Subset, we need to access the dataset directly
            loader = PyGDataLoader(
                data, batch_size=batch_size, shuffle=shuffle, **kwargs
            )
            print(
                f"[DEBUG] DataLoader (Subset): type={type(loader)}, len={len(loader)}"
            )
            return loader
        elif isinstance(data, list):
            loader = PyGDataLoader(
                data, batch_size=batch_size, shuffle=shuffle, **kwargs
            )
            print(f"[DEBUG] DataLoader (list): type={type(loader)}, len={len(loader)}")
            return loader
        elif isinstance(data, Data) or isinstance(data, HeteroData):
            loader = PyGDataLoader(
                [data], batch_size=batch_size, shuffle=shuffle, **kwargs
            )
            print(
                f"[DEBUG] DataLoader (single): type={type(loader)}, len={len(loader)}"
            )
            return loader
        else:
            # Handle other dataset-like objects
            loader = PyGDataLoader(
                data, batch_size=batch_size, shuffle=shuffle, **kwargs
            )
            print(
                f"[DEBUG] DataLoader (dataset): type={type(loader)}, len={len(loader)}"
            )
            return loader


class RadiusSampler(AstroLabSampler, SpatialSamplerMixin, AstronomicalSamplerMixin):
    """Radius-based graph sampler for astronomical neighborhoods."""

    def __init__(
        self,
        radius: float = 10.0,
        max_num_neighbors: int = 64,
        loop: bool = False,
        **kwargs,
    ):
        """Initialize radius sampler.

        Args:
            radius: Connection radius (in coordinate units, e.g., parsecs)
            max_num_neighbors: Maximum neighbors per node
            loop: Include self-loops
            **kwargs: Additional config
        """
        super().__init__(
            {
                "radius": radius,
                "max_num_neighbors": max_num_neighbors,
                "loop": loop,
                **kwargs,
            }
        )
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create radius graph from astronomical data.

        Args:
            coordinates: 3D positions [N, 3]
            features: Node features [N, F]
            **kwargs: Additional data

        Returns:
            PyG Data object with radius-based edges
        """
        self.validate_inputs(coordinates, features)
        
        # Ensure features are 2D
        if features.dim() == 1:
            features = features.unsqueeze(1)

        # Create radius graph
        edge_index = self.radius_graph(
            coordinates,
            r=self.radius,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )

        # Calculate edge features
        edge_attr = self.calculate_edge_features(coordinates, edge_index)

        # Create Data object
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Add additional attributes
        for key, value in kwargs.items():
            if key not in ['x', 'pos', 'edge_index', 'edge_attr']:
                setattr(data, key, value)

        # Update stats
        self.sampling_stats.update(
            {
                "num_nodes": coordinates.size(0),
                "num_edges": edge_index.size(1),
                "avg_degree": edge_index.size(1) / coordinates.size(0),
                "radius": self.radius,
                "isolated_nodes": (torch.bincount(edge_index[0]) == 0).sum().item(),
            }
        )

        return data

    def create_dataloader(
        self,
        data: Union[Data, List[Data], torch.utils.data.Subset],
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs,
    ) -> DataLoader:
        """Create DataLoader for radius graphs."""
        from torch_geometric.loader import DataLoader as PyGDataLoader
        
        if isinstance(data, torch.utils.data.Subset):
            return PyGDataLoader(
                data, batch_size=batch_size, shuffle=shuffle, **kwargs
            )
        elif isinstance(data, list):
            return PyGDataLoader(
                data, batch_size=batch_size, shuffle=shuffle, **kwargs
            )
        else:
            return PyGDataLoader(
                data, batch_size=batch_size, shuffle=shuffle, **kwargs
            )


class NeighborSubgraphSampler(AstroLabSampler, SpatialSamplerMixin):
    """Subgraph sampler for local neighborhood extraction."""

    def __init__(
        self, num_neighbors: List[int] = [25, 10], batch_size: int = 1024, **kwargs
    ):
        """Initialize subgraph sampler.

        Args:
            num_neighbors: Number of neighbors per hop
            batch_size: Number of nodes per batch
            **kwargs: Additional config
        """
        super().__init__(
            {"num_neighbors": num_neighbors, "batch_size": batch_size, **kwargs}
        )
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create initial graph for subgraph sampling.

        First creates full k-NN graph, then samples subgraphs during training.
        """
        # Ensure features are 2D
        if features.dim() == 1:
            features = features.unsqueeze(1)
            
        # Start with k-NN graph
        k = max(self.num_neighbors[0], 20)  # Ensure enough connectivity
        edge_index = self.knn_graph(coordinates, k=k, loop=False)

        # Create full graph
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
        )

        # Add additional attributes
        for key, value in kwargs.items():
            if key not in ['x', 'pos', 'edge_index']:
                setattr(data, key, value)

        return data

    def create_dataloader(
        self,
        data: Data,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> NeighborLoader:
        """Create NeighborLoader for efficient subgraph sampling.

        Args:
            data: Full graph Data object
            batch_size: Override default batch size
            shuffle: Whether to shuffle nodes
            **kwargs: Additional NeighborLoader args

        Returns:
            NeighborLoader instance
        """
        batch_size = batch_size or self.batch_size

        return NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            input_nodes=data.train_mask.nonzero(as_tuple=False).view(-1)
            if hasattr(data, 'train_mask') else None,
            **kwargs,
        )


class AdaptiveRadiusSampler(RadiusSampler):
    """Adaptive radius sampler that adjusts radius based on local density."""

    def __init__(
        self,
        base_radius: float = 10.0,
        target_neighbors: int = 20,
        min_radius: float = 1.0,
        max_radius: float = 100.0,
        **kwargs,
    ):
        """Initialize adaptive radius sampler.

        Args:
            base_radius: Starting radius
            target_neighbors: Target number of neighbors
            min_radius: Minimum allowed radius
            max_radius: Maximum allowed radius
            **kwargs: Additional config
        """
        super().__init__(radius=base_radius, **kwargs)
        self.target_neighbors = target_neighbors
        self.min_radius = min_radius
        self.max_radius = max_radius

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create adaptive radius graph.

        Adjusts radius per node to achieve target neighbor count.
        """
        self.validate_inputs(coordinates, features)

        # First pass with base radius
        edge_index = self.radius_graph(
            coordinates,
            r=self.radius,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )

        # Count neighbors per node
        neighbor_counts = torch.bincount(edge_index[0], minlength=coordinates.size(0))

        # Identify nodes needing adjustment
        too_few = neighbor_counts < self.target_neighbors // 2
        too_many = neighbor_counts > self.target_neighbors * 2

        # Adaptive radius adjustment (simplified)
        if too_few.any() or too_many.any():
            # This is a simplified version - in practice you'd want
            # per-node radius adjustment
            avg_neighbors = neighbor_counts.float().mean()
            if avg_neighbors < self.target_neighbors:
                self.radius = min(self.radius * 1.5, self.max_radius)
            elif avg_neighbors > self.target_neighbors * 1.5:
                self.radius = max(self.radius * 0.7, self.min_radius)

            # Recreate graph with adjusted radius
            edge_index = self.radius_graph(
                coordinates,
                r=self.radius,
                loop=self.loop,
                max_num_neighbors=self.max_num_neighbors,
            )

        # Continue with standard processing
        return super().create_graph(coordinates, features, **kwargs)
