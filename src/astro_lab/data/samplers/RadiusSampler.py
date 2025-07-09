import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

from .base import AstroLabSampler, AstronomicalSamplerMixin, SpatialSamplerMixin


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
        edge_index = radius_graph(
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

        # Add any additional attributes
        for key, value in kwargs.items():
            if key not in ["x", "pos", "edge_index", "edge_attr"]:
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
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        **kwargs,
    ):
        """Create DataLoader for radius graphs."""

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
