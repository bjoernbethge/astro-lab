from typing import List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import knn_graph

from .base import AstroLabSampler, SpatialSamplerMixin


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

        # Use a reasonable k value based on num_neighbors[0]
        k = min(self.num_neighbors[0], 10)  # Cap at 10 to avoid overly dense graphs

        edge_index = knn_graph(coordinates, k=k, loop=False)

        # Create full graph
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
        )

        # Add additional attributes
        for key, value in kwargs.items():
            if key not in ["x", "pos", "edge_index"]:
                setattr(data, key, value)

        return data

    def create_dataloader(
        self,
        dataset,
        indices=None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ):
        """Create NeighborLoader for node classification/regression tasks."""

        batch_size = batch_size or self.batch_size

        if indices is not None:
            # For node tasks with splits, use DataLoader on subset
            subset = [dataset[i] for i in indices]
            return DataLoader(
                subset,
                batch_size=min(batch_size, len(subset)),
                shuffle=shuffle,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=torch.cuda.is_available(),
                **kwargs,
            )
        else:
            # For full graph node tasks, use NeighborLoader
            # Get the first graph as full graph
            full_graph = dataset[0] if hasattr(dataset, "__getitem__") else dataset

            # Ensure train_mask exists
            if not hasattr(full_graph, "train_mask"):
                mask = torch.zeros(full_graph.x.size(0), dtype=torch.bool)
                mask[: int(0.7 * full_graph.x.size(0))] = True
                full_graph.train_mask = mask

            return NeighborLoader(
                full_graph,
                num_neighbors=self.num_neighbors,
                batch_size=batch_size,
                shuffle=shuffle,
                input_nodes=full_graph.train_mask.nonzero(as_tuple=False).view(-1)
                if hasattr(full_graph, "train_mask")
                else None,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=torch.cuda.is_available(),
                **kwargs,
            )
