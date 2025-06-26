"""
PointNet++ GNN for Astronomical Point Clouds
==========================================

Implements PointNet++ architecture for processing 3D astronomical point clouds
such as star clusters, galaxy distributions, and simulation data.

Migrated from astro_gnn.model.py to the consolidated models module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool, knn_graph
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple, Union

from ..components import create_mlp, create_output_head


class PointNetLayer(nn.Module):
    """
    Custom PointNet Layer for astronomical features.
    
    Processes 3D coordinates and additional features such as:
    - Magnitude/brightness
    - Colors (BP-RP, g-r, etc.)
    - Proper motions
    - Spectroscopic features
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        use_batch_norm: bool = True,
        local_nn_dims: Optional[list] = None,
        global_nn_dims: Optional[list] = None
    ):
        super().__init__()
        
        # Default architectures if not specified
        if local_nn_dims is None:
            local_nn_dims = [in_channels + 3, 64, 128, out_channels]
        if global_nn_dims is None:
            global_nn_dims = [out_channels, out_channels]
        
        # Local MLP: Processes features + relative positions
        self.local_nn = create_mlp(
            local_nn_dims,
            activation="relu",
            use_batch_norm=use_batch_norm,
            dropout=0.0
        )
        
        # Global MLP: Processes aggregated features
        self.global_nn = create_mlp(
            global_nn_dims,
            activation="relu",
            use_batch_norm=use_batch_norm,
            dropout=0.0
        )
        
        self.pointnet = PointNetConv(self.local_nn, self.global_nn)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through PointNet layer."""
        return self.pointnet(x, pos, edge_index)


class AstronomicalPointNetGNN(nn.Module):
    """
    PointNet++ model for astronomical point cloud classification and analysis.
    
    Supports various astronomical tasks:
    - Star cluster classification
    - Galaxy morphology classification
    - Simulation particle type identification
    - Point cloud segmentation
    
    Args:
        num_features: Number of input features per point
        num_classes: Number of output classes (for classification)
        hidden_dim: Base hidden dimension
        num_layers: Number of PointNet++ layers
        dropout: Dropout rate for regularization
        k_neighbors: Number of neighbors for k-NN graph
        task: Task type ('classification', 'regression', 'segmentation')
        pooling: Global pooling type ('max', 'mean', 'sum')
        use_edge_features: Whether to compute and use edge features
    """
    
    def __init__(
        self,
        num_features: int = 3,
        num_classes: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        k_neighbors: int = 16,
        task: str = "classification",
        pooling: str = "max",
        use_edge_features: bool = False,
        use_batch_norm: bool = True,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.task = task
        self.pooling = pooling
        self.use_edge_features = use_edge_features
        
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Build PointNet++ layers with increasing feature dimensions
        dims = [num_features]
        for i in range(num_layers):
            dims.append(hidden_dim * (2 ** i))
        
        self.pointnet_layers = nn.ModuleList()
        for i in range(num_layers):
            self.pointnet_layers.append(
                PointNetLayer(
                    dims[i], 
                    dims[i + 1], 
                    use_batch_norm=use_batch_norm
                )
            )
        
        # Feature dimension after PointNet++ layers
        final_dim = dims[-1]
        
        # Task-specific output head
        if task == "classification":
            self.output_head = create_output_head(
                "classification",
                input_dim=final_dim,
                output_dim=num_classes,
                dropout=dropout
            )
        elif task == "regression":
            self.output_head = create_output_head(
                "regression",
                input_dim=final_dim,
                output_dim=num_classes,  # num_classes used as output_dim
                dropout=dropout
            )
        elif task == "segmentation":
            # For segmentation, we need per-point predictions
            self.segmentation_head = create_mlp(
                [final_dim, hidden_dim, num_classes],
                activation="relu",
                use_batch_norm=use_batch_norm,
                dropout=dropout,
                output_activation=False
            )
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Edge feature extractor (optional)
        if use_edge_features:
            self.edge_mlp = create_mlp(
                [6, 32, 64],  # 6D: relative position + distance
                activation="relu",
                use_batch_norm=use_batch_norm
            )
        
        self.to(self.device)
    
    def create_graph_from_pointcloud(
        self, 
        pos: torch.Tensor, 
        features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Data:
        """
        Create a k-NN graph from point cloud data.
        
        Args:
            pos: 3D positions [N, 3]
            features: Optional features [N, F]
            batch: Batch assignment [N]
            
        Returns:
            PyG Data object
        """
        # Create k-NN graph
        edge_index = knn_graph(pos, k=self.k_neighbors, batch=batch, loop=False)
        
        # Use positions as features if none provided
        x = features if features is not None else pos
        
        # Compute edge features if requested
        edge_attr = None
        if self.use_edge_features:
            edge_attr = self._compute_edge_features(pos, edge_index)
        
        return Data(
            x=x, 
            pos=pos, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            batch=batch
        )
    
    def _compute_edge_features(
        self, 
        pos: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge features from positions."""
        row, col = edge_index
        
        # Relative positions
        rel_pos = pos[col] - pos[row]
        
        # Euclidean distance
        dist = torch.norm(rel_pos, dim=1, keepdim=True)
        
        # Concatenate relative position and distance
        edge_features = torch.cat([rel_pos, dist], dim=1)
        
        if hasattr(self, 'edge_mlp'):
            edge_features = self.edge_mlp(edge_features)
        
        return edge_features
    
    def forward(self, data: Data) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            data: PyG Data object with x, pos, edge_index, batch
            
        Returns:
            For classification/regression: predictions [B, num_classes]
            For segmentation: per-point predictions [N, num_classes]
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Move to device
        x = x.to(self.device)
        pos = pos.to(self.device)
        edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)
        
        # Process through PointNet++ layers
        for i, layer in enumerate(self.pointnet_layers):
            x = layer(x, pos, edge_index)
            if i < len(self.pointnet_layers) - 1:
                x = F.relu(x)
        
        # Task-specific output
        if self.task == "segmentation":
            # Per-point predictions
            return self.segmentation_head(x)
        else:
            # Global pooling for graph-level predictions
            if self.pooling == "max":
                x = global_max_pool(x, batch)
            elif self.pooling == "mean":
                from torch_geometric.nn import global_mean_pool
                x = global_mean_pool(x, batch)
            elif self.pooling == "sum":
                from torch_geometric.nn import global_add_pool
                x = global_add_pool(x, batch)
            
            # Classification or regression
            return self.output_head(x)
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Extract feature embeddings without classification.
        
        Useful for:
        - Visualization (t-SNE, UMAP)
        - Transfer learning
        - Similarity analysis
        - Clustering
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Move to device
        x = x.to(self.device)
        pos = pos.to(self.device)
        edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)
        
        # Process through PointNet++ layers
        for i, layer in enumerate(self.pointnet_layers):
            x = layer(x, pos, edge_index)
            if i < len(self.pointnet_layers) - 1:
                x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            if self.pooling == "max":
                embeddings = global_max_pool(x, batch)
            elif self.pooling == "mean":
                from torch_geometric.nn import global_mean_pool
                embeddings = global_mean_pool(x, batch)
            else:
                from torch_geometric.nn import global_add_pool
                embeddings = global_add_pool(x, batch)
        else:
            # Single graph
            if self.pooling == "max":
                embeddings = x.max(dim=0)[0]
            elif self.pooling == "mean":
                embeddings = x.mean(dim=0)
            else:
                embeddings = x.sum(dim=0)
        
        return embeddings
    
    def compute_attention_weights(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Compute attention weights for interpretability.
        
        Returns attention weights at each layer to understand
        which points are most important for the prediction.
        """
        # This would require modifying PointNetConv to return attention weights
        # For now, return empty dict as placeholder
        return {}


# Convenience function to create the model
def create_pointnet_gnn(
    num_features: int = 3,
    num_classes: int = 7,
    task: str = "classification",
    **kwargs
) -> AstronomicalPointNetGNN:
    """Create an AstronomicalPointNetGNN model."""
    return AstronomicalPointNetGNN(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs
    )
