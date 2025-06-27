"""
AstroPointNet - Unified Point Cloud Neural Network
=================================================

Single model for all point cloud astronomical tasks:
- Point cloud classification
- Point cloud segmentation
- Point cloud registration
- Point cloud generation
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch_geometric.nn import global_max_pool, global_mean_pool

from ..components import AstroLightningMixin, create_mlp, create_output_head


class PointNetEncoder(nn.Module):
    """
    PointNet encoder for point cloud feature extraction.

    This is the core of PointNet architecture that learns
    permutation-invariant features from point clouds.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Point-wise MLP layers
        self.point_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim

            layer = create_mlp(
                input_dim=in_dim,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim // 2],
                activation="relu",
                batch_norm=False,  # Disable BatchNorm
                layer_norm=use_batch_norm,  # Use LayerNorm if normalization requested
                dropout=dropout,
            )
            self.point_layers.append(layer)

        # Global feature transformation
        if use_batch_norm:
            # Use LayerNorm instead of BatchNorm for single-sample compatibility
            self.global_transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Works with any batch size
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.global_transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through PointNet encoder.

        Args:
            x: Point cloud features [N, F] or [B, P, F] where N=total points, B=batch, P=points, F=features
            batch: Batch indices for graph batching (optional)

        Returns:
            Global features [B, hidden_dim]
        """
        # Handle different input formats
        if x.dim() == 2 and batch is not None:
            # Graph batch format [N, F] with batch indices
            # Convert to point cloud format [B, P, F]
            unique_batches = torch.unique(batch)
            batch_size = len(unique_batches)
            max_points = torch.bincount(batch).max().item()
            
            x_batched = torch.zeros(batch_size, max_points, x.size(1), device=x.device)
            for i, b in enumerate(unique_batches):
                mask = batch == b
                points = x[mask]
                x_batched[i, :points.size(0)] = points
            
            x = x_batched
        elif x.dim() == 2:
            # Single point cloud [P, F] -> [1, P, F]
            x = x.unsqueeze(0)
        
        # Now x is [B, P, F]
        # Point-wise feature extraction
        for layer in self.point_layers:
            x = layer(x)

        # Global max pooling (permutation invariant)
        global_features = torch.max(x, dim=1)[0]  # [B, hidden_dim]

        # Global feature transformation
        global_features = self.global_transform(global_features)

        return global_features


class AstroPointNet(AstroLightningMixin, LightningModule):
    """
    Unified Point Cloud Neural Network for astronomical tasks.

    Supports all point cloud tasks with a single, flexible architecture:
    - Point Cloud Classification: Object types, survey types
    - Point Cloud Segmentation: Point-wise labels, cluster membership
    - Point Cloud Registration: Spatial alignment, coordinate transformation
    - Point Cloud Generation: Synthetic point cloud generation

    Args:
        num_features: Number of input features per point
        num_classes: Number of output classes (for classification)
        hidden_dim: Base hidden dimension
        num_layers: Number of PointNet layers
        task: Task type ('point_classification', 'point_segmentation', 'point_registration')
        learning_rate: Learning rate for training
        optimizer: Optimizer type ('adam', 'adamw')
        scheduler: Scheduler type ('cosine', 'onecycle')
        weight_decay: Weight decay for regularization
        dropout: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        num_features: int = 3,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        task: str = "point_classification",
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        # Initialize Lightning mixin first
        super().__init__(
            task=task,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            **kwargs,
        )

        # Initialize LightningModule
        LightningModule.__init__(self)
        self.save_hyperparameters()

        # Setup criterion after LightningModule initialization
        self._setup_criterion()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroPointNet] Init: num_features={num_features}, hidden_dim={hidden_dim}"
        )

        # PointNet encoder
        self.pointnet_encoder = PointNetEncoder(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Task-specific output head
        if task == "point_classification":
            # Global classification
            self.output_head = create_output_head(
                "classification",
                input_dim=hidden_dim,
                output_dim=num_classes,
                dropout=dropout,
            )
        elif task == "point_segmentation":
            # Point-wise segmentation
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        elif task == "point_registration":
            # Point cloud registration (transformation matrix)
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 12),  # 3x4 transformation matrix (flattened)
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, data) -> Tensor:
        """
        Forward pass through the network.

        Args:
            data: PyG Data object with x, edge_index, batch
                  - x: Node features [N, num_features] 
                  - edge_index: Edge connectivity [2, E]
                  - batch: Batch assignment for nodes [N] (optional)

        Returns:
            Predictions based on task type:
            - point_classification: [B, num_classes] where B is number of graphs
            - point_segmentation: [N, num_classes] where N is number of nodes
            - point_registration: [B, 12] transformation matrices
            
        Expected shapes:
            - For graph-level tasks (point_classification): targets should be [B]
            - For node-level tasks (point_segmentation): targets should be [N]
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroPointNet] Forward: x.shape={x.shape}, expected num_features={self.num_features}"
        )
        
        # Validate input features
        assert x.shape[1] == self.num_features, (
            f"Input feature mismatch: x.shape[1]={x.shape[1]}, expected={self.num_features}"
        )

        if batch is None:
            # Single graph - create batch indices
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            logger.info(f"[AstroPointNet] Single graph mode, created batch indices")

        # PointNet encoding with batch support
        global_features = self.pointnet_encoder(x, batch)  # [B, hidden_dim]
        
        # Log shapes for debugging
        unique_batches = torch.unique(batch)
        logger.info(
            f"[AstroPointNet] Batch info: {len(unique_batches)} graphs, "
            f"global_features.shape={global_features.shape}"
        )

        # Task-specific output
        if self.task == "point_classification":
            # Global classification - use global features directly
            output = self.output_head(global_features)
            logger.info(
                f"[AstroPointNet] point_classification output shape: {output.shape} "
                f"(expecting targets shape: [{output.shape[0]}])"
            )
            return output
            
        elif self.task == "point_segmentation":
            # Node-level segmentation - need to map back to nodes
            # Create mapping from nodes to their batch
            node_features = []
            for i, b in enumerate(unique_batches):
                mask = batch == b
                num_nodes = mask.sum()
                # Expand global features to all nodes in this batch
                expanded = global_features[i:i+1].expand(num_nodes, -1)
                node_features.append(expanded)
            
            node_features = torch.cat(node_features, dim=0)  # [N, hidden_dim]
            output = self.output_head(node_features)
            logger.info(
                f"[AstroPointNet] point_segmentation output shape: {output.shape} "
                f"(expecting targets shape: [{output.shape[0]}])"
            )
            return output
            
        elif self.task == "point_registration":
            # Registration - use global features
            output = self.output_head(global_features)
            logger.info(
                f"[AstroPointNet] point_registration output shape: {output.shape}"
            )
            return output
            
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def get_point_embeddings(self, data) -> Tensor:
        """
        Get point cloud embeddings from the PointNet encoder.

        Args:
            data: PyG Data object

        Returns:
            Point cloud embeddings [B, hidden_dim]
        """
        x = data.x
        batch = getattr(data, "batch", None)

        if batch is None:
            # Single graph - create batch indices
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # PointNet encoding (without final output head)
        global_features = self.pointnet_encoder(x, batch)

        return global_features


def create_astro_pointnet(
    num_features: int = 3,
    num_classes: int = 2,
    task: str = "point_classification",
    **kwargs,
) -> AstroPointNet:
    """
    Factory function to create AstroPointNet model.

    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type
        **kwargs: Additional model parameters

    Returns:
        Configured AstroPointNet model
    """
    return AstroPointNet(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )
