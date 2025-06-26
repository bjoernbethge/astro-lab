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
                batch_norm=use_batch_norm,
                dropout=dropout,
            )
            self.point_layers.append(layer)

        # Global feature transformation
        self.global_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through PointNet encoder.

        Args:
            x: Point cloud features [N, P, F] where N=batch, P=points, F=features

        Returns:
            Global features [N, hidden_dim]
        """
        # Point-wise feature extraction
        for layer in self.point_layers:
            x = layer(x)

        # Global max pooling (permutation invariant)
        global_features = torch.max(x, dim=1)[0]  # [N, hidden_dim]

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

        Returns:
            Node-level predictions [N, num_classes] or [N, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroPointNet] Forward: x.shape={x.shape}, expected num_features={self.num_features}"
        )
        assert x.shape[1] == self.num_features, (
            f"Input feature mismatch: x.shape[1]={x.shape[1]}, expected={self.num_features}"
        )

        if batch is None:
            # Single graph - create batch indices
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Reshape for PointNet: [N, F] -> [1, N, F] where N=num_nodes, F=num_features
        # Treat each node as a "point" in the point cloud
        x = x.unsqueeze(0)  # [1, N, F]

        # PointNet encoding
        global_features = self.pointnet_encoder(x)  # [1, hidden_dim]

        # For node-level tasks, we need to expand back to node-level
        if self.task == "point_classification":
            # Global classification - use global features directly
            return self.output_head(global_features)
        elif self.task == "point_segmentation":
            # Node-level segmentation - expand global features to all nodes
            global_features = global_features.expand(x.size(1), -1)  # [N, hidden_dim]
            return self.output_head(global_features)
        elif self.task == "point_registration":
            # Registration - use global features
            return self.output_head(global_features)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def get_point_embeddings(self, data) -> Tensor:
        """
        Get point cloud embeddings from the PointNet encoder.

        Args:
            data: PyG Data object

        Returns:
            Point cloud embeddings [N, hidden_dim]
        """
        x = data.x

        # Reshape if needed
        if x.dim() == 2:
            N = 1
            x = x.unsqueeze(0)
        else:
            N = x.size(0)

        # PointNet encoding (without final output head)
        global_features = self.pointnet_encoder(x)

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
