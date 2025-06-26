"""
AstroNodeGNN - Unified Node-Level Graph Neural Network
====================================================

Single model for all node-level astronomical tasks:
- Star classification (Gaia, SDSS, etc.)
- Galaxy type classification
- Object categorization
- Node regression
- Node segmentation
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ..components import AstroLightningMixin, create_mlp, create_output_head
from ..encoders import SurveyEncoder


class AstroNodeGNN(AstroLightningMixin, LightningModule):
    """
    Unified Graph Neural Network for node-level astronomical tasks.

    Supports all node-level tasks with a single, flexible architecture:
    - Node Classification: Star types, galaxy morphologies, object categories
    - Node Regression: Magnitudes, distances, physical properties
    - Node Segmentation: Point cloud segmentation, cluster membership

    Args:
        num_features: Number of input features per node
        num_classes: Number of output classes (for classification)
        hidden_dim: Base hidden dimension
        num_layers: Number of GNN layers
        conv_type: GNN convolution type ('gcn', 'gat', 'sage', 'transformer')
        task: Task type ('node_classification', 'node_regression', 'node_segmentation')
        learning_rate: Learning rate for training
        optimizer: Optimizer type ('adam', 'adamw')
        scheduler: Scheduler type ('cosine', 'onecycle')
        weight_decay: Weight decay for regularization
        dropout: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        pooling: Global pooling type for graph-level tasks ('mean', 'max', 'sum')
    """

    def __init__(
        self,
        num_features: int = 64,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gcn",
        task: str = "node_classification",
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        pooling: str = "mean",
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
        self.conv_type = conv_type
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling = pooling

        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroNodeGNN] Init: num_features={num_features}, hidden_dim={hidden_dim}"
        )

        # Feature encoder (if needed)
        if num_features != hidden_dim:
            self.feature_encoder = create_mlp(
                input_dim=num_features,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim // 2],
                activation="relu",
                batch_norm=use_batch_norm,
                dropout=dropout,
            )
        else:
            self.feature_encoder = nn.Identity()

        # GNN layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim

            if conv_type == "gcn":
                conv = GCNConv(in_dim, hidden_dim)
            elif conv_type == "gat":
                heads = kwargs.get("num_heads", 8)
                conv = GATConv(in_dim, hidden_dim // heads, heads=heads)
            elif conv_type == "sage":
                conv = SAGEConv(in_dim, hidden_dim)
            elif conv_type == "transformer":
                heads = kwargs.get("num_heads", 8)
                conv = TransformerConv(in_dim, hidden_dim // heads, heads=heads)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.conv_layers.append(conv)

            if use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())

        # Task-specific output head
        if task == "node_classification":
            self.output_head = create_output_head(
                "classification",
                input_dim=hidden_dim,
                output_dim=num_classes,
                dropout=dropout,
            )
        elif task == "node_regression":
            self.output_head = create_output_head(
                "regression",
                input_dim=hidden_dim,
                output_dim=num_classes,  # num_classes used as output_dim
                dropout=dropout,
            )
        elif task == "node_segmentation":
            self.output_head = create_mlp(
                input_dim=hidden_dim,
                output_dim=num_classes,
                hidden_dims=[hidden_dim // 2],
                activation="relu",
                batch_norm=use_batch_norm,
                dropout=dropout,
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
            f"[AstroNodeGNN] Forward: x.shape={x.shape}, expected num_features={self.num_features}"
        )
        assert x.shape[1] == self.num_features, (
            f"Input feature mismatch: x.shape[1]={x.shape[1]}, expected={self.num_features}"
        )

        # Feature encoding
        x = self.feature_encoder(x)

        # GNN layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Task-specific output
        if self.task == "node_segmentation":
            # Per-node predictions (no pooling)
            return self.output_head(x)
        else:
            # Node-level predictions
            return self.output_head(x)

    def get_embeddings(self, data) -> Tensor:
        """
        Get node embeddings from the last GNN layer.

        Args:
            data: PyG Data object

        Returns:
            Node embeddings [N, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index

        # Feature encoding
        x = self.feature_encoder(x)

        # GNN layers (without final output head)
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


def create_astro_node_gnn(
    num_features: int = 64,
    num_classes: int = 2,
    task: str = "node_classification",
    **kwargs,
) -> AstroNodeGNN:
    """
    Factory function to create AstroNodeGNN model.

    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type
        **kwargs: Additional model parameters

    Returns:
        Configured AstroNodeGNN model
    """
    return AstroNodeGNN(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )
