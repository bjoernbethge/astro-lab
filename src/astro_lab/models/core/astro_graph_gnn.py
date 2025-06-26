"""
AstroGraphGNN - Unified Graph-Level Graph Neural Network
======================================================

Single model for all graph-level astronomical tasks:
- Survey classification
- Cluster analysis
- Graph property prediction
- Graph regression
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


class AstroGraphGNN(AstroLightningMixin, LightningModule):
    """
    Unified Graph Neural Network for graph-level astronomical tasks.

    Supports all graph-level tasks with a single, flexible architecture:
    - Graph Classification: Survey types, cluster types, object categories
    - Graph Regression: Survey properties, cluster properties, global features
    - Graph Generation: Survey synthesis, cluster generation

    Args:
        num_features: Number of input features per node
        num_classes: Number of output classes (for classification)
        hidden_dim: Base hidden dimension
        num_layers: Number of GNN layers
        conv_type: GNN convolution type ('gcn', 'gat', 'sage', 'transformer')
        task: Task type ('graph_classification', 'graph_regression')
        learning_rate: Learning rate for training
        optimizer: Optimizer type ('adam', 'adamw')
        scheduler: Scheduler type ('cosine', 'onecycle')
        weight_decay: Weight decay for regularization
        dropout: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        pooling: Global pooling type ('mean', 'max', 'sum', 'attention')
    """

    def __init__(
        self,
        num_features: int = 64,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gcn",
        task: str = "graph_classification",
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
            f"[AstroGraphGNN] Init: num_features={num_features}, hidden_dim={hidden_dim}"
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

        # Global pooling layer
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Task-specific output head
        if task == "graph_classification":
            self.output_head = create_output_head(
                "classification",
                input_dim=hidden_dim,
                output_dim=num_classes,
                dropout=dropout,
            )
        elif task == "graph_regression":
            self.output_head = create_output_head(
                "regression",
                input_dim=hidden_dim,
                output_dim=num_classes,  # num_classes used as output_dim
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
            Graph-level predictions [B, num_classes] or [B, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroGraphGNN] Forward: x.shape={x.shape}, expected num_features={self.num_features}"
        )
        assert x.shape[1] == self.num_features, (
            f"Input feature mismatch: x.shape[1]={x.shape[1]}, expected={self.num_features}"
        )

        if batch is None:
            # Single graph - create batch indices
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Feature encoding
        x = self.feature_encoder(x)

        # GNN layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        elif self.pooling == "attention":
            # Attention-based pooling
            attention_weights = self.attention(x)
            x = global_mean_pool(x * attention_weights, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # Task-specific output
        return self.output_head(x)

    def get_graph_embeddings(self, data) -> Tensor:
        """
        Get graph embeddings from the last GNN layer.

        Args:
            data: PyG Data object

        Returns:
            Graph embeddings [B, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Feature encoding
        x = self.feature_encoder(x)

        # GNN layers (without final output head)
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        elif self.pooling == "attention":
            attention_weights = self.attention(x)
            x = global_mean_pool(x * attention_weights, batch)

        return x


def create_astro_graph_gnn(
    num_features: int = 64,
    num_classes: int = 2,
    task: str = "graph_classification",
    **kwargs,
) -> AstroGraphGNN:
    """
    Factory function to create AstroGraphGNN model.

    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type
        **kwargs: Additional model parameters

    Returns:
        Configured AstroGraphGNN model
    """
    return AstroGraphGNN(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )
