"""Unified AstroLab Model - Clean GNN implementation for astronomical data."""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from astro_lab.models.layers.hetero import HeteroGNNLayer

from .base_model import AstroBaseModel


class AstroModel(AstroBaseModel):
    """
    Unified astronomical GNN model.

    Supports:
    - Node classification/regression
    - Graph classification/regression
    - Link prediction
    - Multiple convolution types (GCN, GAT, GIN, SAGE, Transformer)
    - Flexible pooling strategies
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        pooling: str = "mean",
        task: str = "node_classification",
        activation: str = "gelu",
        norm: str = "layer",
        residual: bool = True,
        metadata: tuple = (),  # (node_types, edge_types) for HeteroData
        **kwargs,
    ):
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.num_layers = num_layers
        self.conv_type = conv_type.lower()
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.pooling = pooling.lower()
        self.residual = residual
        self.is_hetero = metadata is not None

        # Always at least 2 classes for node classification
        if task == "node_classification" and num_classes < 2:
            self.num_classes = 2
        else:
            self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Build convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if self.is_hetero:
            self.hetero_layers = nn.ModuleList(
                [HeteroGNNLayer(metadata, hidden_dim) for _ in range(num_layers)]
            )

        for i in range(num_layers):
            # Convolution layer
            conv = self._build_conv_layer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                is_first=(i == 0),
                is_last=(i == num_layers - 1),
            )
            self.convs.append(conv)

            # Normalization layer
            if norm == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.Identity())

            # Dropout
            self.dropouts.append(nn.Dropout(dropout))

        # Activation function
        self.activation = self._get_activation(activation)

        # Output projection (must use self.num_classes)
        self.output_proj = nn.Linear(hidden_dim, self.num_classes)

    def _build_conv_layer(
        self,
        in_dim: int,
        out_dim: int,
        is_first: bool,
        is_last: bool,
    ) -> nn.Module:
        """Build a convolution layer based on conv_type."""
        if self.conv_type == "gcn":
            return GCNConv(in_dim, out_dim)

        elif self.conv_type == "gat":
            # Adjust dimensions for multi-head attention
            if is_last:
                return GATConv(
                    in_dim,
                    out_dim,
                    heads=1,
                    dropout=self.dropout,
                    edge_dim=self.edge_dim,
                )
            else:
                return GATConv(
                    in_dim,
                    out_dim // self.heads,
                    heads=self.heads,
                    dropout=self.dropout,
                    edge_dim=self.edge_dim,
                    concat=True,
                )

        elif self.conv_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim * 2),
                nn.BatchNorm1d(out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim),
            )
            return GINConv(mlp)

        elif self.conv_type == "sage":
            return SAGEConv(in_dim, out_dim)

        elif self.conv_type == "transformer":
            return TransformerConv(
                in_dim,
                out_dim // self.heads,
                heads=self.heads,
                dropout=self.dropout,
                edge_dim=self.edge_dim,
                beta=True,
            )

        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()

    def forward(self, batch):
        if hasattr(batch, "node_types") and hasattr(batch, "edge_types"):
            # True HeteroData
            x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
            for layer in self.hetero_layers:
                x_dict = layer(x_dict, edge_index_dict)
                x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
            # Output projection for each node type
            out_dict = {k: self.output_proj(v) for k, v in x_dict.items()}
            return out_dict
        else:
            return self._forward_homogeneous(batch)

    def _forward_homogeneous(self, batch):
        """Forward pass for regular Data."""
        # Extract features
        x = getattr(batch, "x", None)
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)

        # Comprehensive null checks and error handling
        if x is None:
            # Create default features on the same device as the batch
            num_nodes = getattr(batch, "num_nodes", 1)
            # Get device from batch or use default
            device = getattr(batch, "device", torch.device("cpu"))
            x = torch.ones((num_nodes, self.num_features), device=device)

        # Ensure all tensors are on the same device as the model
        model_device = next(self.parameters()).device

        x = x.to(model_device)

        if edge_index is not None:
            edge_index = edge_index.to(model_device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(model_device)

        # Additional safety checks based on PyTorch best practices
        if x is None or not isinstance(x, torch.Tensor):
            raise ValueError(f"Invalid x tensor: {x}")
        if not x.numel() > 0:
            raise ValueError(f"Empty x tensor with shape: {x.shape}")
        if edge_index is None:
            # Create self-loops if no edges
            num_nodes = x.shape[0]
            edge_index = torch.arange(num_nodes, device=model_device).repeat(2, 1)

        # Input projection
        x = self.input_proj(x)

        # Apply convolution layers
        for i in range(self.num_layers):
            identity = x
            conv = self.convs[i]
            if self.conv_type in ["gat", "transformer"]:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            # Normalization
            x = self.norms[i](x)

            # Activation
            x = self.activation(x)

            # Dropout
            x = self.dropouts[i](x)

            # Residual connection
            if self.residual and i > 0:
                x = x + identity

        # Task-specific output
        if self.task in ["node_classification", "node_regression"]:
            # Always output [batch_size, num_classes] for classification
            out = self.output_proj(x)
            if self.task == "node_classification" and self.num_classes == 1:
                # For binary classification, output logits for 2 classes
                out = torch.cat([-out, out], dim=1)
            return out

        elif self.task in ["graph_classification", "graph_regression"]:
            # Pooling
            batch_idx = getattr(batch, "batch", None)

            if self.pooling == "mean":
                x = global_mean_pool(x, batch_idx)
            elif self.pooling == "max":
                x = global_max_pool(x, batch_idx)
            elif self.pooling == "sum":
                x = global_add_pool(x, batch_idx)
            elif self.pooling == "cat":
                # Concatenate multiple pooling methods
                x_mean = global_mean_pool(x, batch_idx)
                x_max = global_max_pool(x, batch_idx)
                x_sum = global_add_pool(x, batch_idx)
                x = torch.cat([x_mean, x_max, x_sum], dim=-1)

            return self.output_proj(x)

        elif self.task == "link_prediction":
            # For link prediction, we need edge indices to predict
            edge_index_pred = getattr(batch, "edge_index_pred", edge_index)

            # Get embeddings for source and target nodes
            src_emb = x[edge_index_pred[0]]
            dst_emb = x[edge_index_pred[1]]

            # Concatenate and predict
            edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
            return self.output_proj(edge_emb).squeeze(-1)

        else:
            raise ValueError(f"Unknown task: {self.task}")

    def get_embeddings(self, batch: Union[Data, HeteroData, Batch]) -> Tensor:
        """Get node embeddings without final projection."""
        # Handle HeteroData case
        if isinstance(batch, HeteroData):
            raise NotImplementedError("HeteroData support not yet implemented")

        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)

        # Input projection
        x = self.input_proj(x)

        # Apply convolution layers
        for i in range(self.num_layers):
            identity = x
            conv = self.convs[i]
            if self.conv_type in ["gat", "transformer"]:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            # Normalization
            x = self.norms[i](x)

            # Activation
            x = self.activation(x)

            # Dropout (only in training)
            if self.training:
                x = self.dropouts[i](x)

            # Residual connection
            if self.residual and i > 0:
                x = x + identity

        return x


# Factory functions for common configurations
def create_cosmic_web_model(
    num_features: int, num_classes: int, **kwargs
) -> AstroModel:
    """Create model for cosmic web analysis."""
    defaults = {
        "conv_type": "gat",
        "num_layers": 4,
        "hidden_dim": 256,
        "heads": 8,
        "pooling": "cat",
        "task": "graph_classification",
        "dropout": 0.2,
    }
    defaults.update(kwargs)
    return AstroModel(num_features=num_features, num_classes=num_classes, **defaults)


def create_stellar_model(num_features: int, num_classes: int, **kwargs) -> AstroModel:
    """Create model for stellar classification."""
    defaults = {
        "conv_type": "sage",
        "num_layers": 3,
        "hidden_dim": 128,
        "task": "node_classification",
        "dropout": 0.1,
    }
    defaults.update(kwargs)
    return AstroModel(num_features=num_features, num_classes=num_classes, **defaults)


def create_galaxy_model(num_features: int, num_classes: int, **kwargs) -> AstroModel:
    """Create model for galaxy analysis."""
    defaults = {
        "conv_type": "gin",
        "num_layers": 5,
        "hidden_dim": 256,
        "pooling": "mean",
        "task": "graph_classification",
        "dropout": 0.2,
    }
    defaults.update(kwargs)
    return AstroModel(num_features=num_features, num_classes=num_classes, **defaults)


def create_exoplanet_model(num_features: int, num_classes: int, **kwargs) -> AstroModel:
    """Create model for exoplanet detection."""
    defaults = {
        "conv_type": "transformer",
        "num_layers": 3,
        "hidden_dim": 128,
        "heads": 4,
        "task": "node_classification",
        "dropout": 0.1,
    }
    defaults.update(kwargs)
    return AstroModel(num_features=num_features, num_classes=num_classes, **defaults)
