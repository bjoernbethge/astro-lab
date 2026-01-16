"""Unified AstroLab Model - GNN implementation with native TensorDict support."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
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

from astro_lab.models.layers.encoders.temporal_encoder import AdvancedTemporalEncoder
from astro_lab.models.layers.hetero import HeteroGNNLayer
from astro_lab.models.layers.normalization import AdaptiveNormalization
from astro_lab.models.layers.point_cloud import AstroPointCloudLayer
from astro_lab.models.layers.pooling import AttentivePooling, HierarchicalPooling

from .base_model import AstroBaseModel


class AstroModel(AstroBaseModel):
    """
    Unified AstroLab Model for astronomical data processing.

    Supports multiple GNN architectures and data formats with native TensorDict integration.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        conv_type: str = "GAT",
        num_layers: int = 3,
        dropout: float = 0.1,
        heads: int = 4,
        pooling: Optional[str] = None,
        task: str = "node_classification",
        metadata: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None,
        temporal_features: Optional[int] = None,
        point_cloud_features: Optional[int] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        max_epochs: int = 100,
        activation: str = "relu",
        norm: str = "layer",
        residual: bool = True,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the AstroModel.

        Args:
            num_features: Number of input features
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes/dimensions
            conv_type: Type of convolution layer (GAT, GCN, GIN, SAGE, Transformer)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            heads: Number of attention heads (for GAT/Transformer)
            pooling: Global pooling type (mean, max, add, attention, hierarchical)
            task: Task type (node_classification, graph_classification, regression)
            metadata: Metadata for heterogeneous graphs
            temporal_features: Number of temporal features
            point_cloud_features: Number of point cloud features
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            optimizer: Optimizer type
            scheduler: Learning rate scheduler type
            max_epochs: Maximum number of epochs
            activation: Activation function (relu, elu, leaky_relu)
            norm: Normalization type (layer, batch, adaptive)
            residual: Whether to use residual connections
            edge_dim: Edge feature dimension
        """
        # Initialize base model
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            **kwargs,
        )

        self.conv_type = conv_type.upper()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.pooling = pooling
        self.metadata = metadata
        self.activation = activation
        self.norm = norm
        self.residual = residual
        self.edge_dim = edge_dim

        # Build the model
        self._build_model(temporal_features, point_cloud_features)

    def _build_model(
        self,
        temporal_features: Optional[int] = None,
        point_cloud_features: Optional[int] = None,
    ):
        """Build the model architecture."""

        # Input projection
        self.input_proj = nn.Linear(self.num_features, self.hidden_dim)

        # Optional temporal encoder
        if temporal_features:
            self.temporal_encoder = AdvancedTemporalEncoder(
                input_size=temporal_features,
                hidden_size=self.hidden_dim,
                num_layers=2,
                dropout=self.dropout,
            )
        else:
            self.temporal_encoder = None

        # Optional point cloud layer
        if point_cloud_features:
            self.point_cloud_layer = AstroPointCloudLayer(
                in_channels=point_cloud_features,
                out_channels=self.hidden_dim,
                k=16,
                aggr="max",
            )
        else:
            self.point_cloud_layer = None

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_channels = self.hidden_dim

        for i in range(self.num_layers):
            out_channels = self.hidden_dim

            # Create convolution layer based on type
            if self.conv_type == "GAT":
                conv = GATConv(
                    in_channels,
                    out_channels // self.heads,
                    heads=self.heads,
                    dropout=self.dropout,
                    concat=True if i < self.num_layers - 1 else False,
                    edge_dim=self.edge_dim,
                )
                if i == self.num_layers - 1:
                    out_channels = (out_channels // self.heads) * self.heads

            elif self.conv_type == "GCN":
                conv = GCNConv(
                    in_channels, out_channels, improved=True, add_self_loops=True
                )

            elif self.conv_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    self._get_activation(),
                    nn.Linear(out_channels, out_channels),
                )
                conv = GINConv(mlp)

            elif self.conv_type == "SAGE":
                conv = SAGEConv(in_channels, out_channels, normalize=True)

            elif self.conv_type == "TRANSFORMER":
                conv = TransformerConv(
                    in_channels,
                    out_channels // self.heads,
                    heads=self.heads,
                    dropout=self.dropout,
                    concat=True if i < self.num_layers - 1 else False,
                    edge_dim=self.edge_dim,
                )
                if i == self.num_layers - 1:
                    out_channels = (out_channels // self.heads) * self.heads
            else:
                raise ValueError(f"Unknown conv_type: {self.conv_type}")

            self.convs.append(conv)

            # Normalization
            if self.norm == "layer":
                self.norms.append(nn.LayerNorm(out_channels))
            elif self.norm == "batch":
                self.norms.append(nn.BatchNorm1d(out_channels))
            elif self.norm == "adaptive":
                self.norms.append(AdaptiveNormalization(out_channels))
            else:
                self.norms.append(nn.Identity())

            self.dropouts.append(nn.Dropout(self.dropout))
            in_channels = out_channels

        # Pooling for graph-level tasks
        if self.pooling:
            if self.pooling == "mean":
                self.pool = global_mean_pool
            elif self.pooling == "max":
                self.pool = global_max_pool
            elif self.pooling == "add":
                self.pool = global_add_pool
            elif self.pooling == "attention":
                self.pool = AttentivePooling(
                    in_channels, hidden_channels=self.hidden_dim
                )
            elif self.pooling == "hierarchical":
                self.pool = HierarchicalPooling(in_channels, ratio=0.5, method="top_k")
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling}")
        else:
            self.pool = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            self._get_activation(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        # Heterogeneous support
        if self.metadata:
            self.hetero_layer = HeteroGNNLayer(
                metadata=self.metadata,
                hidden_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_layers=2,
            )
        else:
            self.hetero_layer = None

    def _get_activation(self):
        """Get activation function."""
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "elu":
            return nn.ELU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()

    def forward(
        self,
        x: Union[Tensor, TensorDict, Data, Batch, HeteroData],
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input features or data object
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch indices for graph-level tasks
            pos: Positional information

        Returns:
            Output tensor
        """
        # Handle different input types
        if isinstance(x, (Data, Batch)):
            data = x
            x = data.x  # type: ignore[attr-defined]
            edge_index = data.edge_index  # type: ignore[attr-defined]
            edge_attr = getattr(data, "edge_attr", None)
            batch = getattr(data, "batch", None)
            pos = getattr(data, "pos", None)
        elif isinstance(x, HeteroData):
            # Handle heterogeneous data
            if self.hetero_layer:
                return self.hetero_layer(x)
            else:
                raise ValueError("Model not configured for heterogeneous data")
        elif isinstance(x, TensorDict):
            # Convert TensorDict to tensor
            x = self.from_tensordict(x)

        # Input projection
        h = self.input_proj(x)

        # Optional temporal encoding
        if self.temporal_encoder and "temporal_features" in kwargs:
            temporal_h = self.temporal_encoder(kwargs["temporal_features"])
            h = h + temporal_h

        # Optional point cloud processing
        if self.point_cloud_layer and pos is not None:
            pc_h = self.point_cloud_layer(h, pos, batch)
            h = h + pc_h

        # GNN layers
        for i, (conv, norm, dropout) in enumerate(
            zip(self.convs, self.norms, self.dropouts)
        ):
            h_prev = h

            # Convolution
            if self.conv_type in ["GAT", "TRANSFORMER"] and edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)

            # Normalization and activation
            h = norm(h)
            activation = self._get_activation()
            h = activation(h)
            h = dropout(h)

            # Residual connection
            if self.residual and h_prev.shape == h.shape:
                h = h + h_prev

        # Pooling for graph-level tasks
        if self.pool is not None and batch is not None:
            if isinstance(self.pool, (AttentivePooling, HierarchicalPooling)):
                if isinstance(self.pool, AttentivePooling):
                    h = self.pool(h, batch=batch)
                else:  # HierarchicalPooling
                    h, _, _, batch, _, _ = self.pool(h, edge_index, batch=batch)
            else:
                h = self.pool(h, batch)

        # Output projection
        out = self.output_proj(h)

        return out

    def predict(self, x: Union[Data, Batch, TensorDict], **kwargs) -> Tensor:
        """
        Make predictions on input data.

        Args:
            x: Input data

        Returns:
            Predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, **kwargs)
            if "classification" in self.task:
                return logits.argmax(dim=-1)
            else:
                return logits
