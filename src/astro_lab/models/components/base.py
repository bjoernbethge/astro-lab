"""
Base Components for AstroLab Models - PyG 2025 Enhanced
=============================================================

Reusable components leveraging:
- EdgeIndex support with metadata caching
- Index class for efficient 1D indexing
- VariancePreservingAggregation
- torch.compile compatibility
- normalization techniques
"""

from typing import Dict, List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.aggr as aggr
from torch import Tensor

# PyTorch Geometric
from torch_geometric import EdgeIndex
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# Import our modern layers
# from .layers import AstroGraphLayer


class EnhancedMLPBlock(nn.Module):
    """
    MLP block with modern features.

    Features:
    - Multiple normalization options
    - activation functions
    - Residual connections
    - torch.compile compatibility
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
        residual: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.residual = residual and (in_dim == out_dim)

        # Build layers
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))

            # Skip norm and activation on last layer
            if i < len(dims) - 2:
                # Normalization
                if norm_type == "layer":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                elif norm_type == "batch":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif norm_type == "instance":
                    layers.append(nn.InstanceNorm1d(dims[i + 1]))

                # Activation
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "silu":
                    layers.append(nn.SiLU(inplace=True))
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.01, inplace=True))

                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP block."""
        identity = x

        for i, layer in enumerate(self.layers):
            # Only pass x to LayerNorm, never batch
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x)

            # Add residual connection if enabled and not the last layer
            if self.residual and i < len(self.layers) - 1 and x.shape == identity.shape:
                x = x + identity
                identity = x

        return x

    def reset_parameters(self):
        """Reset all parameters."""
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class ModernGraphEncoder(nn.Module):
    """
    graph encoder with PyG 2025 features.

    Features:
    - EdgeIndex support with metadata caching
    - VariancePreservingAggregation
    - Multiple conv layer types
    - Residual connections
    - Flexible normalization
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        conv_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        norm_type: str = "layer",
        residual: bool = True,
        aggr: Union[str, aggr.Aggregation] = "mean",
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.residual = residual
        self.conv_type = conv_type

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Graph layers using our modern implementation
        self.layers = nn.ModuleList()

        # Layer dimensions
        dims = [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            layer_in = hidden_channels if i == 0 else dims[i - 1]
            layer_out = dims[i]

            layer = create_graph_layer(
                layer_type=conv_type,
                in_channels=layer_in,
                out_channels=layer_out,
                heads=heads,
                edge_dim=edge_dim,
                **kwargs,
            )

            self.layers.append(layer)

        # Output projection if needed
        if out_channels != hidden_channels and num_layers == 1:
            self.output_proj = nn.Linear(hidden_channels, out_channels)
        else:
            self.output_proj = nn.Identity()

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, EdgeIndex],
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through graph encoder."""

        # Input projection
        x = self.input_proj(x)
        x = F.gelu(x)

        # Graph layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)

        # Output projection
        x = self.output_proj(x)

        return x

    def reset_parameters(self):
        """Reset all parameters."""
        self.input_proj.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        if hasattr(self.output_proj, "reset_parameters"):
            self.output_proj.reset_parameters()


class AdvancedTemporalEncoder(nn.Module):
    """
    temporal encoder with attention mechanisms.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        encoder_type: str = "lstm",
        bidirectional: bool = True,
        attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        self.attention = attention

        # RNN layers
        if encoder_type == "lstm":
            self.rnn = nn.LSTM(
                in_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif encoder_type == "gru":
            self.rnn = nn.GRU(
                in_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Calculate RNN output dimension
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention mechanism
        if attention:
            self.attention_layer = nn.MultiheadAttention(
                rnn_out_dim, num_heads=8, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(rnn_out_dim)

        # Output projection
        self.output_proj = EnhancedMLPBlock(
            rnn_out_dim, out_dim, hidden_dim=rnn_out_dim // 2, dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through temporal encoder."""

        # RNN encoding
        rnn_out, _ = self.rnn(x)

        # Attention mechanism
        if self.attention:
            attn_out, _ = self.attention_layer(rnn_out, rnn_out, rnn_out)
            rnn_out = self.attention_norm(rnn_out + attn_out)

        # Global pooling over time dimension
        if self.bidirectional:
            # Use mean pooling for bidirectional
            temporal_features = rnn_out.mean(dim=1)
        else:
            # Use last time step for unidirectional
            temporal_features = rnn_out[:, -1]

        # Output projection
        out = self.output_proj(temporal_features)

        return out

    def reset_parameters(self):
        """Reset all parameters."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        if self.attention:
            self.attention_layer._reset_parameters()
            self.attention_norm.reset_parameters()

        self.output_proj.reset_parameters()


class PointNetEncoder(nn.Module):
    """
    PointNet encoder with set attention.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.1,
        use_attention: bool = True,
        pooling: str = "max",
    ):
        super().__init__()

        self.use_attention = use_attention
        self.pooling = pooling

        # Point-wise MLPs
        layers = []
        dims = [in_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Conv1d(dims[i], dims[i + 1], 1),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

        self.point_encoder = nn.Sequential(*layers)

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dims[-1], num_heads=8, dropout=dropout, batch_first=True
            )

        # Output projection
        self.output_proj = EnhancedMLPBlock(hidden_dims[-1], out_dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through PointNet encoder."""

        # x: [batch, num_points, features]
        batch_size, num_points, _ = x.shape

        # Point-wise encoding
        x = x.transpose(1, 2)  # [batch, features, num_points]
        x = self.point_encoder(x)
        x = x.transpose(1, 2)  # [batch, num_points, features]

        # Attention mechanism
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out

        # Global pooling
        if self.pooling == "max":
            global_features = x.max(dim=1)[0]
        elif self.pooling == "mean":
            global_features = x.mean(dim=1)
        elif self.pooling == "sum":
            global_features = x.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Output projection
        out = self.output_proj(global_features)

        return out

    def reset_parameters(self):
        """Reset all parameters."""
        for layer in self.point_encoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        if self.use_attention:
            self.attention._reset_parameters()

        self.output_proj.reset_parameters()


class TaskSpecificHead(nn.Module):
    """
    task-specific output head with advanced features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        task_type: str,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_layers: int = 2,
        use_uncertainty: bool = False,
    ):
        super().__init__()

        self.task_type = task_type
        self.use_uncertainty = use_uncertainty

        hidden_dim = hidden_dim or max(in_dim // 2, out_dim * 2)

        # Main prediction head
        self.prediction_head = EnhancedMLPBlock(
            in_dim,
            out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Uncertainty head (for regression tasks)
        if use_uncertainty and task_type in [
            "regression",
            "node_regression",
            "graph_regression",
        ]:
            self.uncertainty_head = EnhancedMLPBlock(
                in_dim,
                out_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=max(1, num_layers - 1),
                dropout=dropout,
            )

    def forward(self, x: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward pass through task head."""

        predictions = self.prediction_head(x)

        if self.use_uncertainty and hasattr(self, "uncertainty_head"):
            uncertainties = F.softplus(self.uncertainty_head(x)) + 1e-6
            return {"predictions": predictions, "uncertainties": uncertainties}

        return predictions

    def reset_parameters(self):
        """Reset all parameters."""
        self.prediction_head.reset_parameters()
        if hasattr(self, "uncertainty_head"):
            self.uncertainty_head.reset_parameters()


def create_encoder(
    encoder_type: str, in_dim: int, hidden_dim: int, out_dim: int, **kwargs
) -> nn.Module:
    """
    Factory function for creating encoders.

    Args:
        encoder_type: Type of encoder ('graph', 'temporal', 'pointnet', 'mlp')
        in_dim: Input dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        **kwargs: Additional arguments

    Returns:
        Configured encoder module
    """

    if encoder_type == "graph":
        return ModernGraphEncoder(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            **kwargs,
        )

    elif encoder_type == "temporal":
        return AdvancedTemporalEncoder(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, **kwargs
        )

    elif encoder_type == "pointnet":
        hidden_dims = kwargs.get("hidden_dims", [hidden_dim, hidden_dim * 2])
        return PointNetEncoder(
            in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim, **kwargs
        )

    elif encoder_type == "mlp":
        return EnhancedMLPBlock(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, **kwargs
        )

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


def create_output_head(
    task_type: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    Factory function for creating task-specific output heads.

    Args:
        task_type: Type of task
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dim: Hidden dimension
        dropout: Dropout rate
        **kwargs: Additional arguments

    Returns:
        Configured output head
    """

    return TaskSpecificHead(
        in_dim=in_dim,
        out_dim=out_dim,
        task_type=task_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        **kwargs,
    )


class GraphPooling(nn.Module):
    """
    graph pooling with multiple strategies using our modern implementation.
    """

    def __init__(
        self,
        pooling_type: str = "mean",
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
    ):
        super().__init__()

        self.pooling_type = pooling_type

        if pooling_type == "attention" and hidden_dim is not None:
            self.pooling = nn.Identity()
        elif pooling_type == "multi":
            self.pooling = nn.Identity()
        else:
            self.pooling = None

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """Apply graph pooling."""

        if self.pooling is not None:
            return self.pooling(x, batch)

        # Standard pooling
        if batch is None:
            # Single graph case
            if self.pooling_type == "mean":
                return x.mean(dim=0, keepdim=True)
            elif self.pooling_type == "max":
                return x.max(dim=0)[0].unsqueeze(0)
            elif self.pooling_type == "sum":
                return x.sum(dim=0, keepdim=True)
        else:
            # Batched graphs
            if self.pooling_type == "mean":
                return global_mean_pool(x, batch)
            elif self.pooling_type == "max":
                return global_max_pool(x, batch)
            elif self.pooling_type == "sum":
                return global_add_pool(x, batch)

        return x

    def reset_parameters(self):
        """Reset all parameters."""
        if self.pooling is not None and hasattr(self.pooling, "reset_parameters"):
            self.pooling.reset_parameters()


def create_graph_layer(layer_type: str, in_channels: int, out_channels: int, **kwargs):
    """Create a graph layer based on type."""
    # Remove norm_type from kwargs if present
    kwargs = {k: v for k, v in kwargs.items() if k != "norm_type"}
    if layer_type == "gcn":
        from torch_geometric.nn import GCNConv

        return GCNConv(in_channels, out_channels)
    elif layer_type == "gat":
        from torch_geometric.nn import GATConv

        return GATConv(in_channels, out_channels, **kwargs)
    elif layer_type == "sage":
        from torch_geometric.nn import SAGEConv

        return SAGEConv(in_channels, out_channels)
    elif layer_type == "graph":
        from torch_geometric.nn import GraphConv

        return GraphConv(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
