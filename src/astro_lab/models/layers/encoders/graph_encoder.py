"""
Graph Encoder Layers for AstroLab Models
=======================================

Graph neural network encoder layers.
"""

from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.aggr as aggr
from torch import Tensor

# PyTorch Geometric
from torch_geometric import EdgeIndex

from ..convolution import FlexibleGraphConv


class ModernGraphEncoder(nn.Module):
    """
    Graph encoder with PyG 2025 features.

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

            layer = FlexibleGraphConv(
                in_channels=layer_in,
                out_channels=layer_out,
                conv_type=conv_type,
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
