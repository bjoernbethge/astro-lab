"""
Graph Decoder Layers for AstroLab Models
=======================================

Graph neural network decoder layers for autoencoders.
"""

from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# PyTorch Geometric
from torch_geometric import EdgeIndex

from ..convolution import FlexibleGraphConv


class ModernGraphDecoder(nn.Module):
    """
    Graph decoder for autoencoders.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        conv_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.conv_type = conv_type

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_channels)

        # Graph layers
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

        # Output projection
        if out_channels != hidden_channels and num_layers == 1:
            self.output_proj = nn.Linear(hidden_channels, out_channels)
        else:
            self.output_proj = nn.Identity()

    def forward(
        self,
        latent: Tensor,
        edge_index: Union[Tensor, EdgeIndex],
        batch: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through graph decoder."""

        # Project latent to initial features
        latent.size(0)
        if batch is not None:
            batch.size(0)
            # Expand latent to all nodes in batch
            x = latent[batch]  # [num_nodes, latent_dim]
        else:
            latent.size(0)
            x = latent

        # Project to hidden dimension
        x = self.latent_proj(x)
        x = F.gelu(x)

        # Graph layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)

        # Output projection
        x = self.output_proj(x)

        return x

    def reset_parameters(self):
        """Reset all parameters."""
        self.latent_proj.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        if hasattr(self.output_proj, "reset_parameters"):
            self.output_proj.reset_parameters()
