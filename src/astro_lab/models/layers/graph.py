"""
Graph Layers for AstroLab Models
===============================

Graph neural network layers and pooling operations.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

# PyTorch Geometric
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class GraphPooling(nn.Module):
    """
    Graph pooling with multiple strategies.
    """

    def __init__(
        self,
        pooling_type: str = "mean",
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
    ):
        super().__init__()

        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim

        # Attention-based pooling
        if pooling_type == "attention" and hidden_dim is not None:
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=num_heads, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """Forward pass through pooling layer."""

        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "sum":
            return global_add_pool(x, batch)
        elif self.pooling_type == "attention":
            # Attention-based pooling
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            return global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def reset_parameters(self):
        """Reset all parameters."""
        if hasattr(self, "attention"):
            self.attention._reset_parameters()
        if hasattr(self, "norm"):
            self.norm.reset_parameters()
