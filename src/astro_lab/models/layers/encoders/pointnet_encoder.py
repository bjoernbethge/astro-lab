"""
PointNet Encoder Layers for AstroLab Models
==========================================

PointNet-based encoder layers for point cloud processing.
"""

from typing import List

import torch.nn as nn
from torch import Tensor


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
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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
