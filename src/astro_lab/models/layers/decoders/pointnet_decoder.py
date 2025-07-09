"""
PointNet Decoder Layers for AstroLab Models
==========================================

PointNet-based decoder layers for autoencoders.
"""

from typing import List

import torch.nn as nn
from torch import Tensor


class PointNetDecoder(nn.Module):
    """
    PointNet decoder for autoencoders.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        num_points: int,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.num_points = num_points
        self.use_attention = use_attention

        # Latent projection to initial features
        self.latent_proj = nn.Linear(latent_dim, hidden_dims[0])

        # Point-wise MLPs (reverse of encoder)
        layers = []
        dims = [hidden_dims[0]] + hidden_dims[1:] + [out_dim]

        for i in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Conv1d(dims[i], dims[i + 1], 1),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

        self.point_decoder = nn.Sequential(*layers)

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dims[-1], num_heads=8, dropout=dropout, batch_first=True
            )

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass through PointNet decoder."""

        latent.size(0)

        # Project latent to initial features for all points
        x = self.latent_proj(latent)  # [batch, hidden_dim]
        x = x.unsqueeze(1).repeat(
            1, self.num_points, 1
        )  # [batch, num_points, hidden_dim]

        # Point-wise decoding
        x = x.transpose(1, 2)  # [batch, hidden_dim, num_points]
        x = self.point_decoder(x)
        x = x.transpose(1, 2)  # [batch, num_points, out_dim]

        # Attention mechanism
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out

        return x

    def reset_parameters(self):
        """Reset all parameters."""
        self.latent_proj.reset_parameters()

        for layer in self.point_decoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        if self.use_attention:
            self.attention._reset_parameters()
