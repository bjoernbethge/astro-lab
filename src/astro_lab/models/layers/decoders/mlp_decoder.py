"""
MLP Decoder Layers for AstroLab Models
=====================================

Simple MLP decoder for autoencoders.
"""

from typing import List

import torch.nn as nn
from torch import Tensor


class MLPDecoder(nn.Module):
    """
    Simple MLP decoder for autoencoders.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_dim

        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass through MLP decoder."""
        return self.mlp(latent)

    def reset_parameters(self):
        """Reset all parameters."""
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
