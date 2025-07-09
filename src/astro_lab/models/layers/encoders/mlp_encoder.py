"""
MLP Encoder Layers for AstroLab Models
=====================================

Simple MLP encoder for tabular data.
"""

from typing import List

import torch.nn as nn
from torch import Tensor


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for tabular data.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.in_dim = in_dim
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

        # Build layers
        layers = []
        prev_dim = in_dim

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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP encoder."""
        return self.mlp(x)

    def reset_parameters(self):
        """Reset all parameters."""
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
