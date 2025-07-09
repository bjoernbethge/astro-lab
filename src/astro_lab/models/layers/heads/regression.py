"""
Regression Heads for AstroLab Models
===================================

Regression output heads for continuous value prediction.
"""

import torch.nn as nn
from torch import Tensor

from ..base import BaseLayer


def _create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list = None,
    dropout: float = 0.1,
):
    """Create a simple MLP using standard PyTorch layers."""
    if hidden_dims is None:
        hidden_dims = [input_dim // 2]

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


class RegressionHead(BaseLayer):
    """
    Regression head for continuous value prediction.

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (default: 1)
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.regressor = _create_mlp(input_dim, output_dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for regression.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Regression output of shape (batch_size, output_dim)
        """
        return self.regressor(x)
