"""
Classification Heads for AstroLab Models
=======================================

Classification output heads for multi-class classification tasks.
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


class ClassificationHead(BaseLayer):
    """
    Classification head for multi-class classification tasks.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = _create_mlp(input_dim, num_classes, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)
