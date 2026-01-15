"""
Specialized Heads for AstroLab Models
====================================

Specialized output heads for astronomical tasks.
"""

from typing import Dict

import torch
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


class PeriodDetectionHead(BaseLayer):
    """
    Head for asteroid period detection with uncertainty estimation.

    Args:
        input_dim: Input feature dimension
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        # Period detection: predict period value and uncertainty
        self.period_net = _create_mlp(
            input_dim,
            2,  # period and uncertainty
            hidden_dims=[input_dim // 2, input_dim // 4],
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass for period detection.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary with 'period' and 'uncertainty' tensors
        """
        output = self.period_net(x)
        period = torch.abs(output[:, 0:1])  # Ensure positive
        uncertainty = torch.abs(output[:, 1:2])
        return {"period": period, "uncertainty": uncertainty}


class ShapeModelingHead(BaseLayer):
    """
    Head for asteroid shape modeling using spherical harmonics.

    Args:
        input_dim: Input feature dimension
        num_harmonics: Number of spherical harmonic coefficients
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim: int, num_harmonics: int = 10, dropout: float = 0.1):
        super().__init__()
        self.num_harmonics = num_harmonics
        # Predict spherical harmonic coefficients
        self.shape_net = _create_mlp(
            input_dim,
            num_harmonics * 2,  # Real and imaginary parts
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass for shape modeling.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary with 'real_coeffs' and 'imag_coeffs' tensors
        """
        coeffs = self.shape_net(x)
        real_coeffs = coeffs[:, : self.num_harmonics]
        imag_coeffs = coeffs[:, self.num_harmonics :]
        return {"real_coeffs": real_coeffs, "imag_coeffs": imag_coeffs}
