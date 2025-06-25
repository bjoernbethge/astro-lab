"""Simple output head functions for AstroLab models."""

import inspect
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import filter_kwargs
from .layers import create_mlp


class ClassificationHead(nn.Module):
    """
    Simple classification head for multi-class classification tasks.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = create_mlp(
            input_dim, num_classes, hidden_dims=[input_dim // 2], dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)


class RegressionHead(nn.Module):
    """
    Simple regression head for continuous value prediction.

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (default: 1)
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.regressor = create_mlp(
            input_dim, output_dim, hidden_dims=[input_dim // 2], dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Regression output of shape (batch_size, output_dim)
        """
        return self.regressor(x)


class PeriodDetectionHead(nn.Module):
    """
    Head for asteroid period detection with uncertainty estimation.

    Args:
        input_dim: Input feature dimension
        dropout: Dropout rate for regularization
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        # Period detection: predict period value and uncertainty
        self.period_net = create_mlp(
            input_dim,
            2,  # period and uncertainty
            hidden_dims=[input_dim // 2, input_dim // 4],
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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


class ShapeModelingHead(nn.Module):
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
        self.shape_net = create_mlp(
            input_dim,
            num_harmonics * 2,  # Real and imaginary parts
            hidden_dims=[input_dim // 2],
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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


# Simple dictionary mapping
OUTPUT_HEADS: Dict[str, Type[nn.Module]] = {
    "classification": ClassificationHead,
    "regression": RegressionHead,
    "period_detection": PeriodDetectionHead,
    "shape_modeling": ShapeModelingHead,
}


def create_output_head(
    head_type: str, input_dim: int, output_dim: Optional[int] = None, **kwargs
) -> nn.Module:
    """
    Factory function for creating output heads with robust parameter filtering.

    Args:
        head_type: Type of output head ('classification', 'regression', etc.)
        input_dim: Input feature dimension
        output_dim: Output dimension (optional, depends on head type)
        **kwargs: Additional parameters for the head

    Returns:
        Configured output head module

    Raises:
        ValueError: If head_type is not supported
    """
    if head_type not in OUTPUT_HEADS:
        available = list(OUTPUT_HEADS.keys())
        raise ValueError(f"Unknown head type: {head_type}. Available: {available}")

    head_class = OUTPUT_HEADS[head_type]

    # Prepare kwargs for each head type
    if head_type == "classification":
        config = {"input_dim": input_dim, "num_classes": output_dim or 2}
        config.update(kwargs)
        filtered = filter_kwargs(head_class, **config)
        return head_class(**filtered)
    elif head_type == "regression":
        config = {"input_dim": input_dim, "output_dim": output_dim or 1}
        config.update(kwargs)
        filtered = filter_kwargs(head_class, **config)
        return head_class(**filtered)
    else:
        # Period detection and shape modeling don't need output_dim
        config = {"input_dim": input_dim}
        config.update(kwargs)
        filtered = filter_kwargs(head_class, **config)
        return head_class(**filtered)
