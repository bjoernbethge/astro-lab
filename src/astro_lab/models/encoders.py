"""
Encoder Modules for AstroLab Models
===================================

This module provides a suite of encoder classes designed to process various types
of astronomical data encapsulated in AstroLab's tensor objects. Each encoder is
a `torch.nn.Module` tailored to a specific data modality, such as photometry,
spectroscopy, or time-series data.

The encoders serve as the initial layers in a larger neural network, transforming
raw tensor data into a fixed-size latent representation suitable for downstream
tasks like classification, regression, or clustering. They are designed to be
modular and composable within the AstroLab model-building framework.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from astro_lab.tensors import (
        AstrometryTensor,
        AstroTensorBase,
        FeatureTensor,
        LightcurveTensor,
        PhotometricTensor,
        Spatial3DTensor,
        SpectralTensor,
        SurveyTensor,
    )


class BaseEncoder(nn.Module):
    """
    Base class for all encoders. It defines the basic interface and provides
    a common initialization for device handling.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim
        self.dropout = dropout

        # Device setup
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Create default MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

        self.to(self.device)

    def forward(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Process data through encoder."""
        # Extract tensor data if needed
        if hasattr(data, "data"):
            tensor_data = data.data
        else:
            tensor_data = data

        # Ensure correct device
        if tensor_data.device != self.device:
            tensor_data = tensor_data.to(self.device)

        # Add batch dimension if needed
        if tensor_data.dim() == 1:
            tensor_data = tensor_data.unsqueeze(0)

        return self.encoder(tensor_data.float())


class PhotometryEncoder(BaseEncoder):
    """Encoder for photometric data."""

    def __init__(self, output_dim: int, input_dim: int = 5, **kwargs):
        # Default 5 bands for photometry
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)


class AstrometryEncoder(BaseEncoder):
    """Encoder for astrometric data."""

    def __init__(self, output_dim: int, input_dim: int = 5, **kwargs):
        # Default: ra, dec, parallax, pmra, pmdec
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)


class SpectroscopyEncoder(BaseEncoder):
    """Encoder for spectroscopic data."""

    def __init__(self, output_dim: int, input_dim: int = 3, **kwargs):
        # Default: teff, logg, feh
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)


class LightcurveEncoder(nn.Module):
    """Encoder for lightcurve data using LSTM."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.to(self.device)

    def forward(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Process lightcurve data."""
        # Extract tensor data if needed
        if hasattr(data, "data"):
            tensor_data = data.data
        else:
            tensor_data = data

        # Ensure correct device
        if tensor_data.device != self.device:
            tensor_data = tensor_data.to(self.device)

        # Ensure correct shape: (batch, seq_len, features)
        if tensor_data.dim() == 1:
            # Single sequence, single feature
            tensor_data = tensor_data.unsqueeze(0).unsqueeze(-1)
        elif tensor_data.dim() == 2:
            # Either (batch, seq_len) or (seq_len, features)
            if tensor_data.size(-1) == self.input_dim:
                # (seq_len, features) -> add batch
                tensor_data = tensor_data.unsqueeze(0)
            else:
                # (batch, seq_len) -> add feature
                tensor_data = tensor_data.unsqueeze(-1)

        # Process through LSTM
        lstm_out, (h_n, _) = self.lstm(tensor_data.float())

        # Use last hidden state from top layer
        final_hidden = h_n[-1]  # (batch, hidden_dim)

        # Project to output dimension
        return self.output_proj(final_hidden)


class Spatial3DEncoder(BaseEncoder):
    """Encodes 3D spatial data."""

    def __init__(self, input_dim: int = 3, output_dim: int = 64, **kwargs: Any):
        super().__init__(input_dim, output_dim, **kwargs)


# Encoder registry for the model factory
ENCODER_REGISTRY = {
    "photometry": PhotometryEncoder,
    "spectroscopy": SpectroscopyEncoder,
    "astrometry": AstrometryEncoder,
    "lightcurve": LightcurveEncoder,
    "spatial_3d": Spatial3DEncoder,
}


def create_encoder(encoder_type: str, **kwargs: Any) -> BaseEncoder:
    """
    Factory function to create an encoder based on its type.

    Args:
        encoder_type: The type of encoder to create (e.g., 'photometry').
        **kwargs: Arguments to pass to the encoder's constructor.

    Returns:
        An instance of the specified encoder.

    Raises:
        ValueError: If the encoder_type is not found in the registry.
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: '{encoder_type}'. Available encoders are: {list(ENCODER_REGISTRY.keys())}"
        )

    encoder_class = ENCODER_REGISTRY[encoder_type]
    return encoder_class(**kwargs)
