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

        # Ensure parameters require gradients
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.to(self.device)

    def forward(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Process data through encoder. Expects: batch x features, float32."""
        # Extract tensor data if needed
        if hasattr(data, "data"):
            tensor_data = data.data
        else:
            tensor_data = data

        # Move to correct device if needed
        if tensor_data.device != self.device:
            tensor_data = tensor_data.to(self.device)
        
        # Convert to float32 if needed
        if tensor_data.dtype != torch.float32:
            tensor_data = tensor_data.float()
            
        assert tensor_data.dim() == 2, (
            f"Input must be 2D (batch, features), got shape {tensor_data.shape}"
        )
        assert tensor_data.shape[1] == self.input_dim, (
            f"Input must have {self.input_dim} features, got {tensor_data.shape[1]}"
        )

        return self.encoder(tensor_data)


class PhotometryEncoder(BaseEncoder):
    """Encoder for photometric data with NaN handling."""

    def __init__(self, output_dim: int, input_dim: int = 5, **kwargs):
        # Default 5 bands for photometry
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)

    def forward(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Process photometric data with NaN handling. Expects: batch x features, float32."""
        # Extract tensor data if needed
        if hasattr(data, "data"):
            tensor_data = data.data
        else:
            tensor_data = data

        # Move to correct device if needed
        if tensor_data.device != self.device:
            tensor_data = tensor_data.to(self.device)
        
        # Convert to float32 if needed
        if tensor_data.dtype != torch.float32:
            tensor_data = tensor_data.float()
            
        assert tensor_data.dim() == 2, (
            f"Input must be 2D (batch, features), got shape {tensor_data.shape}"
        )
        assert tensor_data.shape[1] == self.input_dim, (
            f"Input must have {self.input_dim} features, got {tensor_data.shape[1]}"
        )

        # Handle NaN values by replacing with zeros
        if torch.isnan(tensor_data).any():
            tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=0.0, neginf=0.0)

        return self.encoder(tensor_data)


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
    """Encoder for lightcurve data using LSTM with attention."""

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

        # Attention mechanism for better feature extraction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Output projection with more capacity
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.to(self.device)

    def forward(self, data: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Process lightcurve data. Expects: batch x seq_len x features, float32."""
        # Extract tensor data if needed
        if hasattr(data, "data"):
            tensor_data = data.data
        else:
            tensor_data = data

        # Move to correct device if needed
        if tensor_data.device != self.device:
            tensor_data = tensor_data.to(self.device)
        
        # Convert to float32 if needed
        if tensor_data.dtype != torch.float32:
            tensor_data = tensor_data.float()
            
        assert tensor_data.dim() == 3, (
            f"Input must be 3D (batch, seq_len, features), got shape {tensor_data.shape}"
        )
        assert tensor_data.shape[2] == self.input_dim, (
            f"Input must have {self.input_dim} features, got {tensor_data.shape[2]}"
        )

        # Process through LSTM
        lstm_out, (h_n, _) = self.lstm(tensor_data)

        # Apply attention to LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last hidden state from top layer + attention output
        final_hidden = h_n[-1]  # (batch, hidden_dim)
        attn_pooled = attn_out.mean(dim=1)  # (batch, hidden_dim)

        # Combine both representations
        combined = final_hidden + attn_pooled

        # Project to output dimension
        return self.output_proj(combined)


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
