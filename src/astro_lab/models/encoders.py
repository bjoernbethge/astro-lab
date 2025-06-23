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
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import torch
import torch.nn as nn

from .layers import LayerFactory

if TYPE_CHECKING:
    from astro_lab.tensors import (
        AstrometryTensor,
        FeatureTensor,
        LightcurveTensor,
        PhotometricTensor,
        Spatial3DTensor,
        SpectralTensor,
        SurveyTensor,
        AstroTensorBase,
    )


class BaseEncoder(nn.Module):
    """
    Base class for all encoders. It defines the basic interface and provides
    a common initialization for device handling.
    """
    def __init__(self, input_dim: int, output_dim: int, device: Optional[Union[str, torch.device]] = None, **kwargs):
        # Extract encoder-specific parameters before calling super()
        self.hidden_dim = kwargs.pop('hidden_dim', 128)  # Default hidden_dim
        self.dropout = kwargs.pop('dropout', 0.1)  # Default dropout
        
        super().__init__(**kwargs)
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create layers and move to device
        self.layers = self._create_layers()
        if self.layers is not None:
            self.layers = self.layers.to(self.device)
        
        # Move entire encoder to device
        self.to(self.device)

    def _create_layers(self) -> nn.Module:
        """
        Creates the neural network layers for the encoder. Subclasses should
        implement this method to define their specific architecture.
        """
        # Default implementation is a simple MLP
        return nn.Sequential(
            nn.Linear(self.input_dim, (self.input_dim + self.output_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.input_dim + self.output_dim) // 2, self.output_dim),
        )

    def forward(self, tensor: Any, *args, **kwargs) -> torch.Tensor:
        """
        Defines the forward pass of the encoder. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def process_batch(self, data: Union[torch.Tensor, "AstroTensorBase"]) -> torch.Tensor:
        """Process a batch of data through the encoder."""
        if hasattr(data, 'data'):
            tensor_data = data.data
        else:
            tensor_data = data
        
        # Move data to correct device
        if tensor_data.device != self.device:
            tensor_data = tensor_data.to(self.device)
            
        return self.layers(tensor_data.float())


class LightcurveEncoder(BaseEncoder):
    """Encodes lightcurve data using an LSTM."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, **kwargs):
        # Extract lightcurve-specific parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Remove parameters that shouldn't go to super()
        kwargs.pop('dropout', None)  # Remove dropout if present
        
        # DO NOT pass hidden_dim to super() to avoid the default 128 override
        super().__init__(input_dim=input_dim, output_dim=output_dim, device=kwargs.pop('device', None))
        # Note: BaseEncoder's _create_layers is not used here, we define layers directly
        
    def _create_layers(self) -> nn.Module:
        """Override to create LSTM and Linear layers."""
        return nn.ModuleDict({
            'lstm': nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True).to(self.device),
            'attention': nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True).to(self.device),
            'linear': nn.Linear(self.hidden_dim, self.output_dim).to(self.device),
            'dropout': nn.Dropout(0.1).to(self.device)
        })

    def forward(self, data: "LightcurveTensor") -> torch.Tensor:
        """Forward pass for lightcurve data."""
        # Reshape for LSTM: [batch, seq_len, features]
        batch_data = data.data.unsqueeze(0)  # Add batch dimension
        
        # Move data to correct device
        if batch_data.device != self.device:
            batch_data = batch_data.to(self.device)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.layers['lstm'](batch_data)
        
        # Apply attention
        attn_out, _ = self.layers['attention'](lstm_out, lstm_out, lstm_out)
        
        # Use dropout
        attn_out = self.layers['dropout'](attn_out)
        
        # Return the attention output with shape [batch, seq_len, hidden_dim]
        # NOT applying the final linear layer to preserve sequence dimension
        return attn_out


class AstrometryEncoder(BaseEncoder):
    """Encodes astrometric data."""
    def __init__(self, input_dim: int, output_dim: int, **kwargs: Any):
        super().__init__(input_dim, output_dim, **kwargs)

    def forward(
        self, astrometry_tensor: "AstrometryTensor", *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Process the astrometric tensor."""
        data = astrometry_tensor.data
        if data.dim() == 1:
            data = data.unsqueeze(0)
        return self.process_batch(data)


class SpectroscopyEncoder(BaseEncoder):
    """Encodes spectroscopic data."""
    def __init__(self, input_dim: int, output_dim: int, **kwargs: Any):
        super().__init__(input_dim, output_dim, **kwargs)

    def forward(
        self, spectroscopy_tensor: "SpectralTensor", *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Process the spectroscopy tensor."""
        data = spectroscopy_tensor.data
        if data.dim() == 1:
            data = data.unsqueeze(0)
        return self.process_batch(data)


class PhotometryEncoder(BaseEncoder):
    """Encodes photometric measurements with band-aware processing."""

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        # Extract photometry-specific parameters
        hidden_dim = kwargs.pop('hidden_dim', 128)  # Extract and remove hidden_dim
        dropout = kwargs.pop('dropout', 0.1)  # Extract and remove dropout
        
        super().__init__(input_dim, output_dim, **kwargs)

    def forward(
        self, photometric_tensor: "PhotometricTensor", *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Process the photometric tensor."""
        data = photometric_tensor.data
        if data.dim() == 1:
            data = data.unsqueeze(0)
        return self.process_batch(data)


class Spatial3DEncoder(BaseEncoder):
    """Encodes 3D spatial data."""
    def __init__(self, input_dim: int, output_dim: int, **kwargs: Any):
        super().__init__(input_dim, output_dim, **kwargs)

    def forward(
        self, spatial_tensor: "Spatial3DTensor", *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Process the 3D spatial tensor."""
        data = spatial_tensor.data
        if data.dim() == 1:
            data = data.unsqueeze(0)
        return self.process_batch(data)

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
        raise ValueError(f"Unknown encoder type: '{encoder_type}'. Available encoders are: {list(ENCODER_REGISTRY.keys())}")
    
    encoder_class = ENCODER_REGISTRY[encoder_type]
    return encoder_class(**kwargs)
