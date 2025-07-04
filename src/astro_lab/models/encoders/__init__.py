"""
Encoders for AstroLab Models
===========================

Specialized encoders for different astronomical data types.
"""

from .multimodal import MultiModalFusion
from .photometric import PhotometricEncoder, PhotometricEncoderModule
from .spatial import (
    CosmicWebEncoder,
    CosmicWebEncoderModule,
    SpatialEncoder,
    SpatialEncoderModule,
)
from .spectral import SpectralEncoder, SpectralEncoderModule
from .temporal import (
    MultiTimescaleEncoderModule,
    TemporalEncoder,
    TemporalEncoderModule,
)

__all__ = [
    # Standard encoders (nn.Module)
    "PhotometricEncoder",
    "SpatialEncoder",
    "SpectralEncoder",
    "TemporalEncoder",
    "CosmicWebEncoder",
    # TensorDict modules
    "PhotometricEncoderModule",
    "SpatialEncoderModule",
    "SpectralEncoderModule",
    "TemporalEncoderModule",
    "CosmicWebEncoderModule",
    "MultiTimescaleEncoderModule",
    # Fusion
    "MultiModalFusion",
]
