"""AstroLab Models Module - Graph neural networks for astronomical data."""

from .astro_model import (
    AstroModel,
    create_cosmic_web_model,
    create_exoplanet_model,
    create_galaxy_model,
    create_stellar_model,
)
from .autoencoders.base import BaseAutoencoder
from .autoencoders.pointcloud_autoencoder import PointCloudAutoencoder
from .base_model import AstroBaseModel
from .mixins import ExplainabilityMixin

__all__ = [
    "AstroModel",
    "AstroBaseModel",
    "BaseAutoencoder",
    "PointCloudAutoencoder",
    "create_cosmic_web_model",
    "create_stellar_model",
    "create_galaxy_model",
    "create_exoplanet_model",
    "ExplainabilityMixin",
]

# For details on layers, encoders, components etc., import from the respective submodules:
#   from astro_lab.models.layers import FlexibleGraphConv
#   from astro_lab.models.encoders import PhotometricEncoder
#   from astro_lab.models.components import EnhancedMLPBlock
