"""
Autoencoder Models
=================

Autoencoder implementations for astronomical data analysis.
Supports TensorDict and PyTorch Geometric integration.
"""

from .base import BaseAutoencoder
from .pointcloud_autoencoder import PointCloudAutoencoder

__all__ = [
    "BaseAutoencoder",
    "PointCloudAutoencoder",
]
