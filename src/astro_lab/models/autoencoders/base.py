"""
BaseAutoencoder
===============

Abstract base class for autoencoder models with TensorDict support.
"""

import torch.nn as nn


class BaseAutoencoder(nn.Module):
    """
    Abstract base class for autoencoder models.

    Provides interface for encode/decode operations with TensorDict support.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input data (torch.Tensor or TensorDict)
        Returns:
            Reconstructed data
        """
        raise NotImplementedError

    def encode(self, x):
        """
        Encode input data to latent representation.

        Args:
            x: Input data (torch.Tensor or TensorDict)
        Returns:
            Latent representation
        """
        raise NotImplementedError

    def decode(self, z):
        """
        Decode latent representation back to data space.

        Args:
            z: Latent representation
        Returns:
            Reconstructed data
        """
        raise NotImplementedError
