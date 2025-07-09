"""
PointCloudAutoencoder
====================

Autoencoder module for astronomical point clouds (e.g., galaxies, stars, simulations).
- Encoder/decoder architecture
- TensorDict compatible
- GPU accelerated
- Extensible for various point cloud features
"""

import os

os.environ["LIST_TO_STACK"] = "1"

import tensordict
import torch.nn as nn
from tensordict import TensorDict
from torch_geometric.data import Data

tensordict.set_list_to_stack(True)

from .base import BaseAutoencoder


class PointCloudAutoencoder(BaseAutoencoder):
    """
    Autoencoder for astronomical point clouds.

    Extensible for additional features (e.g., colors, magnitudes, uncertainties).
    Supports both regular tensors and PyTorch Geometric Data objects.

    Args:
        input_dim: Input feature dimension (default: 3 for coordinates)
        latent_dim: Latent space dimension (default: 16)
        hidden_dim: Hidden layer dimension (default: 64)
        use_geometric: Whether to use PyG layers (default: True)
    """

    def __init__(self, input_dim=3, latent_dim=16, hidden_dim=64, use_geometric=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_geometric = use_geometric

        if use_geometric:
            # PyTorch Geometric encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

            # PyTorch Geometric decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
        else:
            # Standard MLP encoder/decoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def forward(self, x):
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input data (torch.Tensor, TensorDict, or PyG Data)
        Returns:
            Reconstructed point cloud (same shape as input)
        """
        if isinstance(x, TensorDict):
            x = x["coordinates"]
        elif isinstance(x, Data):
            # Handle PyG Data object
            x = x.x

        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def encode(self, x):
        """
        Encode point cloud to latent representation.

        Args:
            x: Input data (torch.Tensor, TensorDict, or PyG Data)
        Returns:
            Latent representation [N, latent_dim]
        """
        if isinstance(x, TensorDict):
            x = x["coordinates"]
        elif isinstance(x, Data):
            x = x.x

        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent representation back to point cloud.

        Args:
            z: Latent representation [N, latent_dim]
        Returns:
            Reconstructed point cloud [N, input_dim]
        """
        return self.decoder(z)

    def get_latent_representation(self, x):
        """
        Get latent representation for analysis or visualization.

        Args:
            x: Input data (torch.Tensor, TensorDict, or PyG Data)
        Returns:
            Latent representation
        """
        return self.encode(x)
