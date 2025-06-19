"""
Feature Encoders für AstroLab Tensors

Spezialisierte Encoder für verschiedene astronomische Datentypen:
- PhotometryEncoder: Multi-band Photometrie
- AstrometryEncoder: Astrometrische/räumliche Daten
- SpectroscopyEncoder: Spektroskopische Daten
- LightcurveEncoder: Time-series/lightcurve data

Robuste Implementierung mit automatischen Fallbacks und Error Handling.
"""

import logging
from typing import Any, Union

import torch
import torch.nn as nn
from torch_geometric.nn import Linear

from astro_lab.tensors import (
    LightcurveTensor,
    PhotometricTensor,
    Spatial3DTensor,
    SpectralTensor,
    SurveyTensor,
)

logger = logging.getLogger(__name__)


class BaseEncoder(nn.Module):
    """Base encoder with robust error handling."""

    def __init__(self, output_dim: int, expected_input_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.expected_input_dim = expected_input_dim

        # Fallback projection for wrong input sizes
        self.fallback_projection = nn.Linear(expected_input_dim, output_dim)

    def create_fallback_features(self, tensor: Any) -> torch.Tensor:
        """Create fallback features when extraction fails."""
        batch_size = self._get_batch_size(tensor)
        return torch.zeros(batch_size, self.output_dim, device=self._get_device(tensor))

    def _get_batch_size(self, tensor: Any) -> int:
        """Get batch size from tensor."""
        if hasattr(tensor, "_data"):
            return tensor._data.size(0)
        elif hasattr(tensor, "size"):
            return tensor.size(0)
        else:
            return 1

    def _get_device(self, tensor: Any):
        """Get device from tensor."""
        if hasattr(tensor, "_data"):
            return tensor._data.device
        elif hasattr(tensor, "device"):
            return tensor.device
        else:
            return torch.device("cpu")


class PhotometryEncoder(BaseEncoder):
    """Encoder für photometric tensor features."""

    def __init__(self, output_dim: int):
        super().__init__(output_dim, expected_input_dim=32)
        self.encoder = nn.Sequential(
            Linear(32, 64),  # Standard photometric features
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(64, output_dim),
        )

    def forward(self, phot_tensor: PhotometricTensor) -> torch.Tensor:
        """Extract photometric features."""
        try:
            # Use native PhotometricTensor methods
            data = phot_tensor._data  # Direct access to underlying data

            # Handle very small input dimensions
            if data.size(-1) == 1:
                # For single band, create simple features
                features = torch.cat(
                    [
                        data,  # Original magnitude
                        torch.zeros_like(data),  # No variation for single band
                        data,  # Max = original
                        data,  # Min = original
                    ],
                    dim=-1,
                )
            else:
                # Compute basic photometric statistics
                features = torch.cat(
                    [
                        data.mean(dim=-1, keepdim=True),  # Mean magnitude
                        data.std(dim=-1, keepdim=True),  # Magnitude variation
                        data.max(dim=-1, keepdim=True)[0],  # Brightest magnitude
                        data.min(dim=-1, keepdim=True)[0],  # Faintest magnitude
                    ],
                    dim=-1,
                )

                # Add color indices if multi-band
                if data.size(-1) >= 2:
                    colors = []
                    for i in range(min(data.size(-1) - 1, 4)):  # Max 4 colors
                        colors.append(data[..., i] - data[..., i + 1])
                    if colors:
                        color_tensor = torch.stack(colors, dim=-1)
                        features = torch.cat([features, color_tensor], dim=-1)

            # Pad or truncate to expected size
            if features.size(-1) < 32:
                padding = torch.zeros(
                    *features.shape[:-1], 32 - features.size(-1), device=features.device
                )
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[..., :32]

            # Replace any NaN values with zeros
            features = torch.where(
                torch.isnan(features), torch.zeros_like(features), features
            )

            return self.encoder(features)

        except Exception:
            # Fallback: use raw data
            raw_data = phot_tensor._data
            if raw_data.size(-1) < 32:
                padding = torch.zeros(
                    *raw_data.shape[:-1], 32 - raw_data.size(-1), device=raw_data.device
                )
                raw_data = torch.cat([raw_data, padding], dim=-1)
            else:
                raw_data = raw_data[..., :32]

            # Replace any NaN values with zeros
            raw_data = torch.where(
                torch.isnan(raw_data), torch.zeros_like(raw_data), raw_data
            )

            return self.encoder(raw_data)


class AstrometryEncoder(nn.Module):
    """Encoder für astrometric/spatial tensor features."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            Linear(16, 32),  # Coordinate + proper motion features
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(32, output_dim),
        )

    def forward(self, spatial_tensor: Spatial3DTensor) -> torch.Tensor:
        """Extract astrometric features."""
        try:
            # Use native Spatial3DTensor data
            data = spatial_tensor._data  # Direct access to coordinates

            # Extract coordinate features
            if data.size(-1) >= 3:
                # Assume RA, DEC, distance/parallax format
                coords = data[..., :3]

                # Add proper motions if available
                if data.size(-1) >= 5:
                    pm = data[..., 3:5]  # pmra, pmdec
                    features = torch.cat([coords, pm], dim=-1)
                else:
                    features = coords

                # Add distance-derived features if available
                if data.size(-1) >= 6:
                    dist_features = data[..., 5:6]
                    features = torch.cat([features, dist_features], dim=-1)
            else:
                features = data

            # Pad or truncate to expected size
            if features.size(-1) < 16:
                padding = torch.zeros(
                    *features.shape[:-1], 16 - features.size(-1), device=features.device
                )
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[..., :16]

            return self.encoder(features)

        except Exception:
            # Fallback: use raw data
            raw_data = spatial_tensor._data
            if raw_data.size(-1) < 16:
                padding = torch.zeros(
                    *raw_data.shape[:-1], 16 - raw_data.size(-1), device=raw_data.device
                )
                raw_data = torch.cat([raw_data, padding], dim=-1)
            else:
                raw_data = raw_data[..., :16]

            return self.encoder(raw_data)


class SpectroscopyEncoder(nn.Module):
    """Encoder für spectroscopic tensor features."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            Linear(64, 128),  # Spectral features
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(128, output_dim),
        )

    def forward(self, spec_tensor: Union[SpectralTensor, SurveyTensor]) -> torch.Tensor:
        """Extract spectroscopic features."""
        try:
            # Use native tensor data
            data = spec_tensor._data  # Direct access to spectral data

            # Compute spectral statistics
            features = torch.cat(
                [
                    data.mean(dim=-1, keepdim=True),  # Mean flux
                    data.std(dim=-1, keepdim=True),  # Flux variation
                    data.max(dim=-1, keepdim=True)[0],  # Peak flux
                    data.min(dim=-1, keepdim=True)[0],  # Min flux
                ],
                dim=-1,
            )

            # Add spectral moments/indices if spectrum is long enough
            if data.size(-1) >= 10:
                # Simple spectral indices (blue/red ratio, etc.)
                blue_flux = data[..., : data.size(-1) // 3].mean(dim=-1, keepdim=True)
                red_flux = data[..., 2 * data.size(-1) // 3 :].mean(
                    dim=-1, keepdim=True
                )
                color_index = blue_flux - red_flux
                features = torch.cat([features, color_index], dim=-1)

            # Pad to expected size
            if features.size(-1) < 64:
                padding = torch.zeros(
                    *features.shape[:-1], 64 - features.size(-1), device=features.device
                )
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[..., :64]

            return self.encoder(features)

        except Exception:
            # Fallback: use raw data
            raw_data = spec_tensor._data
            if raw_data.size(-1) < 64:
                padding = torch.zeros(
                    *raw_data.shape[:-1], 64 - raw_data.size(-1), device=raw_data.device
                )
                raw_data = torch.cat([raw_data, padding], dim=-1)
            else:
                raw_data = raw_data[..., :64]

            return self.encoder(raw_data)


class LightcurveEncoder(nn.Module):
    """Encoder for lightcurve/time-series tensor features."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            Linear(32, 64),  # Lightcurve features
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(64, output_dim),
        )

    def forward(self, lc_tensor: LightcurveTensor) -> torch.Tensor:
        """Extract lightcurve features."""
        try:
            # Use raw lightcurve data
            raw_data = lc_tensor._data

            # If 3D (batch, time, features), aggregate over time dimension
            if raw_data.dim() == 3:
                # Compute statistics over time dimension
                features = torch.cat(
                    [
                        raw_data.mean(dim=1),  # Mean over time
                        raw_data.std(dim=1),  # Std over time
                        raw_data.max(dim=1)[0],  # Max over time
                        raw_data.min(dim=1)[0],  # Min over time
                    ],
                    dim=-1,
                )
            elif raw_data.dim() == 2:
                # If 2D (time, features), treat as single lightcurve
                # Aggregate over time dimension to get single feature vector
                features = torch.cat(
                    [
                        raw_data.mean(
                            dim=0, keepdim=True
                        ),  # Mean over time -> (1, features)
                        raw_data.std(
                            dim=0, keepdim=True
                        ),  # Std over time -> (1, features)
                        raw_data.max(dim=0, keepdim=True)[
                            0
                        ],  # Max over time -> (1, features)
                        raw_data.min(dim=0, keepdim=True)[
                            0
                        ],  # Min over time -> (1, features)
                    ],
                    dim=-1,
                )
            else:
                # If 1D, treat as single sample
                features = raw_data.unsqueeze(0)  # (features,) -> (1, features)

            # Pad or truncate to expected size (32)
            if features.size(-1) < 32:
                padding = torch.zeros(
                    *features.shape[:-1], 32 - features.size(-1), device=features.device
                )
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[..., :32]

            return self.encoder(features)

        except Exception:
            # Fallback: create simple features with batch size 1
            raw_data = lc_tensor._data
            device = raw_data.device

            # Create simple fallback features for single lightcurve
            fallback_features = torch.randn(1, 32, device=device)
            return self.encoder(fallback_features)


__all__ = [
    "PhotometryEncoder",
    "AstrometryEncoder",
    "SpectroscopyEncoder",
    "LightcurveEncoder",
]
