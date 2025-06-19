"""
Feature Encoders für AstroLab Tensors

Spezialisierte Encoder für verschiedene astronomische Datentypen:
- PhotometryEncoder: Multi-band Photometrie
- AstrometryEncoder: Astrometrische/räumliche Daten
- SpectroscopyEncoder: Spektroskopische Daten
- LightcurveEncoder: Time-series/lightcurve data
"""

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


class PhotometryEncoder(nn.Module):
    """Encoder für photometric tensor features."""

    def __init__(self, output_dim: int):
        super().__init__()
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
    """Encoder for lightcurve/time-series tensor features using native methods."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            Linear(48, 96),  # Rich lightcurve features
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(96, output_dim),
        )

    def forward(self, lc_tensor: LightcurveTensor) -> torch.Tensor:
        """Extract lightcurve features using native LightcurveTensor methods."""
        try:
            # Use native LightcurveTensor methods
            basic_stats = lc_tensor.compute_statistics()
            variability_stats = lc_tensor.compute_variability_stats()

            # Extract feature values
            features = []

            # Basic statistics
            for key in ["mean", "std", "min", "max", "median"]:
                if key in basic_stats:
                    features.append(basic_stats[key].flatten())

            # Variability statistics
            for key in ["rms", "mad", "skewness", "kurtosis"]:
                if key in variability_stats:
                    features.append(variability_stats[key].flatten())

            # Period detection
            try:
                periods = lc_tensor.detect_periods()
                if periods.numel() > 0:
                    features.append(periods[:1])  # Use first period
                else:
                    features.append(torch.zeros(1, device=lc_tensor._data.device))
            except Exception:
                features.append(torch.zeros(1, device=lc_tensor._data.device))

            # Amplitude (if available)
            try:
                amp = lc_tensor.get_amplitude()
                if amp is not None:
                    features.append(torch.tensor([amp], device=lc_tensor._data.device))
                else:
                    features.append(torch.zeros(1, device=lc_tensor._data.device))
            except Exception:
                features.append(torch.zeros(1, device=lc_tensor._data.device))

            # Combine all features
            if features:
                combined = torch.cat([f.float() for f in features], dim=-1)

                # Pad or truncate to expected size
                if combined.size(-1) < 48:
                    padding = torch.zeros(
                        *combined.shape[:-1],
                        48 - combined.size(-1),
                        device=combined.device,
                    )
                    combined = torch.cat([combined, padding], dim=-1)
                else:
                    combined = combined[..., :48]

                return self.encoder(combined)
            else:
                # No features extracted, use raw data
                raise Exception("No features extracted")

        except Exception:
            # Fallback: use raw lightcurve data
            raw_data = lc_tensor._data

            # Compute basic stats manually
            features = torch.cat(
                [
                    raw_data.mean(dim=-1, keepdim=True),  # Mean magnitude
                    raw_data.std(dim=-1, keepdim=True),  # Magnitude variation
                    raw_data.max(dim=-1, keepdim=True)[0],  # Peak magnitude
                    raw_data.min(dim=-1, keepdim=True)[0],  # Min magnitude
                ],
                dim=-1,
            )

            if features.size(-1) < 48:
                padding = torch.zeros(
                    *features.shape[:-1], 48 - features.size(-1), device=features.device
                )
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[..., :48]

            return self.encoder(features)


__all__ = [
    "PhotometryEncoder",
    "AstrometryEncoder",
    "SpectroscopyEncoder",
    "LightcurveEncoder",
]
