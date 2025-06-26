"""
TensorDict-Native Encoder Modules for AstroLab Models
===================================================

Modern encoder classes designed to process astronomical data using the
AstroLab TensorDict system. Each encoder is specifically designed for
different astronomical data modalities and uses the native methods
of our TensorDict classes.

Supported TensorDict types:
- SurveyTensorDict: Multi-component survey data
- PhotometricTensorDict: Multi-band photometry
- SpatialTensorDict: 3D spatial coordinates
- LightcurveTensorDict: Time series photometry
- SpectralTensorDict: Spectroscopic data
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict

# Import our TensorDict classes to use their methods
from astro_lab.tensors.tensordict_astro import (
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
    SpectralTensorDict,
    SurveyTensorDict,
)


class SurveyEncoder(nn.Module):
    """
    Encoder for SurveyTensorDict data.

    Handles combined spatial + photometric + optional spectral data
    from astronomical surveys using native TensorDict methods.
    """

    def __init__(
        self,
        output_dim: int,
        use_photometry: bool = True,
        use_astrometry: bool = True,
        use_spectroscopy: bool = False,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim
        self.use_photometry = use_photometry
        self.use_astrometry = use_astrometry
        self.use_spectroscopy = use_spectroscopy
        self.dropout = dropout

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Component encoders
        if use_photometry:
            self.photometry_encoder = PhotometryEncoder(
                output_dim=self.hidden_dim // 2, device=device, **kwargs
            )

        if use_astrometry:
            # Enable 3D coordinates for point cloud processing
            self.astrometry_encoder = AstrometryEncoder(
                output_dim=self.hidden_dim // 2,
                use_3d=True,  # Enable full 3D coordinate support for point clouds
                device=device,
                **kwargs,
            )

        if use_spectroscopy:
            self.spectroscopy_encoder = SpectroscopyEncoder(
                output_dim=self.hidden_dim // 4, device=device, **kwargs
            )

        # Calculate combined dimension
        combined_dim = 0
        if use_photometry:
            combined_dim += self.hidden_dim // 2
        if use_astrometry:
            combined_dim += self.hidden_dim // 2
        if use_spectroscopy:
            combined_dim += self.hidden_dim // 4

        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

        self.to(self.device)

    def forward(self, data: SurveyTensorDict) -> torch.Tensor:
        """Process SurveyTensorDict through component encoders using native methods."""
        if not isinstance(data, SurveyTensorDict):
            raise ValueError("SurveyEncoder requires SurveyTensorDict input")

        encoded_components = []

        # Process photometry using PhotometricTensorDict methods
        if self.use_photometry and "photometric" in data:
            photometric_data = data["photometric"]
            if isinstance(photometric_data, PhotometricTensorDict):
                phot_encoded = self.photometry_encoder(photometric_data)
                encoded_components.append(phot_encoded)

        # Process astrometry using SpatialTensorDict methods
        if self.use_astrometry and "spatial" in data:
            spatial_data = data["spatial"]
            if isinstance(spatial_data, SpatialTensorDict):
                astro_encoded = self.astrometry_encoder(spatial_data)
                encoded_components.append(astro_encoded)

        # Process spectroscopy using SpectralTensorDict methods
        if self.use_spectroscopy and "spectral" in data:
            spectral_data = data["spectral"]
            if isinstance(spectral_data, SpectralTensorDict):
                spec_encoded = self.spectroscopy_encoder(spectral_data)
                encoded_components.append(spec_encoded)

        if not encoded_components:
            raise ValueError("No compatible data components found in SurveyTensorDict")

        # Combine all components
        combined = torch.cat(encoded_components, dim=-1)

        # Final projection
        output = self.final_projection(combined)
        return output


class PhotometryEncoder(nn.Module):
    """Encoder for PhotometricTensorDict using native methods."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim * 2
        self.dropout = dropout

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # We'll determine input dim dynamically based on data
        self.encoder = None
        self.to(self.device)

    def _build_encoder(self, input_dim: int):
        """Build encoder based on actual input dimension."""
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)

    def forward(self, data: PhotometricTensorDict) -> torch.Tensor:
        """Process PhotometricTensorDict using native methods."""
        if not isinstance(data, PhotometricTensorDict):
            raise ValueError("PhotometryEncoder requires PhotometricTensorDict input")

        # Use native PhotometricTensorDict methods
        magnitudes = data["magnitudes"].to(self.device)

        # Prepare features using native methods
        input_features = [magnitudes]

        # Add errors if available
        if "errors" in data:
            errors = data["errors"].to(self.device)
            input_features.append(errors)

        # Compute colors using native method if multiple bands
        if data.n_bands > 1:
            # Use compute_colors method for standard color pairs
            bands = data.bands
            if len(bands) >= 2:
                # Create color pairs from adjacent bands
                color_pairs = [(bands[i], bands[i + 1]) for i in range(len(bands) - 1)]
                try:
                    colors_dict = data.compute_colors(color_pairs)
                    # Extract color values and stack them
                    color_values = []
                    for color_name in colors_dict.keys():
                        if isinstance(colors_dict[color_name], torch.Tensor):
                            color_values.append(colors_dict[color_name])
                    if color_values:
                        colors = torch.stack(color_values, dim=-1)
                        input_features.append(colors)
                except Exception:
                    # Fallback to manual color computation
                    colors = magnitudes[..., :-1] - magnitudes[..., 1:]
                    input_features.append(colors)

        combined = torch.cat(input_features, dim=-1)

        # Build encoder if not exists
        if self.encoder is None:
            self._build_encoder(combined.shape[-1])

        # Handle NaN values
        if torch.isnan(combined).any():
            combined = torch.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        if self.encoder is not None:
            return self.encoder(combined)
        else:
            raise RuntimeError("Failed to initialize encoder")


class AstrometryEncoder(nn.Module):
    """Encoder for SpatialTensorDict using native coordinate methods with full 3D support."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_3d: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim * 2
        self.use_3d = use_3d

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Support both 2D and 3D coordinates
        input_dim = 3 if use_3d else 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, output_dim),
        )

        self.to(self.device)

    def forward(self, data: SpatialTensorDict) -> torch.Tensor:
        """Process SpatialTensorDict using native coordinate access with 3D support."""
        if not isinstance(data, SpatialTensorDict):
            raise ValueError("AstrometryEncoder requires SpatialTensorDict input")

        # Use native coordinate access methods
        x_coord = data.x.to(self.device)  # RA or x
        y_coord = data.y.to(self.device)  # Dec or y

        if self.use_3d:
            z_coord = data.z.to(self.device)  # Distance or z
            # Stack full 3D coordinates for point cloud processing
            coords = torch.stack([x_coord, y_coord, z_coord], dim=-1)
        else:
            # Stack 2D coordinates for traditional sky coordinates
            coords = torch.stack([x_coord, y_coord], dim=-1)

        return self.encoder(coords)


class SpectroscopyEncoder(nn.Module):
    """Encoder for SpectralTensorDict using native spectral methods."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim * 2

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # We'll build encoder dynamically based on spectrum dimension
        self.encoder = None
        self.to(self.device)

    def _build_encoder(self, input_dim: int):
        """Build encoder based on spectrum dimension."""
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)

    def forward(self, data: SpectralTensorDict) -> torch.Tensor:
        """Process SpectralTensorDict using native methods."""
        if not isinstance(data, SpectralTensorDict):
            raise ValueError("SpectroscopyEncoder requires SpectralTensorDict input")

        flux = data["flux"].to(self.device)

        # Use rest_wavelengths property if redshift correction is needed
        if hasattr(data, "redshift") and data.redshift > 0:
            # For now, just use flux directly
            # Could implement redshift correction using rest_wavelengths
            pass

        # Handle different flux dimensions
        if flux.dim() > 2:
            # Average over wavelength dimension
            flux_features = flux.mean(dim=-1)
        else:
            flux_features = flux

        # Build encoder if needed
        if self.encoder is None:
            self._build_encoder(flux_features.shape[-1])

        if self.encoder is not None:
            return self.encoder(flux_features)
        else:
            raise RuntimeError("Failed to initialize encoder")


class LightcurveEncoder(nn.Module):
    """Encoder for LightcurveTensorDict using LSTM with native time methods."""

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Input dimension: time + magnitude(s)
        input_dim = 2  # Default: time + single band

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.to(self.device)

    def forward(self, data: LightcurveTensorDict) -> torch.Tensor:
        """Process LightcurveTensorDict using native time methods."""
        if not isinstance(data, LightcurveTensorDict):
            raise ValueError("LightcurveEncoder requires LightcurveTensorDict input")

        # Use native TensorDict data access
        times = data["times"].to(self.device)
        magnitudes = data["magnitudes"].to(self.device)

        # Use time_span property for additional features (optional)
        # time_spans = data.time_span.to(self.device)

        # Prepare sequence data: [batch, seq_len, features]
        if times.dim() == 1:
            # Single lightcurve
            times = times.unsqueeze(0).unsqueeze(-1)
            magnitudes = magnitudes.unsqueeze(0).unsqueeze(-1)
        elif times.dim() == 2:
            # Multiple lightcurves
            times = times.unsqueeze(-1)
            magnitudes = magnitudes.unsqueeze(-1)

        # Handle multiple bands in magnitudes
        if magnitudes.dim() == 3 and magnitudes.shape[-1] > 1:
            # Use first band only for simplicity
            magnitudes = magnitudes[..., 0:1]

        # Combine time and magnitude
        sequence = torch.cat([times, magnitudes], dim=-1)

        # Process through LSTM
        lstm_out, (h_n, _) = self.lstm(sequence)

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last hidden state + attention pooling
        final_hidden = h_n[-1]  # [batch, hidden_dim]
        attn_pooled = attn_out.mean(dim=1)  # [batch, hidden_dim]

        # Combine representations
        combined = final_hidden + attn_pooled

        # Project to output dimension
        return self.output_proj(combined)


# TensorDict-specific encoder registry
TENSORDICT_ENCODER_REGISTRY = {
    "survey": SurveyEncoder,
    "photometry": PhotometryEncoder,
    "astrometry": AstrometryEncoder,
    "spectroscopy": SpectroscopyEncoder,
    "lightcurve": LightcurveEncoder,
}


def create_tensordict_encoder(encoder_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create TensorDict-native encoders.

    Args:
        encoder_type: Type of encoder ('survey', 'photometry', etc.)
        **kwargs: Arguments for encoder constructor

    Returns:
        TensorDict-native encoder instance

    Raises:
        ValueError: If encoder_type is not supported
    """
    if encoder_type not in TENSORDICT_ENCODER_REGISTRY:
        available = list(TENSORDICT_ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown encoder type: '{encoder_type}'. Available: {available}"
        )

    encoder_class = TENSORDICT_ENCODER_REGISTRY[encoder_type]
    return encoder_class(**kwargs)
