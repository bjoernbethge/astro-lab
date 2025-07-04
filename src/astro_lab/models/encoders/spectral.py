"""
Spectral Encoders
================

TensorDict-based encoders for spectroscopic data.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule


class SpectralEncoderModule(TensorDictModule):
    """
    TensorDict module for encoding spectroscopic data.

    Features:
    - Wavelength-aware processing
    - Line detection
    - Redshift handling
    """

    def __init__(
        self,
        wavelength_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        num_lines: int = 10,
        use_attention: bool = True,
        in_keys: List[str] = ["wavelengths", "flux"],
        out_keys: List[str] = ["spectral_features"],
        flux_errors_key: Optional[str] = "flux_errors",
    ):
        output_dim = output_dim or hidden_dim

        encoder = SpectralEncoder(
            wavelength_dim=wavelength_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_lines=num_lines,
            use_attention=use_attention,
        )

        # Add flux errors to input keys if provided
        if flux_errors_key:
            in_keys = in_keys + [flux_errors_key]

        super().__init__(
            module=encoder,
            in_keys=in_keys,
            out_keys=out_keys,
        )

        self.flux_errors_key = flux_errors_key


class SpectralEncoder(nn.Module):
    """Core spectral encoding network."""

    def __init__(
        self,
        wavelength_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_lines: int = 10,
        use_attention: bool = True,
    ):
        super().__init__()

        self.wavelength_dim = wavelength_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_lines = num_lines

        # 1D CNN for spectral features
        self.spectral_cnn = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Global features
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )
        else:
            self.attention = None

        # Line detection head
        if num_lines > 0:
            self.line_detector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_lines),
            )
        else:
            self.line_detector = None

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 if use_attention else hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass expecting (wavelengths, flux) or (wavelengths, flux, flux_errors).
        """
        args[0]
        flux = args[1]
        flux_errors = args[2] if len(args) > 2 else None

        flux.size(0)

        # Weight by inverse variance if errors provided
        if flux_errors is not None:
            weights = 1.0 / (flux_errors**2 + 1e-6)
            weights = weights / weights.mean(dim=-1, keepdim=True)
            weighted_flux = flux * weights
        else:
            weighted_flux = flux

        # CNN processing
        flux_1d = weighted_flux.unsqueeze(1)  # [B, 1, wavelength_dim]
        cnn_features = self.spectral_cnn(flux_1d)  # [B, hidden_dim, wavelength_dim]

        # Global pooling
        global_features = self.global_pool(cnn_features).squeeze(-1)  # [B, hidden_dim]

        # Attention processing
        if self.attention is not None:
            # Transpose for attention [B, wavelength_dim, hidden_dim]
            cnn_features_t = cnn_features.transpose(1, 2)
            attended, _ = self.attention(cnn_features_t, cnn_features_t, cnn_features_t)
            # Pool attended features
            attended_pool = attended.mean(dim=1)  # [B, hidden_dim]

            # Combine global and attended features
            combined = torch.cat([global_features, attended_pool], dim=-1)
        else:
            combined = global_features

        # Output projection
        return self.output_net(combined)


class MultiResolutionSpectralModule(TensorDictModule):
    """
    TensorDict module for multi-resolution spectral data.

    Handles spectra at different resolutions or from different instruments.
    """

    def __init__(
        self,
        resolutions: List[int],
        hidden_dim: int = 128,
        fusion_dim: int = 256,
        in_key_prefix: str = "spectrum",
        out_key: str = "fused_spectral_features",
    ):
        # Create encoder for each resolution
        encoders = []
        in_keys = []

        for i, resolution in enumerate(resolutions):
            encoder = SpectralEncoder(
                wavelength_dim=resolution,
                hidden_dim=hidden_dim,
                output_dim=fusion_dim,
                use_attention=resolution > 1000,  # Use attention for high-res
            )
            encoders.append(encoder)
            in_keys.extend(
                [f"{in_key_prefix}_{i}_wavelengths", f"{in_key_prefix}_{i}_flux"]
            )

        # Build multi-resolution module
        module = MultiResolutionEncoder(encoders, fusion_dim)

        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=[out_key],
        )


class MultiResolutionEncoder(nn.Module):
    """Core multi-resolution spectral encoder."""

    def __init__(self, encoders: List[nn.Module], fusion_dim: int):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(encoders), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, *resolution_data) -> torch.Tensor:
        """Process multi-resolution spectral data."""
        # Each resolution provides (wavelengths, flux) pair
        encoded_features = []

        for i in range(0, len(resolution_data), 2):
            wavelengths = resolution_data[i]
            flux = resolution_data[i + 1]

            encoder_idx = i // 2
            encoded = self.encoders[encoder_idx](wavelengths, flux)
            encoded_features.append(encoded)

        # Fuse all resolutions
        combined = torch.cat(encoded_features, dim=-1)
        return self.fusion(combined)
