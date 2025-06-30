"""
Photometric Encoders
===================

TensorDict-based encoders for astronomical photometric data.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule


class PhotometricEncoderModule(TensorDictModule):
    """
    TensorDict module for encoding photometric data.

    Processes magnitude measurements and computes colors.
    """

    def __init__(
        self,
        num_bands: int,
        hidden_dim: int = 64,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        compute_colors: bool = True,
        normalize_magnitudes: bool = True,
        in_keys: List[str] = ["magnitudes"],
        out_keys: List[str] = ["photometric_features"],
        magnitude_errors_key: Optional[str] = "magnitude_errors",
    ):
        # Build the encoder network
        output_dim = output_dim or hidden_dim
        encoder = PhotometricEncoder(
            num_bands=num_bands,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            compute_colors=compute_colors,
            normalize_magnitudes=normalize_magnitudes,
        )

        # Handle optional magnitude errors
        if magnitude_errors_key:
            in_keys = in_keys + [magnitude_errors_key]

        super().__init__(
            module=encoder,
            in_keys=in_keys,
            out_keys=out_keys,
        )

        self.magnitude_errors_key = magnitude_errors_key


class PhotometricEncoder(nn.Module):
    """Core photometric encoding network."""

    def __init__(
        self,
        num_bands: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
        compute_colors: bool = True,
        normalize_magnitudes: bool = True,
    ):
        super().__init__()

        self.num_bands = num_bands
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.compute_colors = compute_colors
        self.normalize_magnitudes = normalize_magnitudes

        # Magnitude processing
        self.magnitude_net = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Color processing
        if compute_colors and num_bands >= 2:
            num_colors = num_bands - 1
            self.color_net = nn.Sequential(
                nn.Linear(num_colors, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_dim = hidden_dim + hidden_dim // 2
        else:
            self.color_net = None
            fusion_dim = hidden_dim

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass expecting either magnitudes or (magnitudes, errors).
        """
        magnitudes = args[0]
        magnitude_errors = args[1] if len(args) > 1 else None

        # Normalize magnitudes
        if self.normalize_magnitudes:
            # Center around m=15 (typical stellar magnitude)
            normalized_mags = (magnitudes - 15.0) / 5.0
        else:
            normalized_mags = magnitudes

        # Weight by inverse variance if errors provided
        if magnitude_errors is not None:
            weights = 1.0 / (magnitude_errors**2 + 0.01)
            weights = weights / weights.mean(dim=-1, keepdim=True)
            normalized_mags = normalized_mags * weights

        # Process magnitudes
        mag_features = self.magnitude_net(normalized_mags)

        # Process colors
        if self.color_net is not None:
            colors = magnitudes[:, :-1] - magnitudes[:, 1:]
            color_features = self.color_net(colors)
            combined = torch.cat([mag_features, color_features], dim=-1)
        else:
            combined = mag_features

        # Output projection
        return self.output_net(combined)


class MultiSurveyPhotometricModule(TensorDictModule):
    """
    TensorDict module for handling photometry from multiple surveys.

    Handles different filter systems and cross-calibration.
    """

    def __init__(
        self,
        survey_configs: dict[str, dict],
        fusion_dim: int = 128,
        in_key_prefix: str = "magnitudes",
        out_key: str = "fused_photometric_features",
    ):
        # Create encoder for each survey
        encoders = {}
        in_keys = []

        for survey_name, config in survey_configs.items():
            encoder = PhotometricEncoder(
                num_bands=config["num_bands"],
                hidden_dim=config.get("hidden_dim", 64),
                output_dim=fusion_dim,
            )
            encoders[survey_name] = encoder
            in_keys.append(f"{in_key_prefix}_{survey_name}")

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(encoders), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # Build module
        module = MultiSurveyEncoder(encoders, self.fusion)

        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=[out_key],
        )


class MultiSurveyEncoder(nn.Module):
    """Core multi-survey encoding network."""

    def __init__(self, encoders: dict[str, nn.Module], fusion: nn.Module):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion

    def forward(self, *survey_magnitudes) -> torch.Tensor:
        """Process multiple survey magnitude inputs."""
        encoded_features = []

        for survey_mags, encoder in zip(survey_magnitudes, self.encoders.values()):
            encoded = encoder(survey_mags)
            encoded_features.append(encoded)

        # Concatenate and fuse
        combined = torch.cat(encoded_features, dim=-1)
        return self.fusion(combined)
