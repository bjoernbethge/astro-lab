"""
Spatial Encoders
===============

TensorDict-based encoders for spatial and positional data.
"""

# Set tensordict behavior globally for this module
import os

os.environ["LIST_TO_STACK"] = "1"

from typing import List, Optional

import tensordict
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

tensordict.set_list_to_stack(True)


class SpatialEncoderModule(TensorDictModule):
    """
    TensorDict module for encoding spatial coordinates.

    Features:
    - Multi-scale processing
    - Coordinate system awareness
    - Distance encoding
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        num_scales: int = 3,
        max_distance: float = 1000.0,
        coordinate_system: str = "galactocentric",
        in_keys: List[str] = ["coordinates"],
        out_keys: List[str] = ["spatial_features"],
    ):
        output_dim = output_dim or hidden_dim

        encoder = SpatialEncoder(
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_scales=num_scales,
            max_distance=max_distance,
            coordinate_system=coordinate_system,
        )

        super().__init__(
            module=encoder,
            in_keys=in_keys,
            out_keys=out_keys,
        )


class SpatialEncoder(nn.Module):
    """Core spatial encoding network."""

    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_scales: int = 3,
        max_distance: float = 1000.0,
        coordinate_system: str = "galactocentric",
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_scales = num_scales
        self.max_distance = max_distance
        self.coordinate_system = coordinate_system

        # Positional encoding
        self.pos_encoder = PositionalEncoding3D(
            hidden_dim // 2,
            max_distance=max_distance,
        )

        # Multi-scale processing
        self.scale_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(spatial_dim + hidden_dim // 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_scales)
            ]
        )

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode spatial coordinates."""
        coordinates.size(0)

        # Add positional encoding
        pos_encoding = self.pos_encoder(coordinates)
        coord_with_pos = torch.cat([coordinates, pos_encoding], dim=-1)

        # Multi-scale processing
        scale_features = []
        for scale_encoder in self.scale_encoders:
            scale_feat = scale_encoder(coord_with_pos)
            scale_features.append(scale_feat)

        # Fuse scales
        multi_scale = torch.cat(scale_features, dim=-1)
        fused = self.scale_fusion(multi_scale)

        # Output
        return self.output_net(fused)


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for astronomical coordinates."""

    def __init__(self, encoding_dim: int, max_distance: float = 1000.0):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.max_distance = max_distance

        # Ensure encoding_dim is divisible by 6 (3 dims × 2 for sin/cos)
        self.num_frequencies = encoding_dim // 6
        self.remaining_dims = encoding_dim - (self.num_frequencies * 6)

        # Learnable frequency parameters
        self.frequencies = nn.Parameter(torch.randn(self.num_frequencies) * 0.1)

        # Extra projection for remaining dimensions
        if self.remaining_dims > 0:
            self.extra_proj = nn.Linear(3, self.remaining_dims)
        else:
            self.extra_proj = None

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Apply sinusoidal positional encoding."""
        # Normalize by max distance
        norm_coords = coordinates / self.max_distance

        encoding = []

        # Encode each dimension with each frequency
        for dim in range(3):
            for i in range(self.num_frequencies):
                freq = self.frequencies[i]
                encoding.append(torch.sin(norm_coords[:, dim : dim + 1] * freq))
                encoding.append(torch.cos(norm_coords[:, dim : dim + 1] * freq))

        # Concatenate all encodings
        encoded = torch.cat(encoding, dim=-1)

        # Add extra dimensions if needed
        if self.extra_proj is not None:
            extra = self.extra_proj(coordinates)
            encoded = torch.cat([encoded, extra], dim=-1)

        return encoded


class CosmicWebEncoderModule(TensorDictModule):
    """
    TensorDict module for cosmic web structure encoding.

    Specialized for large-scale structure analysis.
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 256,
        num_scales: int = 4,
        cosmic_web_classes: int = 4,  # void, wall, filament, node
        scales: List[float] = [5.0, 10.0, 50.0, 100.0],  # Mpc
        in_keys: List[str] = ["coordinates"],
        out_keys: List[str] = ["cosmic_web_features", "structure_logits"],
    ):
        encoder = CosmicWebEncoder(
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_scales=num_scales,
            cosmic_web_classes=cosmic_web_classes,
            scales=scales,
        )

        super().__init__(
            module=encoder,
            in_keys=in_keys,
            out_keys=out_keys,
        )


class CosmicWebEncoder(nn.Module):
    """Core cosmic web encoding network."""

    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 256,
        num_scales: int = 4,
        cosmic_web_classes: int = 4,
        scales: List[float] = [5.0, 10.0, 50.0, 100.0],
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.cosmic_web_classes = cosmic_web_classes
        self.scales = scales

        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(spatial_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_scales)
            ]
        )

        # Structure detection heads
        self.structure_detector = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cosmic_web_classes),
        )

        # Feature output
        self.feature_output = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode cosmic web structures.

        Returns:
            - cosmic_web_features: [N, hidden_dim]
            - structure_logits: [N, cosmic_web_classes]
        """
        # Multi-scale encoding
        scale_features = []

        for i, (scale_encoder, scale) in enumerate(
            zip(self.scale_encoders, self.scales)
        ):
            # Scale-normalize coordinates
            scaled_coords = coordinates / scale
            scale_feat = scale_encoder(scaled_coords)
            scale_features.append(scale_feat)

        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=-1)

        # Detect structures
        structure_logits = self.structure_detector(multi_scale)

        # Output features
        features = self.feature_output(multi_scale)

        return features, structure_logits
