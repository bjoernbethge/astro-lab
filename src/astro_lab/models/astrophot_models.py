"""
AstroPhot Integration Models for Galaxy Modeling

Modernized for native SurveyTensor integration and PyTorch 2.x APIs.
Uses existing encoder infrastructure from encoders.py module.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, global_mean_pool

from astro_lab.tensors import SurveyTensor
from astro_lab.models.encoders import AstrometryEncoder, PhotometryEncoder
from astro_lab.models.utils import initialize_weights
from astro_lab.models.layers import LayerFactory

# AstroPhot integration
try:
    import astrophot

    ASTROPHOT_AVAILABLE = True
except ImportError:
    ASTROPHOT_AVAILABLE = False

class AstroPhotGNN(nn.Module):
    """Graph Neural Network with AstroPhot integration for galaxy modeling."""

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 12,  # Typical Sersic + disk parameters
        num_layers: int = 3,
        model_components: List[str] = ["sersic", "disk"],
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model_components = model_components
        self.dropout = dropout

        # Use existing encoders
        self.photometry_encoder = PhotometryEncoder(hidden_dim // 2)
        self.astrometry_encoder = AstrometryEncoder(hidden_dim // 2)

        # Feature fusion
        self.feature_fusion = Linear(hidden_dim, hidden_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, normalize=True) for _ in range(num_layers)]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Component-specific heads
        self.component_heads = nn.ModuleDict()
        for component in model_components:
            if component == "sersic":
                self.component_heads[component] = SersicParameterHead(hidden_dim)
            elif component == "disk":
                self.component_heads[component] = DiskParameterHead(hidden_dim)
            elif component == "bulge":
                self.component_heads[component] = BulgeParameterHead(hidden_dim)

        # Global galaxy parameters
        self.global_head = GlobalGalaxyHead(hidden_dim, output_dim)

        self.apply(initialize_weights)

    def extract_galaxy_features(self, survey_tensor: SurveyTensor) -> torch.Tensor:
        """Extract galaxy features from SurveyTensor using existing encoders."""
        features = []

        # Use photometric encoder
        try:
            phot_tensor = survey_tensor.get_photometric_tensor()
            phot_features = self.photometry_encoder(phot_tensor)
            features.append(phot_features)
        except Exception:
            batch_size = len(survey_tensor)
            phot_features = torch.zeros(
                batch_size, self.hidden_dim // 2, device=survey_tensor._data.device
            )
            features.append(phot_features)

        # Use astrometry encoder for spatial/morphological info
        try:
            spatial_tensor = survey_tensor.get_spatial_tensor()
            astro_features = self.astrometry_encoder(spatial_tensor)
            features.append(astro_features)
        except Exception:
            batch_size = len(survey_tensor)
            astro_features = torch.zeros(
                batch_size, self.hidden_dim // 2, device=survey_tensor._data.device
            )
            features.append(astro_features)

        # Fuse features
        combined_features = torch.cat(features, dim=-1)
        fused_features = self.feature_fusion(combined_features)
        return fused_features

    def forward(
        self,
        x: Union[torch.Tensor, SurveyTensor],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with AstroPhot integration."""
        # Handle different input types
        if hasattr(x, "_data") and hasattr(x, "get_photometric_tensor"):
            h = self.extract_galaxy_features(x)  # type: ignore
        else:
            h = x if isinstance(x, torch.Tensor) else x._data  # type: ignore
            if h.dim() == 1:
                h = h.unsqueeze(0)

        if not isinstance(h, torch.Tensor):
            h = h._data  # type: ignore

        if h.size(-1) != self.hidden_dim:
            if not hasattr(self, "input_proj"):
                self.input_proj = Linear(h.size(-1), self.hidden_dim).to(h.device)
            h = self.input_proj(h)

        # Graph convolutions for galaxy neighborhood effects
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)

            if self.training:
                h = F.dropout(h, p=self.dropout)

            if i > 0 and h.size(-1) == h_prev.size(-1):
                h = h + h_prev

        # Global pooling for galaxy-level predictions
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        pooled = global_mean_pool(h, batch)

        # Component-specific predictions
        if return_components:
            results = {}
            for component, head in self.component_heads.items():
                results[component] = head(pooled)
            results["global"] = self.global_head(pooled)
            return results
        else:
            return self.global_head(pooled)

class SersicParameterHead(nn.Module):
    """Output head for Sersic profile parameters."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim // 2, 4),  # [Re, n, I_e, PA]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.head(x)

        # Apply parameter-specific constraints
        re = F.softplus(params[..., 0:1])  # Effective radius > 0
        n = torch.clamp(params[..., 1:2], 0.1, 8.0)  # Sersic index
        ie = F.softplus(params[..., 2:3])  # Surface brightness > 0
        pa = torch.remainder(params[..., 3:4], 180.0)  # Position angle [0, 180)

        return torch.cat([re, n, ie, pa], dim=-1)

class DiskParameterHead(nn.Module):
    """Output head for exponential disk parameters."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim // 2, 3),  # [Rd, I0, PA]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.head(x)

        rd = F.softplus(params[..., 0:1])  # Scale radius > 0
        i0 = F.softplus(params[..., 1:2])  # Central surface brightness > 0
        pa = torch.remainder(params[..., 2:3], 180.0)  # Position angle

        return torch.cat([rd, i0, pa], dim=-1)

class BulgeParameterHead(nn.Module):
    """Output head for bulge component parameters."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim // 2, 3),  # [Rb, Ib, q]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.head(x)

        rb = F.softplus(params[..., 0:1])  # Bulge radius > 0
        ib = F.softplus(params[..., 1:2])  # Bulge intensity > 0
        q = torch.sigmoid(params[..., 2:3])  # Axis ratio [0, 1]

        return torch.cat([rb, ib, q], dim=-1)

class GlobalGalaxyHead(nn.Module):
    """Output head for global galaxy parameters."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

# NSA-specific model
class NSAGalaxyModeler(AstroPhotGNN):
    """Specialized model for NSA galaxy catalog."""

    def __init__(self, **kwargs):
        super().__init__(
            model_components=["sersic", "disk", "bulge"],
            output_dim=20,  # Rich NSA parameter set
            **kwargs,
        )

__all__ = [
    "AstroPhotGNN",
    "NSAGalaxyModeler",
    "SersicParameterHead",
    "DiskParameterHead",
    "BulgeParameterHead",
    "GlobalGalaxyHead",
]
