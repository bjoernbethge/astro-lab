"""
Modern Graph Neural Network models for astronomical data

Optimized for PyTorch Geometric 2.6+ with native AstroLab tensor integration:
- SurveyTensor: Main coordinator for survey data
- PhotometricTensor: Multi-band photometry
- SpectralTensor: Spectroscopic data
- Spatial3DTensor: 3D spatial coordinates
- LightcurveTensor: Time-series data
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    Linear,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from astro_lab.models.base_gnn import BaseAstroGNN, ConvType, FeatureFusion
from astro_lab.models.encoders import (
    AstrometryEncoder,
    PhotometryEncoder,
    SpectroscopyEncoder,
)
from astro_lab.models.output_heads import OutputHeadRegistry, create_output_head
from astro_lab.models.utils import get_activation, initialize_weights
from astro_lab.tensors import SurveyTensor


class AstroSurveyGNN(BaseAstroGNN):
    """Graph Neural Network for astronomical survey data with native tensor support."""

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 1,
        conv_type: ConvType = "gcn",
        num_layers: int = 3,
        dropout: float = 0.1,
        task: str = "node_classification",
        use_photometry: bool = True,
        use_astrometry: bool = True,
        use_spectroscopy: bool = False,
        use_3d_stellar_processing: bool = False,
        stellar_radius: float = 0.1,
        pooling: str = "mean",
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            conv_type=conv_type,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )

        self.output_dim = output_dim
        self.task = task
        self.use_photometry = use_photometry
        self.use_astrometry = use_astrometry
        self.use_spectroscopy = use_spectroscopy
        self.use_3d_stellar_processing = use_3d_stellar_processing
        self.stellar_radius = stellar_radius
        self.pooling = pooling

        # Feature encoders
        if use_photometry:
            self.photometry_encoder = PhotometryEncoder(hidden_dim // 2)
        if use_astrometry:
            self.astrometry_encoder = AstrometryEncoder(hidden_dim // 2)
        if use_spectroscopy:
            self.spectroscopy_encoder = SpectroscopyEncoder(hidden_dim // 2)

        # Feature fusion
        fusion_dim = self._calculate_fusion_dim()
        self.feature_fusion = Linear(fusion_dim, hidden_dim)

        # Graph convolutions
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim, normalize=True)
            elif conv_type == "gat":
                conv = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            elif conv_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim, normalize=True)
            elif conv_type == "transformer":
                conv = TransformerConv(
                    hidden_dim, hidden_dim // 8, heads=8, dropout=dropout
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.convs.append(conv)

        # Normalization
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Output head
        self.output_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            get_activation("relu"),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, output_dim),
        )

        self.apply(initialize_weights)

    def _calculate_fusion_dim(self) -> int:
        """Calculate dimension for feature fusion."""
        dim = 0
        if self.use_photometry:
            dim += self.hidden_dim // 2
        if self.use_astrometry:
            dim += self.hidden_dim // 2
        if self.use_spectroscopy:
            dim += self.hidden_dim // 2

        return max(dim, self.hidden_dim)

    def extract_survey_features(self, survey_tensor: SurveyTensor) -> torch.Tensor:
        """Extract features from SurveyTensor using native methods."""
        features = []

        if self.use_photometry:
            try:
                phot_tensor = survey_tensor.get_photometric_tensor()
                phot_features = self.photometry_encoder(phot_tensor)
                features.append(phot_features)
            except (ValueError, AttributeError):
                batch_size = len(survey_tensor)
                phot_features = torch.zeros(
                    batch_size, self.hidden_dim // 2, device=survey_tensor._data.device
                )
                features.append(phot_features)

        if self.use_astrometry:
            try:
                spatial_tensor = survey_tensor.get_spatial_tensor()
                astro_features = self.astrometry_encoder(spatial_tensor)
                features.append(astro_features)
            except (ValueError, AttributeError):
                batch_size = len(survey_tensor)
                astro_features = torch.zeros(
                    batch_size, self.hidden_dim // 2, device=survey_tensor._data.device
                )
                features.append(astro_features)

        if self.use_spectroscopy:
            try:
                spec_features = self.spectroscopy_encoder(survey_tensor)
                features.append(spec_features)
            except (ValueError, AttributeError):
                batch_size = len(survey_tensor)
                spec_features = torch.zeros(
                    batch_size, self.hidden_dim // 2, device=survey_tensor._data.device
                )
                features.append(spec_features)

        if features:
            combined_features = torch.cat(features, dim=-1)
            fused_features = self.feature_fusion(combined_features)
        else:
            raw_data = survey_tensor._data
            if not hasattr(self, "raw_data_proj"):
                self.raw_data_proj = Linear(raw_data.size(-1), self.hidden_dim).to(
                    raw_data.device
                )
            fused_features = self.raw_data_proj(raw_data)

        return fused_features

    def forward(
        self,
        x: Union[torch.Tensor, SurveyTensor],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with native tensor support."""
        # Handle different input types
        if hasattr(x, "_data") and hasattr(x, "get_photometric_tensor"):
            h = self.extract_survey_features(x)  # type: ignore
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

        # Graph convolutions
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)

            if self.training:
                h = F.dropout(h, p=self.dropout)

            if i > 0 and h.size(-1) == h_prev.size(-1):
                h = h + h_prev

        embeddings = h

        if "graph" in self.task:
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

            if self.pooling == "mean":
                h = global_mean_pool(h, batch)
            elif self.pooling == "max":
                h = global_max_pool(h, batch)
            elif self.pooling == "add":
                h = global_add_pool(h, batch)
            else:
                h = global_mean_pool(h, batch)

        output = self.output_head(h)

        if return_embeddings:
            return {"output": output, "embeddings": embeddings}
        return output


__all__ = ["AstroSurveyGNN"]
