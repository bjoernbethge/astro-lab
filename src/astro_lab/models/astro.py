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
from astro_lab.models.layers import LayerFactory

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

        # Encoder for different data modalities
        if self.use_photometry:
            self.photometry_encoder = PhotometryEncoder(
                input_dim=len(kwargs.get("photometry_bands", [])), 
                output_dim=hidden_dim // 2,
                device=self.device
            )
        if self.use_spectroscopy:
            self.spectroscopy_encoder = SpectroscopyEncoder(
                input_dim=kwargs.get("spectroscopy_features", 100), 
                output_dim=hidden_dim // 2,
                device=self.device
            )
        if self.use_astrometry:
            self.astrometry_encoder = AstrometryEncoder(
                input_dim=kwargs.get("astrometry_features", 5), 
                output_dim=hidden_dim // 2,
                device=self.device
            )

        # Feature fusion
        fusion_dim = self._calculate_fusion_dim()
        self.feature_fusion = Linear(fusion_dim, hidden_dim)

        # Graph convolutions
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = LayerFactory.create_conv_layer(
                self.conv_type, hidden_dim, hidden_dim, **kwargs
            )
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
        self.to(self.device) # Ensure the entire model is on the correct device

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

    def extract_survey_features(self, survey_data: Union[SurveyTensor, Dict[str, Any]]) -> torch.Tensor:
        """Extract and fuse features from a SurveyTensor or a dictionary of tensors."""
        features = []
        
        is_tensor_obj = not isinstance(survey_data, dict)

        def get_data_from_source(feature_type: str):
            if is_tensor_obj:
                try:
                    method_name = f"get_{feature_type}_tensor"
                    return getattr(survey_data, method_name)()
                except (ValueError, AttributeError):
                    return None
            else:
                return survey_data.get(feature_type)

        if self.use_photometry:
            phot_data = get_data_from_source("photometry")
            if phot_data is not None:
                features.append(self.photometry_encoder(phot_data.to(self.device)))

        if self.use_astrometry:
            astro_data = get_data_from_source("astrometry") or get_data_from_source("spatial")
            if astro_data is not None:
                features.append(self.astrometry_encoder(astro_data.to(self.device)))

        if self.use_spectroscopy:
            spec_data = get_data_from_source("spectroscopy")
            if spec_data is not None:
                features.append(self.spectroscopy_encoder(spec_data.to(self.device)))

        if not features:
             raise ValueError("No features could be extracted from the input data.")

        if len(features) > 1:
            combined_features = torch.cat(features, dim=-1)
            fused_features = self.feature_fusion(combined_features)
        else:
            fused_features = features[0]
            
        return fused_features

    def forward(
        self,
        x: Union[torch.Tensor, SurveyTensor, Dict[str, Any]],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with native tensor support."""
        # Ensure edge_index is on the correct device
        edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)

        if isinstance(x, (SurveyTensor, dict)):
            h = self.extract_survey_features(x)
        else:
            h = x.to(self.device)

        if not isinstance(h, torch.Tensor):
            h = h.data.to(self.device)

        if h.size(-1) != self.hidden_dim:
            if not hasattr(self, "input_proj"):
                self.input_proj = Linear(h.size(-1), self.hidden_dim).to(h.device)
            h = self.input_proj(h)

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
            
            pooling_fn = self.get_pooling_fn(self.pooling)
            h = pooling_fn(h, batch)

        output = self.output_head(h)

        if return_embeddings:
            return {"logits": output, "embeddings": embeddings}
        return output

__all__ = ["AstroSurveyGNN"]
