"""
TensorDict-Native Survey GNN Models
==================================

Graph Neural Network models for astronomical survey data processing
using native TensorDict methods and properties.
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

# Import our TensorDict classes to use their methods
from astro_lab.tensors.tensordict_astro import (
    PhotometricTensorDict,
    SpatialTensorDict,
    SpectralTensorDict,
    SurveyTensorDict,
)

from ..components.base import BaseGNNLayer, TensorDictFeatureProcessor
from ..encoders import SurveyEncoder


class AstroSurveyGNN(nn.Module):
    """
    Graph Neural Network for astronomical survey data using native TensorDict methods.

    Processes SurveyTensorDict data through specialized encoders and GNN layers,
    utilizing the native methods and properties of our TensorDict classes.
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        use_photometry: bool = True,
        use_astrometry: bool = True,
        use_spectroscopy: bool = False,
        pooling_type: str = "mean",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.use_photometry = use_photometry
        self.use_astrometry = use_astrometry
        self.use_spectroscopy = use_spectroscopy
        self.pooling_type = pooling_type
        self.dropout = dropout

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # TensorDict-native feature processor
        self.feature_processor = TensorDictFeatureProcessor(
            output_dim=hidden_dim,
            feature_dim=hidden_dim,
            use_photometry=use_photometry,
            use_astrometry=use_astrometry,
            use_spectroscopy=use_spectroscopy,
            device=device,
        )

        # Survey encoder using native TensorDict methods
        self.survey_encoder = SurveyEncoder(
            output_dim=hidden_dim,
            use_photometry=use_photometry,
            use_astrometry=use_astrometry,
            use_spectroscopy=use_spectroscopy,
            device=device,
            **kwargs,
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList(
            [
                BaseGNNLayer(
            input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    layer_type="gcn",
            dropout=dropout,
                    device=device,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        # Global pooling
        from ..components.base import PoolingModule

        self.pooling = PoolingModule(pooling_type=pooling_type)

        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.to(self.device)

    def forward(
        self,
        data: SurveyTensorDict,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using native SurveyTensorDict methods.

        Args:
            data: SurveyTensorDict with native method access
            edge_index: Graph edge indices
            batch: Batch assignment for nodes

        Returns:
            Encoded survey features
        """
        if not isinstance(data, SurveyTensorDict):
            raise ValueError("AstroSurveyGNN requires SurveyTensorDict input")

        # Use survey encoder with native TensorDict methods
        node_features = self.survey_encoder(data)

        # Create graph edge index if not provided
        if edge_index is None:
            # Create k-NN graph using spatial coordinates if available
            if "spatial" in data and isinstance(data["spatial"], SpatialTensorDict):
                spatial_data = data["spatial"]
                # Use native 3D coordinate access for point cloud processing
                coordinates = torch.stack(
                    [spatial_data.x, spatial_data.y, spatial_data.z], dim=-1
                )
                edge_index = self._create_knn_graph(coordinates, k=8)
            else:
                # Create fully connected graph as fallback
                num_nodes = node_features.shape[0]
                edge_index = self._create_fully_connected_graph(num_nodes)

        edge_index = edge_index.to(self.device)

        # Process through GNN layers
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling
        graph_embedding = self.pooling(h, batch)

        # Final projection
        output = self.output_projection(graph_embedding)

        return output

    def _create_knn_graph(self, coordinates: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Create k-NN graph from spatial coordinates using native methods."""
        device = coordinates.device
        num_nodes = coordinates.shape[0]

        # Compute pairwise distances
        distances = torch.cdist(coordinates, coordinates)

        # Get k nearest neighbors (excluding self)
        _, knn_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        knn_indices = knn_indices[:, 1:]  # Remove self-connections

        # Create edge index
        source_nodes = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k)
        edge_index = torch.stack([source_nodes.flatten(), knn_indices.flatten()])

        return edge_index

    def _create_fully_connected_graph(self, num_nodes: int) -> torch.Tensor:
        """Create fully connected graph as fallback."""
        source = torch.arange(num_nodes).unsqueeze(1).expand(-1, num_nodes).flatten()
        target = torch.arange(num_nodes).unsqueeze(0).expand(num_nodes, -1).flatten()

        # Remove self-loops
        mask = source != target
        edge_index = torch.stack([source[mask], target[mask]])

        return edge_index

    def get_survey_metadata(self, data: SurveyTensorDict) -> Dict[str, Any]:
        """Extract metadata using native SurveyTensorDict methods."""
        if not isinstance(data, SurveyTensorDict):
            raise ValueError("Requires SurveyTensorDict input")

        metadata = {}

        # Survey-level metadata using meta property
        if hasattr(data, "meta") and data.meta is not None:
            meta = data.meta
            if "survey_name" in meta:
                metadata["survey_name"] = meta["survey_name"]
            if "data_release" in meta:
                metadata["data_release"] = meta["data_release"]
        if hasattr(data, "n_objects"):
            metadata["n_objects"] = data.n_objects

        # Photometric metadata
        if "photometric" in data and isinstance(
            data["photometric"], PhotometricTensorDict
        ):
            phot_data = data["photometric"]
            metadata["n_bands"] = phot_data.n_bands
            metadata["bands"] = phot_data.bands
            if hasattr(phot_data, "magnitude_system"):
                metadata["magnitude_system"] = phot_data.magnitude_system

        # Spatial metadata
        if "spatial" in data and isinstance(data["spatial"], SpatialTensorDict):
            spatial_data = data["spatial"]
            metadata["coordinate_system"] = spatial_data.coordinate_system
            if hasattr(spatial_data, "epoch"):
                metadata["epoch"] = spatial_data.epoch

        # Spectral metadata
        if "spectral" in data and isinstance(data["spectral"], SpectralTensorDict):
            spec_data = data["spectral"]
            if hasattr(spec_data, "wavelength_unit"):
                metadata["wavelength_unit"] = spec_data.wavelength_unit
            if hasattr(spec_data, "flux_unit"):
                metadata["flux_unit"] = spec_data.flux_unit

        return metadata


class MultiModalSurveyGNN(AstroSurveyGNN):
    """
    Extended Survey GNN for multi-modal data fusion using native TensorDict methods.

    Handles complex survey data with cross-modal attention and specialized processing
    for different astronomical data modalities.
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 4,
        use_cross_modal_attention: bool = True,
        **kwargs,
    ):
        super().__init__(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            **kwargs,
        )

        self.num_attention_heads = num_attention_heads
        self.use_cross_modal_attention = use_cross_modal_attention

        # Cross-modal attention for data fusion
        if use_cross_modal_attention:
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=self.dropout,
                batch_first=True,
            )

        # Modal-specific projections
        self.modal_projections = nn.ModuleDict(
            {
                "photometry": nn.Linear(hidden_dim, hidden_dim),
                "astrometry": nn.Linear(hidden_dim, hidden_dim),
                "spectroscopy": nn.Linear(hidden_dim, hidden_dim),
            }
        )

    def forward(
        self,
        data: SurveyTensorDict,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with multi-modal fusion using native methods."""
        if not isinstance(data, SurveyTensorDict):
            raise ValueError("MultiModalSurveyGNN requires SurveyTensorDict input")

        # Extract modal features using native methods
        modal_features = []
        modal_masks = []

        # Process photometry if available
        if self.use_photometry and "photometric" in data:
            phot_data = data["photometric"]
            if isinstance(phot_data, PhotometricTensorDict):
                # Use photometry encoder with native methods
                phot_features = self.survey_encoder.photometry_encoder(phot_data)
                phot_features = self.modal_projections["photometry"](phot_features)
                modal_features.append(phot_features)
                modal_masks.append(torch.ones(phot_features.shape[0], dtype=torch.bool))

        # Process astrometry if available
        if self.use_astrometry and "spatial" in data:
            spatial_data = data["spatial"]
            if isinstance(spatial_data, SpatialTensorDict):
                # Use astrometry encoder with native coordinate access
                astro_features = self.survey_encoder.astrometry_encoder(spatial_data)
                astro_features = self.modal_projections["astrometry"](astro_features)
                modal_features.append(astro_features)
                modal_masks.append(
                    torch.ones(astro_features.shape[0], dtype=torch.bool)
                )

        # Process spectroscopy if available
        if self.use_spectroscopy and "spectral" in data:
            spec_data = data["spectral"]
            if isinstance(spec_data, SpectralTensorDict):
                # Use spectroscopy encoder with native methods
                spec_features = self.survey_encoder.spectroscopy_encoder(spec_data)
                spec_features = self.modal_projections["spectroscopy"](spec_features)
                modal_features.append(spec_features)
                modal_masks.append(torch.ones(spec_features.shape[0], dtype=torch.bool))

        if not modal_features:
            raise ValueError("No compatible modal data found in SurveyTensorDict")

        # Fuse modal features
        if len(modal_features) == 1:
            fused_features = modal_features[0]
        else:
            # Stack and apply cross-modal attention
            stacked_features = torch.stack(
                modal_features, dim=1
            )  # [batch, modals, features]

            if self.use_cross_modal_attention:
                attended_features, _ = self.cross_modal_attention(
                    stacked_features, stacked_features, stacked_features
                )
                fused_features = attended_features.mean(dim=1)  # Average over modals
            else:
                fused_features = stacked_features.mean(dim=1)

        # Continue with standard GNN processing
        edge_index = edge_index or self._create_default_edges(fused_features.shape[0])
        edge_index = edge_index.to(self.device)

        # Process through GNN layers
        h = fused_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling and output
        graph_embedding = self.pooling(h, batch)
        output = self.output_projection(graph_embedding)

        return output

    def _create_default_edges(self, num_nodes: int) -> torch.Tensor:
        """Create default edge structure."""
        if num_nodes <= 50:
            return self._create_fully_connected_graph(num_nodes)
        else:
            # For larger graphs, create a sparse structure
            return self._create_random_graph(num_nodes, edge_ratio=0.1)

    def _create_random_graph(
        self, num_nodes: int, edge_ratio: float = 0.1
    ) -> torch.Tensor:
        """Create random sparse graph."""
        num_edges = int(num_nodes * (num_nodes - 1) * edge_ratio / 2)

        # Generate random edges
        edges = []
        for _ in range(num_edges):
            src = torch.randint(0, num_nodes, (1,))
            tgt = torch.randint(0, num_nodes, (1,))
            if src != tgt:
                edges.append([src, tgt])
                edges.append([tgt, src])  # Undirected

        if edges:
            edge_index = torch.tensor(edges).T
            return edge_index
        else:
            # Fallback to simple chain
            source = torch.arange(num_nodes - 1)
            target = torch.arange(1, num_nodes)
            edge_index = torch.stack([source, target])
            return edge_index
