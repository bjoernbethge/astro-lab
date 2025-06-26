"""
TensorDict-Native AstroPhotometry GNN Models
==========================================

Graph Neural Network models for astronomical photometry analysis
using native PhotometricTensorDict methods and properties.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

# Import our TensorDict classes to use their methods
from astro_lab.tensors.tensordict_astro import PhotometricTensorDict

from ..components.base import BaseGNNLayer, TensorDictFeatureProcessor
from ..encoders import PhotometryEncoder


class AstroPhotGNN(nn.Module):
    """
    Graph Neural Network for astronomical photometry using native PhotometricTensorDict methods.

    Processes multi-band photometric data through specialized encoders and GNN layers,
    utilizing the native methods and properties of PhotometricTensorDict.
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        use_color_features: bool = True,
        use_magnitude_errors: bool = True,
        pooling_type: str = "mean",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.use_color_features = use_color_features
        self.use_magnitude_errors = use_magnitude_errors
        self.pooling_type = pooling_type
        self.dropout = dropout

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # PhotometricTensorDict-native encoder
        self.photometry_encoder = PhotometryEncoder(
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        # Color index processor for multi-band data
        if use_color_features:
            self.color_processor = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
            )

        # GNN layers for spatial photometric relationships
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

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.to(self.device)

    def forward(
        self,
        data: PhotometricTensorDict,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using native PhotometricTensorDict methods.

        Args:
            data: PhotometricTensorDict with native photometric access
            edge_index: Graph edge indices for photometric relationships
            batch: Batch assignment for objects

        Returns:
            Encoded photometric features
        """
        if not isinstance(data, PhotometricTensorDict):
            raise ValueError("AstroPhotGNN requires PhotometricTensorDict input")

        # Use photometry encoder with native methods
        node_features = self.photometry_encoder(data)

        # Extract additional color features using native methods
        if self.use_color_features and data.n_bands > 1:
            color_features = self._extract_color_features(data)
            if color_features is not None:
                # Combine with main features
                if hasattr(self, "color_processor"):
                    processed_colors = self.color_processor(color_features)
                    node_features = torch.cat([node_features, processed_colors], dim=-1)

                    # Adjust projection for concatenated features
                    if not hasattr(self, "_adjusted_projection"):
                        combined_dim = node_features.shape[-1]
                        self.feature_adjustment = nn.Linear(
                            combined_dim, self.hidden_dim
                        ).to(self.device)
                        self._adjusted_projection = True

                    if hasattr(self, "feature_adjustment"):
                        node_features = self.feature_adjustment(node_features)

        # Create edge index if not provided
        if edge_index is None:
            num_nodes = node_features.shape[0]
            edge_index = self._create_photometric_graph(data, num_nodes)

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

    def _extract_color_features(
        self, data: PhotometricTensorDict
    ) -> Optional[torch.Tensor]:
        """Extract color features using native PhotometricTensorDict methods."""
        if data.n_bands < 2:
            return None

        bands = data.bands

        # Create standard color pairs
        standard_colors = [
            ("u", "g"),
            ("g", "r"),
            ("r", "i"),
            ("i", "z"),  # SDSS colors
            ("B", "V"),
            ("V", "R"),
            ("R", "I"),  # Johnson colors
            ("J", "H"),
            ("H", "K"),  # Near-IR colors
        ]

        # Filter to available bands
        available_colors = []
        for color1, color2 in standard_colors:
            if color1 in bands and color2 in bands:
                available_colors.append((color1, color2))

        # Fall back to adjacent bands if no standard colors
        if not available_colors:
            for i in range(len(bands) - 1):
                available_colors.append((bands[i], bands[i + 1]))

        if not available_colors:
            return None

        try:
            # Use native compute_colors method
            color_dict = data.compute_colors(available_colors)

            # Extract color values
            color_values = []
            for color_name in color_dict.keys():
                if isinstance(color_dict[color_name], torch.Tensor):
                    color_values.append(color_dict[color_name])

            if color_values:
                colors = torch.stack(color_values, dim=-1)
                return colors.to(self.device)

        except Exception:
            # Fallback to manual color computation
            magnitudes = data["magnitudes"].to(self.device)
            if magnitudes.shape[-1] > 1:
                colors = magnitudes[..., :-1] - magnitudes[..., 1:]
                return colors

        return None

    def _create_photometric_graph(
        self, data: PhotometricTensorDict, num_nodes: int
    ) -> torch.Tensor:
        """Create graph based on photometric similarity using native methods."""
        if num_nodes <= 1:
            return torch.tensor([[0], [0]], device=self.device, dtype=torch.long)

        magnitudes = data["magnitudes"].to(self.device)

        # Compute photometric distances
        if magnitudes.dim() == 2:
            # Multi-band photometry
            distances = torch.cdist(magnitudes, magnitudes)
        else:
            # Single band
            distances = torch.abs(magnitudes.unsqueeze(1) - magnitudes.unsqueeze(0))

        # Create edges based on photometric similarity
        threshold = distances.std() * 0.75  # Adaptive threshold
        adjacency = (distances < threshold) & (distances > 0)

        # Convert to edge index
        edge_indices = adjacency.nonzero(as_tuple=False).T

        if edge_indices.shape[1] == 0:
            # Fallback: k-NN graph
            k = min(8, num_nodes - 1)
            _, knn_indices = torch.topk(distances, k + 1, dim=1, largest=False)
            knn_indices = knn_indices[:, 1:]  # Remove self-connections

            source_nodes = torch.arange(num_nodes).unsqueeze(1).expand(-1, k)
            edge_indices = torch.stack([source_nodes.flatten(), knn_indices.flatten()])

        return edge_indices

    def get_photometric_metadata(self, data: PhotometricTensorDict) -> Dict[str, Any]:
        """Extract photometric metadata using native methods."""
        if not isinstance(data, PhotometricTensorDict):
            raise ValueError("Requires PhotometricTensorDict input")

        metadata = {}

        # Basic photometric properties
        metadata["n_bands"] = data.n_bands
        metadata["bands"] = data.bands

        # Magnitude system information from meta
        if hasattr(data, "meta") and data.meta is not None:
            if "magnitude_system" in data.meta:
                metadata["magnitude_system"] = data.meta["magnitude_system"]

        # Statistical properties
        magnitudes = data["magnitudes"]
        metadata["magnitude_stats"] = {
            "mean": magnitudes.mean(dim=0).tolist(),
            "std": magnitudes.std(dim=0).tolist(),
            "min": magnitudes.min(dim=0)[0].tolist(),
            "max": magnitudes.max(dim=0)[0].tolist(),
        }

        # Color information
        if data.n_bands > 1:
            try:
                # Try to compute some standard colors
                color_pairs = [
                    (data.bands[i], data.bands[i + 1])
                    for i in range(len(data.bands) - 1)
                ]
                colors_dict = data.compute_colors(color_pairs)
                metadata["available_colors"] = [str(k) for k in colors_dict.keys()]
            except Exception:
                metadata["available_colors"] = []

        return metadata


class GalaxyPhotometryGNN(AstroPhotGNN):
    """
    Specialized photometry GNN for galaxy morphology and component modeling.

    Extended AstroPhotGNN for galaxy-specific photometric analysis using
    native PhotometricTensorDict methods with component decomposition.
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        component_types: List[str] = ["sersic", "disk", "bulge", "bar"],
        use_surface_brightness: bool = True,
        **kwargs,
    ):
        super().__init__(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.component_types = component_types
        self.use_surface_brightness = use_surface_brightness

        # Component-specific processing heads
        self.component_heads = nn.ModuleDict(
            {
                component: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_dim // 2, 16),  # Component parameters
                )
                for component in component_types
            }
        )

        # Surface brightness profile processor
        if use_surface_brightness:
            self.sb_processor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim // 2, 32),  # Radial profile features
            )

        # Galaxy classification head
        self.galaxy_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, 4),  # Galaxy types: E, S0, Sa-Sc, Irr
        )

    def forward(
        self,
        data: PhotometricTensorDict,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with galaxy component analysis.

        Returns:
            Dict with 'embedding', 'components', 'classification', 'sb_profile' keys
        """
        if not isinstance(data, PhotometricTensorDict):
            raise ValueError("GalaxyPhotometryGNN requires PhotometricTensorDict input")

        # Get base photometric features
        base_embedding = super().forward(data, edge_index, batch)

        # Component analysis
        component_outputs = {}
        for component in self.component_types:
            component_outputs[component] = self.component_heads[component](
                base_embedding
            )

        # Surface brightness analysis
        sb_profile = None
        if self.use_surface_brightness:
            sb_profile = self.sb_processor(base_embedding)

        # Galaxy classification
        galaxy_class = self.galaxy_classifier(base_embedding)

        results = {
            "embedding": base_embedding,
            "components": component_outputs,
            "classification": galaxy_class,
        }

        if sb_profile is not None:
            results["sb_profile"] = sb_profile

            return results

    def analyze_galaxy_colors(
        self, data: PhotometricTensorDict
    ) -> Dict[str, torch.Tensor]:
        """Analyze galaxy colors using native PhotometricTensorDict methods."""
        if not isinstance(data, PhotometricTensorDict):
            raise ValueError("Requires PhotometricTensorDict input")

        color_analysis = {}

        # Standard galaxy colors
        galaxy_colors = [
            ("u", "g"),
            ("g", "r"),
            ("r", "i"),
            ("i", "z"),  # SDSS
            ("B", "V"),
            ("V", "R"),
            ("R", "I"),  # Johnson
            ("NUV", "r"),
            ("FUV", "NUV"),  # UV colors
        ]

        available_colors = []
        for color1, color2 in galaxy_colors:
            if color1 in data.bands and color2 in data.bands:
                available_colors.append((color1, color2))

        if available_colors:
            try:
                colors_dict = data.compute_colors(available_colors)
                color_analysis.update(colors_dict)

                # Compute color-magnitude relations
                magnitudes = data["magnitudes"]
                for color_name, color_values in colors_dict.items():
                    if isinstance(color_values, torch.Tensor):
                        # Use r-band as reference magnitude if available
                        if "r" in data.bands:
                            r_idx = data.bands.index("r")
                            r_mag = magnitudes[..., r_idx]
                            color_analysis[f"{color_name}_vs_r"] = torch.stack(
                                [r_mag, color_values], dim=-1
                            )

            except Exception as e:
                print(f"Color analysis failed: {e}")

        return color_analysis
