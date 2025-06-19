"""
3D Point Cloud Models für Sternkarten

Spezialisierte GNN-Modelle für 3D-Sternverteilungen mit:
- PointNet++ für hierarchische Strukturen
- Gravitational Message Passing
- Multi-Scale Stellar Analysis
- Adaptive Sterngewichtung
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATConv,
    MessagePassing,
    PointNetConv,
    knn_graph,
    radius_graph,
)
from torch_geometric.transforms import KNNGraph, SamplePoints

from astro_lab.models.base_gnn import BaseAstroGNN
from astro_lab.tensors import SurveyTensor


class GravitationalMessagePassing(MessagePassing):
    """Custom message passing for gravitational interactions between stars."""

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")  # Sum gravitational forces

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable gravitational constant scaling
        self.grav_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute gravitational message passing."""
        # Calculate distances between stellar positions
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)

        # Avoid division by zero and numerical instability
        dist = torch.clamp(dist, min=1e-6)

        return self.propagate(edge_index, x=x, dist=dist)

    def message(self, x_i, x_j, dist):
        """Create messages based on gravitational interaction."""
        # Inverse square law weighting with learnable scaling
        weight = self.grav_scale / (dist**2 + 1e-6)

        # Combine stellar features with distance-based weighting
        msg_input = torch.cat([x_i, x_j, weight], dim=-1)
        return self.mlp(msg_input)


class StellarPointCloudGNN(BaseAstroGNN):
    """GNN for 3D stellar point cloud processing with PointNet++ layers."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_neighbors: int = 6,
        sample_points: int = 1024,
        radius: float = 0.1,  # Physical radius in parsecs
        use_gravitational_mp: bool = True,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, **kwargs)

        self.num_neighbors = num_neighbors
        self.sample_points = sample_points
        self.radius = radius
        self.use_gravitational_mp = use_gravitational_mp

        # Point cloud preprocessing transforms
        self.transform = T.Compose(
            [
                SamplePoints(num=sample_points),  # Sample stars per region
                KNNGraph(k=num_neighbors),  # K nearest neighbor stars
            ]
        )

        # PointNet++ layers for hierarchical stellar structures
        self.pointnet_layers = nn.ModuleList()
        for _ in range(3):
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for position features
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            global_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.pointnet_layers.append(PointNetConv(local_nn, global_nn))

        # GAT layers for adaptive star weighting (brightness/importance)
        self.attention_layers = nn.ModuleList(
            [
                GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1),
                GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1),
            ]
        )

        # Gravitational message passing
        if use_gravitational_mp:
            self.grav_mp = GravitationalMessagePassing(hidden_dim)

        # Stellar feature encoder
        self.stellar_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),  # Start with magnitude
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

    def build_stellar_graph(
        self,
        pos: torch.Tensor,  # [N, 3] stellar coordinates
        features: torch.Tensor,  # [N, F] magnitude, color, spectral class
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build graph structure from stellar positions."""
        # Use radius graph for physical neighborhoods
        edge_index = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=32)

        # Combine with KNN for sparse regions
        knn_edge_index = knn_graph(pos, k=self.num_neighbors, batch=batch)
        edge_index = torch.cat([edge_index, knn_edge_index], dim=1)

        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def forward(self, data: Union[Data, SurveyTensor]) -> Dict[str, torch.Tensor]:
        """Process 3D stellar point cloud."""
        if isinstance(data, SurveyTensor):
            # Extract from SurveyTensor
            spatial_tensor = data.get_spatial_tensor()
            pos = spatial_tensor._data  # [N, 3] coordinates

            # Get photometric features if available
            try:
                phot_tensor = data.get_photometric_tensor()
                features = phot_tensor._data
            except:
                # Fallback: use unit features
                features = torch.ones(pos.size(0), 1, device=pos.device)

            batch = None
        else:
            pos = data.pos
            features = data.x
            batch = getattr(data, "batch", None)

        # Encode stellar features
        h = self.stellar_encoder(features)

        # Build stellar graph structure
        edge_index = self.build_stellar_graph(pos, features, batch)

        # Apply PointNet++ layers for hierarchical processing
        for pointnet in self.pointnet_layers:
            h = pointnet(h, pos, edge_index)
            h = F.relu(h)

        # Apply attention layers for adaptive weighting
        for gat in self.attention_layers:
            h = gat(h, edge_index)
            h = F.elu(h)

        # Apply gravitational message passing if enabled
        if self.use_gravitational_mp:
            h = self.grav_mp(h, edge_index, pos)

        return {"embeddings": h, "edge_index": edge_index, "positions": pos}


class HierarchicalStellarGNN(BaseAstroGNN):
    """Hierarchical GNN for multi-scale stellar structures."""

    def __init__(
        self,
        scales: List[float] = [0.1, 1.0, 10.0],  # parsecs
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.scales = scales

        # Scale-specific processors
        self.scale_processors = nn.ModuleList()
        for scale in scales:
            # Create proper PointNetConv with MLPs
            local_nn = nn.Sequential(
                nn.Linear(self.hidden_dim + 3, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            global_nn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            pointnet = PointNetConv(local_nn, global_nn)

            processor = nn.ModuleList(
                [
                    pointnet,
                    nn.ReLU(),
                    GATConv(
                        self.hidden_dim, self.hidden_dim // 4, heads=4, concat=True
                    ),
                ]
            )
            self.scale_processors.append(processor)

        # Cross-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(len(scales) * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Hierarchical pooling for different scales
        self.hierarchical_pools = nn.ModuleList(
            [nn.AdaptiveAvgPool1d(1) for _ in scales]
        )

    def forward(
        self, pos: torch.Tensor, x: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process stellar data at multiple scales."""
        # Project input features to hidden dimension first
        h = self.input_projection(x)

        scale_features = []

        for scale, processor in zip(self.scales, self.scale_processors):
            # Build graph at this scale
            edge_index = radius_graph(pos, r=scale, batch=batch)

            # Process at this scale
            h_scale = h  # Use projected features
            h_scale = processor[0](h_scale, pos, edge_index)  # PointNetConv
            h_scale = processor[1](h_scale)  # ReLU
            h_scale = processor[2](h_scale, edge_index)  # GATConv

            scale_features.append(h_scale)

        # Fuse multi-scale features
        multi_scale = torch.cat(scale_features, dim=-1)
        return self.scale_fusion(multi_scale)


class StellarClusterGNN(BaseAstroGNN):
    """Specialized GNN for stellar cluster analysis."""

    def __init__(
        self,
        cluster_detection: bool = True,
        age_estimation: bool = True,
        metallicity_estimation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cluster_detection = cluster_detection
        self.age_estimation = age_estimation
        self.metallicity_estimation = metallicity_estimation

        # Stellar evolution features
        self.evolution_encoder = nn.Sequential(
            nn.Linear(5, self.hidden_dim // 2),  # B-V, V-I, etc.
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
        )

        # Cluster-specific heads
        if cluster_detection:
            self.cluster_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        if age_estimation:
            self.age_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )

        if metallicity_estimation:
            self.metallicity_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )

    def forward(
        self,
        pos: torch.Tensor,
        colors: torch.Tensor,  # Color indices
        magnitudes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Analyze stellar cluster properties."""
        # Combine color and magnitude information
        stellar_features = torch.cat([colors, magnitudes], dim=-1)
        h = self.evolution_encoder(stellar_features)

        # Apply graph convolutions
        h = self.graph_forward(h, edge_index)

        outputs = {"embeddings": h}

        # Cluster detection
        if self.cluster_detection:
            cluster_prob = self.cluster_head(h)
            outputs["cluster_probability"] = cluster_prob

        # Age estimation (log years)
        if self.age_estimation:
            log_age = self.age_head(h)
            age = 10 ** (6 + 3 * torch.sigmoid(log_age))  # 1 Myr to 10 Gyr
            outputs["age"] = age

        # Metallicity estimation
        if self.metallicity_estimation:
            metallicity = self.metallicity_head(h)
            outputs["metallicity"] = metallicity

        return outputs


class GalacticStructureGNN(BaseAstroGNN):
    """GNN for analyzing galactic structure from stellar distributions."""

    def __init__(
        self,
        spiral_detection: bool = True,
        bar_detection: bool = True,
        halo_analysis: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.spiral_detection = spiral_detection
        self.bar_detection = bar_detection
        self.halo_analysis = halo_analysis

        # Galactic coordinate encoder
        self.galactic_encoder = nn.Sequential(
            nn.Linear(6, self.hidden_dim),  # l, b, distance, pm_l, pm_b, vrad
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Structure-specific analyzers
        if spiral_detection:
            self.spiral_analyzer = nn.LSTM(
                self.hidden_dim, self.hidden_dim // 2, num_layers=2, batch_first=True
            )
            self.spiral_head = nn.Linear(self.hidden_dim // 2, 4)  # 4 spiral arms

        if bar_detection:
            self.bar_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 3),  # bar strength, angle, length
            )

        if halo_analysis:
            self.halo_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 2),  # halo density, velocity dispersion
            )

    def forward(
        self,
        galactic_coords: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Analyze galactic structure from stellar positions."""
        # Encode galactic coordinates
        h = self.galactic_encoder(galactic_coords)

        # Apply graph convolutions
        h = self.graph_forward(h, edge_index)

        outputs = {"embeddings": h}

        # Spiral structure analysis
        if self.spiral_detection:
            # Reshape for LSTM (assuming radial ordering)
            h_spiral = h.unsqueeze(1)  # Add sequence dimension
            spiral_out, _ = self.spiral_analyzer(h_spiral)
            spiral_params = self.spiral_head(spiral_out.squeeze(1))
            outputs["spiral_parameters"] = spiral_params

        # Bar detection
        if self.bar_detection:
            bar_params = self.bar_head(h)
            outputs["bar_parameters"] = bar_params

        # Halo analysis
        if self.halo_analysis:
            halo_params = self.halo_head(h)
            outputs["halo_parameters"] = halo_params

        return outputs


# Factory functions for 3D stellar models
def create_stellar_point_cloud_model(
    model_type: str = "point_cloud",
    num_stars: int = 1024,
    radius: float = 0.1,
    scales: Optional[List[float]] = None,
    **kwargs,
) -> nn.Module:
    """Create specialized model for 3D stellar data."""

    if model_type == "point_cloud":
        return StellarPointCloudGNN(sample_points=num_stars, radius=radius, **kwargs)
    elif model_type == "hierarchical":
        return HierarchicalStellarGNN(scales=scales or [0.1, 1.0, 10.0], **kwargs)
    elif model_type == "cluster":
        return StellarClusterGNN(**kwargs)
    elif model_type == "galactic":
        return GalacticStructureGNN(**kwargs)
    else:
        raise ValueError(f"Unknown 3D stellar model type: {model_type}")


def create_optimized_stellar_graph(
    positions: torch.Tensor,
    magnitudes: torch.Tensor,
    radius: float = 0.1,
    k_neighbors: int = 6,
    magnitude_threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create optimized graph structure for stellar data."""

    # Filter by magnitude (remove very faint stars)
    bright_mask = magnitudes < magnitude_threshold
    pos_filtered = positions[bright_mask]

    # Build radius graph for physical neighbors
    edge_index = radius_graph(pos_filtered, r=radius, max_num_neighbors=32)

    # Add KNN edges for isolated stars
    knn_edges = knn_graph(pos_filtered, k=k_neighbors)

    # Combine and deduplicate
    combined_edges = torch.cat([edge_index, knn_edges], dim=1)
    edge_index = torch.unique(combined_edges, dim=1)

    return pos_filtered, edge_index
