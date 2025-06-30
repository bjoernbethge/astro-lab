"""
AstroGraphGNN - Graph Neural Network for Astronomical Data
=========================================================

Graph-level tasks with TensorDict integration and modular architecture.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

# PyTorch Geometric
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

# AstroLab components
from ..components.encoders import (
    MultiModalFusion,
    PhotometricEncoder,
    SpatialEncoder,
    SpectralEncoder,
)
from ..components.layers.convolution import FlexibleGraphConv
from ..components.layers.point_cloud import AstroPointCloudLayer, create_point_cloud_encoder
from ..components.layers.pooling import AdaptivePooling, StatisticalPooling
from ..components.output_heads import create_output_head
from .base_model import AstroBaseModel


class AstroGraphGNN(AstroBaseModel):
    """
    Graph neural network for astronomical graph-level tasks.

    Features:
    - TensorDict-based processing
    - Multi-modal encoder support
    - Flexible pooling strategies
    - Clean modular architecture
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gat",
        heads: int = 4,
        pooling: str = "adaptive",
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        task: str = "graph_classification",
        # Point cloud layer support
        graph_layer_type: Optional[str] = None,
        point_cloud_config: Optional[Dict[str, Any]] = None,
        # Output head configuration
        output_head: Optional[str] = None,
        num_harmonics: int = 10,
        # Multi-modal options
        use_photometric: bool = False,
        num_photometric_bands: Optional[int] = None,
        use_spectral: bool = False,
        spectral_wavelengths: Optional[int] = None,
        use_spatial: bool = True,
        spatial_dim: int = 3,
        use_tensordict: bool = True,
        **kwargs,
    ):
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.conv_type = conv_type
        self.heads = heads
        self.pooling = pooling
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.use_tensordict = use_tensordict
        
        # Point cloud configuration
        self.graph_layer_type = graph_layer_type
        self.point_cloud_config = point_cloud_config or {}
        
        # Output head configuration
        self.output_head_type = output_head
        self.num_harmonics = num_harmonics

        # Multi-modal configuration
        self.use_photometric = use_photometric
        self.use_spectral = use_spectral
        self.use_spatial = use_spatial

        # Build the model architecture
        self._build_model(
            num_features,
            hidden_dim,
            num_layers,
            num_photometric_bands,
            spectral_wavelengths,
            spatial_dim,
        )

    def _build_model(
        self,
        num_features: int,
        hidden_dim: int,
        num_layers: int,
        num_photometric_bands: Optional[int],
        spectral_wavelengths: Optional[int],
        spatial_dim: int,
    ):
        """Build the model architecture."""

        # Encoder modules for multi-modal data
        self.encoders = nn.ModuleList()
        self.modality_dims = {}

        if self.use_spatial:
            spatial_encoder = SpatialEncoder(
                spatial_dim=spatial_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
            self.encoders.append(spatial_encoder)
            self.modality_dims["spatial"] = hidden_dim

        if self.use_photometric and num_photometric_bands:
            photometric_encoder = PhotometricEncoder(
                num_bands=num_photometric_bands,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
            self.encoders.append(photometric_encoder)
            self.modality_dims["photometric"] = hidden_dim

        if self.use_spectral and spectral_wavelengths:
            spectral_encoder = SpectralEncoder(
                wavelength_dim=spectral_wavelengths,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
            self.encoders.append(spectral_encoder)
            self.modality_dims["spectral"] = hidden_dim

        # Multi-modal fusion if needed
        if len(self.modality_dims) > 1:
            self.fusion = MultiModalFusion(
                modality_dims=self.modality_dims,
                fusion_dim=hidden_dim,
            )
            self.feature_key = "fused_features"
        elif len(self.modality_dims) == 1:
            self.fusion = None
            self.feature_key = list(self.modality_dims.keys())[0] + "_features"
        else:
            # No multi-modal encoders, use raw features
            self.fusion = None
            self.feature_key = "x"
            self.input_projection = nn.Linear(num_features, hidden_dim)

        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        
        # Use point cloud layers if specified
        if self.graph_layer_type == "point_cloud":
            # Create point cloud encoder
            pc_config = self.point_cloud_config.copy()
            pc_config["input_dim"] = hidden_dim
            pc_config["output_dim"] = hidden_dim
            
            self.point_cloud_encoder = create_point_cloud_encoder(**pc_config)
            
            # No need for traditional graph layers
            self.graph_layers = None
        else:
            # Traditional graph convolution layers
            for i in range(num_layers):
                in_channels = hidden_dim
                out_channels = hidden_dim

                # Graph convolution
                conv = FlexibleGraphConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_type=self.conv_type,
                    heads=self.heads if self.conv_type in ["gat", "transformer"] else 1,
                    edge_dim=self.edge_dim,
                    dropout=self.dropout,
                )

                # Normalization
                norm = torch.nn.LayerNorm(out_channels)

                # Create layer module
                layer = nn.ModuleDict(
                    {
                        "conv": conv,
                        "norm": norm,
                        "act": nn.GELU(),
                        "dropout": nn.Dropout(self.dropout),
                    }
                )

                self.graph_layers.append(layer)
            
            self.point_cloud_encoder = None

        # Pooling layer
        if self.pooling == "adaptive":
            self.pooling_layer = AdaptivePooling(
                in_channels=hidden_dim,
                pooling_methods=["mean", "max", "attention"],
            )
            pooled_dim = hidden_dim * 3
        elif self.pooling == "statistical":
            self.pooling_layer = StatisticalPooling(
                moments=["mean", "std", "max", "min"]
            )
            pooled_dim = hidden_dim * 4
        else:
            # Simple pooling
            self.pooling_layer = None
            pooled_dim = hidden_dim

        # Output head
        if self.output_head_type:
            # Use custom output head
            self.output_head = create_output_head(
                self.output_head_type,
                input_dim=pooled_dim,
                output_dim=self.num_classes,
                num_harmonics=self.num_harmonics,
                dropout=self.dropout
            )
        else:
            # Default classification head
            self.output_head = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, self.num_classes),
            )

        # Input projection
        self.input_projection = nn.Linear(num_features, hidden_dim)

        # Feature fusion for multi-modal data
        if len(self.encoders) > 1:
            self.feature_fusion = MultiModalFusion(
                input_dims=[hidden_dim] * len(self.encoders),
                output_dim=hidden_dim,
            )
        else:
            self.feature_fusion = None

    def forward(self, batch: Union[Data, Batch, Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass through the Graph GNN.

        Args:
            batch: Input batch containing spatial, photometric, and/or spectral data

        Returns:
            Node embeddings [N, hidden_dim]
        """
        # Handle different input types
        if isinstance(batch, (Data, Batch)):
            # For PyG data, extract features directly
            x = getattr(batch, "x", None)
            edge_index = getattr(batch, "edge_index", None)
            edge_attr = getattr(batch, "edge_attr", None)
            batch_idx = getattr(batch, "batch", None)

            if x is None or edge_index is None:
                raise ValueError("PyG batch must have 'x' and 'edge_index'!")

        elif isinstance(batch, dict):
            # For TensorDict format, process through encoders
            node_features = []
            encoder_idx = 0

            # Process spatial data
            if (
                self.use_spatial
                and "spatial" in batch
                and encoder_idx < len(self.encoders)
            ):
                spatial_data = batch["spatial"]
                if isinstance(spatial_data, dict) and "coordinates" in spatial_data:
                    spatial_features = self.encoders[encoder_idx](
                        spatial_data["coordinates"]
                    )
                    node_features.append(spatial_features)
                    encoder_idx += 1

            # Process photometric data
            if (
                self.use_photometric
                and "photometric" in batch
                and encoder_idx < len(self.encoders)
            ):
                photometric_data = batch["photometric"]
                if (
                    isinstance(photometric_data, dict)
                    and "magnitudes" in photometric_data
                ):
                    photometric_features = self.encoders[encoder_idx](
                        photometric_data["magnitudes"]
                    )
                    node_features.append(photometric_features)
                    encoder_idx += 1

            # Process spectral data
            if (
                self.use_spectral
                and "spectral" in batch
                and encoder_idx < len(self.encoders)
            ):
                spectral_data = batch["spectral"]
                if isinstance(spectral_data, dict) and "flux" in spectral_data:
                    spectral_features = self.encoders[encoder_idx](
                        spectral_data["flux"]
                    )
                    node_features.append(spectral_features)
                    encoder_idx += 1

            # Process additional features
            if "features" in batch:
                features = batch["features"]
                if isinstance(features, torch.Tensor):
                    node_features.append(features)

            # Combine features
            if not node_features:
                raise ValueError("No valid input features found")

            if len(node_features) > 1:
                x = self.fusion(node_features)
            else:
                x = node_features[0]

            # For dict input, we need to create a minimal PyG structure
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            edge_attr = None
            batch_idx = None

        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)

        # Apply graph convolutions or point cloud encoder
        if self.point_cloud_encoder is not None:
            # Extract positions for point cloud processing
            pos = getattr(batch, "pos", None)
            if pos is None:
                # Try to get positions from x (first 3 dimensions)
                if x.shape[1] >= 3:
                    pos = x[:, :3]
                else:
                    raise ValueError("Point cloud layers require 3D positions")
            
            # Apply point cloud encoder
            x, _ = self.point_cloud_encoder(x, pos, batch_idx)
        else:
            # Traditional graph convolution
            for i, layer_dict in enumerate(self.graph_layers):
                x = layer_dict["conv"](x, edge_index, edge_attr=edge_attr)
                if "norm" in layer_dict:
                    norm_layer = layer_dict["norm"]
                    if isinstance(norm_layer, nn.LayerNorm):
                        x = norm_layer(x)
                    else:
                        x = norm_layer(x, batch=batch_idx)
                x = layer_dict["act"](x)
                x = layer_dict["dropout"](x)

        # Pooling
        if batch_idx is not None and x.dim() == 2:  # Only pool if not already pooled
            if self.pooling_layer is not None:
                x = self.pooling_layer(x, batch=batch_idx)
            else:
                # Simple pooling
                if self.pooling == "mean":
                    x = global_mean_pool(x, batch_idx)
                elif self.pooling == "max":
                    x = global_max_pool(x, batch_idx)
                elif self.pooling == "sum":
                    x = global_add_pool(x, batch_idx)
        elif x.dim() == 2 and x.size(0) > 1:  # Multiple nodes but no batch
            if self.pooling == "mean":
                x = x.mean(dim=0, keepdim=True)
            elif self.pooling == "max":
                x = x.max(dim=0)[0].unsqueeze(0)
            elif self.pooling == "sum":
                x = x.sum(dim=0, keepdim=True)

        # Output projection
        output = self.output_head(x)
        
        # Handle different output head types
        if self.output_head_type == "shape_modeling" and isinstance(output, dict):
            # Shape modeling returns a dictionary
            return output
        else:
            # Standard classification/regression output
            return output

    def _pyg_to_tensordict(self, pyg_data: Union[Data, Batch]) -> Dict[str, Any]:
        """Convert PyG data to TensorDict format."""
        tensordict = {}

        # Extract spatial data
        if hasattr(pyg_data, "pos") and pyg_data.pos is not None:
            tensordict["spatial"] = {
                "coordinates": pyg_data.pos,
                "coordinate_system": "icrs",
                "unit": "pc",
                "epoch": "J2000",
            }

        # Extract features
        if hasattr(pyg_data, "x") and pyg_data.x is not None:
            tensordict["features"] = pyg_data.x

        return tensordict

    def _tensordict_to_pyg(self, tensordict: Dict[str, Any], x: torch.Tensor) -> Data:
        """Convert TensorDict back to PyG format."""
        # Create minimal PyG data with node features and edge information
        # This is a simplified version - in practice you'd need to preserve edge_index
        return Data(x=x)

    def _data_to_tensordict(self, data: Union[Data, Batch]) -> TensorDict:
        """Convert PyG Data to TensorDict."""

        td_dict = {}

        # Basic graph structure
        if hasattr(data, "x") and data.x is not None:
            td_dict["x"] = data.x
        if hasattr(data, "edge_index") and data.edge_index is not None:
            td_dict["edge_index"] = data.edge_index
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            td_dict["edge_attr"] = data.edge_attr
        if hasattr(data, "batch") and data.batch is not None:
            td_dict["batch"] = data.batch
            batch_size = data.batch.max().item() + 1
        else:
            batch_size = 1

        # Labels
        if hasattr(data, "y") and data.y is not None:
            td_dict["y"] = data.y

        # Multi-modal features
        if hasattr(data, "coordinates") and data.coordinates is not None:
            td_dict["coordinates"] = data.coordinates
        if hasattr(data, "magnitudes") and data.magnitudes is not None:
            td_dict["magnitudes"] = data.magnitudes
        if hasattr(data, "wavelengths") and data.wavelengths is not None:
            td_dict["wavelengths"] = data.wavelengths
        if hasattr(data, "flux") and data.flux is not None:
            td_dict["flux"] = data.flux

        return TensorDict(td_dict, batch_size=[batch_size])

    def get_embeddings(
        self, batch: Union[Data, HeteroData, Batch, TensorDict]
    ) -> Tensor:
        """Get graph embeddings without final classification layer."""

        # Convert to TensorDict if needed and requested
        if self.use_tensordict and not isinstance(batch, TensorDict):
            # Temporarily disable TensorDict to avoid batch size issues
            # batch = self._data_to_tensordict(batch)
            pass

        # Apply encoders if using TensorDict
        if isinstance(batch, TensorDict) and self.encoders:
            # Apply each encoder
            for encoder in self.encoders:
                batch = encoder(batch)

            # Apply fusion if needed
            if self.fusion is not None:
                batch = self.fusion(batch)

            # Extract features
            x = batch[self.feature_key]
            edge_index = batch["edge_index"]
            edge_attr = batch.get("edge_attr", None)
            batch_idx = batch.get("batch", None)
        else:
            # Traditional PyG data handling
            x = getattr(batch, "x", None)
            edge_index = getattr(batch, "edge_index", None)
            if x is None or edge_index is None:
                raise ValueError("Batch must have 'x' and 'edge_index'!")
            edge_attr = getattr(batch, "edge_attr", None)
            batch_idx = getattr(batch, "batch", None)

            # Input projection if not using encoders
            if hasattr(self, "input_projection"):
                x = self.input_projection(x)
                x = F.gelu(x)

        # Graph convolution layers or point cloud encoder
        if self.point_cloud_encoder is not None:
            # Extract positions for point cloud processing
            pos = getattr(batch, "pos", None) if hasattr(batch, "pos") else None
            if pos is None and hasattr(batch, "x"):
                # Try to get positions from x (first 3 dimensions)
                if batch.x.shape[1] >= 3:
                    pos = batch.x[:, :3]
                else:
                    raise ValueError("Point cloud layers require 3D positions")
            
            # Apply point cloud encoder
            x, _ = self.point_cloud_encoder(x, pos, batch_idx)
        else:
            # Traditional graph convolution
            for i, layer_dict in enumerate(self.graph_layers):
                x = layer_dict["conv"](x, edge_index, edge_attr=edge_attr)
                if "norm" in layer_dict:
                    norm_layer = layer_dict["norm"]
                    if isinstance(norm_layer, nn.LayerNorm):
                        x = norm_layer(x)
                    else:
                        x = norm_layer(x, batch=batch_idx)
                x = layer_dict["act"](x)
                x = layer_dict["dropout"](x)

        # Pooling
        if batch_idx is not None and x.dim() == 2:  # Only pool if not already pooled
            if self.pooling_layer is not None:
                x = self.pooling_layer(x, batch=batch_idx)
            else:
                # Simple pooling
                if self.pooling == "mean":
                    x = global_mean_pool(x, batch_idx)
                elif self.pooling == "max":
                    x = global_max_pool(x, batch_idx)
                elif self.pooling == "sum":
                    x = global_add_pool(x, batch_idx)
        elif x.dim() == 2 and x.size(0) > 1:  # Multiple nodes but no batch
            if self.pooling == "mean":
                x = x.mean(dim=0, keepdim=True)
            elif self.pooling == "max":
                x = x.max(dim=0)[0].unsqueeze(0)
            elif self.pooling == "sum":
                x = x.sum(dim=0, keepdim=True)

        return x


def create_astro_graph_gnn(model_type: str = "standard", **kwargs) -> AstroGraphGNN:
    """Factory function for creating AstroGraphGNN variants."""

    if model_type == "cosmic_web":
        kwargs.setdefault("conv_type", "gat")
        kwargs.setdefault("num_layers", 4)
        kwargs.setdefault("hidden_dim", 256)
        kwargs.setdefault("pooling", "statistical")
        kwargs.setdefault("use_spatial", True)
    elif model_type == "galaxy_morphology":
        kwargs.setdefault("conv_type", "gin")
        kwargs.setdefault("pooling", "adaptive")
        kwargs.setdefault("use_photometric", True)
    elif model_type == "transient":
        kwargs.setdefault("conv_type", "transformer")
        kwargs.setdefault("pooling", "attention")
        kwargs.setdefault("use_spectral", True)

    return AstroGraphGNN(**kwargs)
