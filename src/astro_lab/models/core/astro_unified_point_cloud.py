"""
Unified Point Cloud GNN for Astronomical Data
============================================

A unified architecture that combines the best of PointNet and modern
point cloud operators, optimized for astronomical data analysis.
"""

from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    fps,
)

from ..components.layers.point_cloud import (
    MultiScalePointCloudEncoder,
    HierarchicalPointCloudProcessor,
    create_point_cloud_encoder,
)
from ..components.encoders import (
    PhotometricEncoder,
    SpectralEncoder,
    SpatialEncoder,
    MultiModalFusion,
)
from ..components.layers.pooling import AdaptivePooling, StatisticalPooling
from ..components.output_heads import create_output_head
from .base_model import AstroBaseModel
from .mixins import (
    VisualizationMixin,
    InterpretabilityMixin,
    EfficientProcessingMixin,
)


class AstroUnifiedPointCloud(
    AstroBaseModel,
    VisualizationMixin,
    InterpretabilityMixin,
    EfficientProcessingMixin,
):
    """
    Unified Point Cloud GNN combining PointNet and modern operators.
    
    Features:
    - Flexible architecture supporting multiple point cloud operators
    - Multi-modal astronomical data integration
    - Hierarchical processing for 50M+ objects
    - TensorDict native support
    - Built-in visualization and interpretability
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        # Architecture options
        architecture: str = "hybrid",  # "pointnet", "modern", "hybrid"
        encoder_type: str = "multi_scale",
        layer_types: Optional[List[str]] = None,
        num_layers: int = 3,
        # Point cloud specific
        k_neighbors: Union[int, List[int]] = 20,
        use_spatial_transform: bool = True,
        # Pooling
        pooling: str = "adaptive",
        pooling_layers: int = 1,
        # Task configuration
        task: str = "graph_classification",
        output_head: Optional[str] = None,
        # Multi-modal options
        use_photometric: bool = False,
        num_photometric_bands: Optional[int] = None,
        use_spectral: bool = False,
        spectral_wavelengths: Optional[int] = None,
        use_spatial: bool = True,
        spatial_dim: int = 3,
        # Efficiency options
        max_points_per_batch: int = 100_000,
        use_hierarchical: bool = False,
        downsample_ratios: Optional[List[float]] = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        """Initialize unified point cloud model."""
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            **kwargs,
        )
        
        self.architecture = architecture
        self.encoder_type = encoder_type
        self.use_spatial_transform = use_spatial_transform
        self.pooling = pooling
        self.pooling_layers = pooling_layers
        self.dropout = dropout
        self.output_head_type = output_head
        
        # Point cloud parameters
        if isinstance(k_neighbors, int):
            self.k_neighbors = [k_neighbors] * num_layers
        else:
            self.k_neighbors = k_neighbors
            
        # Default layer types based on architecture
        if layer_types is None:
            if architecture == "pointnet":
                layer_types = ["pointnet"] * num_layers
            elif architecture == "modern":
                layer_types = ["dynamic_edge", "point_transformer", "gravnet"]
            else:  # hybrid
                layer_types = ["pointnet", "dynamic_edge", "point_transformer"]
        self.layer_types = layer_types
        
        # Multi-modal configuration
        self.use_photometric = use_photometric
        self.use_spectral = use_spectral
        self.use_spatial = use_spatial
        
        # Efficiency options
        self.max_points_per_batch = max_points_per_batch
        self.use_hierarchical = use_hierarchical or num_features > 10_000_000
        self.downsample_ratios = downsample_ratios or [0.5, 0.25, 0.1]
        
        # Build model
        self._build_model(
            num_photometric_bands=num_photometric_bands,
            spectral_wavelengths=spectral_wavelengths,
            spatial_dim=spatial_dim,
        )
        
    def _build_model(
        self,
        num_photometric_bands: Optional[int],
        spectral_wavelengths: Optional[int],
        spatial_dim: int,
    ):
        """Build the unified model architecture."""
        
        # Multi-modal encoders
        self.encoders = nn.ModuleDict()
        self.modality_dims = {}
        
        if self.use_spatial:
            self.encoders["spatial"] = SpatialEncoder(
                spatial_dim=spatial_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                use_fourier_features=True,
            )
            self.modality_dims["spatial"] = self.hidden_dim
            
        if self.use_photometric and num_photometric_bands:
            self.encoders["photometric"] = PhotometricEncoder(
                num_bands=num_photometric_bands,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
            self.modality_dims["photometric"] = self.hidden_dim
            
        if self.use_spectral and spectral_wavelengths:
            self.encoders["spectral"] = SpectralEncoder(
                wavelength_dim=spectral_wavelengths,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
            self.modality_dims["spectral"] = self.hidden_dim
            
        # Multi-modal fusion
        if len(self.modality_dims) > 1:
            self.fusion = MultiModalFusion(
                modality_dims=self.modality_dims,
                fusion_dim=self.hidden_dim,
                fusion_type="attention",
            )
            input_dim = self.hidden_dim
        else:
            self.fusion = None
            input_dim = self.num_features
            
        # Spatial transform (PointNet T-Net)
        if self.use_spatial_transform and self.architecture in ["pointnet", "hybrid"]:
            self.spatial_transform = SpatialTransformNet(
                k=3,
                hidden_dim=64,
            )
        else:
            self.spatial_transform = None
            
        # Main encoder based on configuration
        encoder_config = {
            "input_dim": input_dim,
            "output_dim": self.hidden_dim,
            "dropout": self.dropout,
        }
        
        if self.use_hierarchical:
            encoder_config.update({
                "encoder_type": "hierarchical",
                "downsample_ratios": self.downsample_ratios,
                "layer_types": self.layer_types,
            })
        else:
            encoder_config.update({
                "encoder_type": self.encoder_type,
                "layer_types": self.layer_types,
                "k_values": self.k_neighbors,
            })
            
        self.point_cloud_encoder = create_point_cloud_encoder(**encoder_config)
        
        # Pooling strategy
        if self.pooling == "adaptive":
            self.pooling_layer = AdaptivePooling(
                in_channels=self.hidden_dim,
                pooling_methods=["mean", "max", "attention"],
                attention_heads=4,
            )
            pooled_dim = self.hidden_dim * 3
        elif self.pooling == "statistical":
            self.pooling_layer = StatisticalPooling(
                moments=["mean", "std", "max", "min", "skew", "kurtosis"]
            )
            pooled_dim = self.hidden_dim * 6
        elif self.pooling == "learned":
            self.pooling_layer = LearnedPooling(
                in_channels=self.hidden_dim,
                num_pools=self.pooling_layers,
            )
            pooled_dim = self.hidden_dim * self.pooling_layers
        else:
            self.pooling_layer = None
            pooled_dim = self.hidden_dim
            
        # Output head
        if self.output_head_type:
            self.output_head = create_output_head(
                self.output_head_type,
                input_dim=pooled_dim,
                output_dim=self.num_classes,
                dropout=self.dropout,
            )
        else:
            # Default MLP head
            self.output_head = nn.Sequential(
                nn.Linear(pooled_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_classes),
            )
            
        # Build TensorDict modules
        self._build_tensordict_modules()
        
    def _build_tensordict_modules(self):
        """Build TensorDict-compatible processing modules."""
        
        # Multi-modal encoding
        self.td_encode = TensorDictModule(
            lambda td: self._encode_multimodal(td),
            in_keys=["data"],
            out_keys=["encoded_features", "positions"],
        )
        
        # Point cloud processing
        self.td_process = TensorDictModule(
            lambda features, pos, batch: self._process_point_cloud(features, pos, batch),
            in_keys=["encoded_features", "positions", "batch"],
            out_keys=["processed_features", "hierarchical_features"],
        )
        
        # Pooling (for graph-level tasks)
        if "graph" in self.task:
            self.td_pool = TensorDictModule(
                lambda features, batch: self._pool_features(features, batch),
                in_keys=["processed_features", "batch"],
                out_keys=["pooled_features"],
            )
            
        # Output head
        output_key = "pooled_features" if "graph" in self.task else "processed_features"
        self.td_output = TensorDictModule(
            self.output_head,
            in_keys=[output_key],
            out_keys=["logits"],
        )
        
    def _encode_multimodal(self, data: Dict[str, Any]) -> tuple:
        """Encode multi-modal features."""
        
        encoded_features = []
        positions = None
        
        # Process each modality
        for modality, encoder in self.encoders.items():
            if modality in data:
                features = encoder(data[modality])
                encoded_features.append(features)
                
                # Extract positions from spatial data
                if modality == "spatial" and positions is None:
                    if isinstance(data[modality], dict):
                        positions = data[modality].get("coordinates", data[modality])
                    else:
                        positions = data[modality][:, :3] if data[modality].shape[-1] >= 3 else data[modality]
                        
        # Default positions if not found
        if positions is None:
            positions = data.get("pos", data.get("positions", torch.zeros(1, 3)))
            
        # Fuse features if multiple modalities
        if len(encoded_features) > 1 and self.fusion is not None:
            fused = self.fusion({
                name: feat for name, feat in zip(self.encoders.keys(), encoded_features)
            })
        elif encoded_features:
            fused = encoded_features[0]
        else:
            # Fallback to raw features
            fused = data.get("features", data.get("x", torch.zeros(1, self.num_features)))
            
        return fused, positions
        
    def _process_point_cloud(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Process point cloud with spatial transform and encoder."""
        
        # Apply spatial transform if available
        if self.spatial_transform is not None:
            transform_matrix = self.spatial_transform(positions, batch)
            # Apply transformation
            if batch is not None:
                # Batched transformation
                transformed_pos = []
                for b in torch.unique(batch):
                    mask = batch == b
                    pos_b = positions[mask]
                    trans_b = transform_matrix[b]
                    transformed_pos.append(torch.matmul(pos_b, trans_b.T))
                positions = torch.cat(transformed_pos, dim=0)
            else:
                positions = torch.matmul(positions, transform_matrix.T)
                
        # Process through encoder
        if isinstance(self.point_cloud_encoder, HierarchicalPointCloudProcessor):
            processed, hierarchical = self.point_cloud_encoder(features, positions, batch)
        else:
            processed, hierarchical = self.point_cloud_encoder(features, positions, batch)
            
        return processed, hierarchical
        
    def _pool_features(
        self,
        features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool features for graph-level predictions."""
        
        if self.pooling_layer is not None:
            return self.pooling_layer(features, batch)
        else:
            # Simple pooling
            if batch is None:
                # Single graph
                if self.pooling == "max":
                    return features.max(dim=0, keepdim=True)[0]
                elif self.pooling == "mean":
                    return features.mean(dim=0, keepdim=True)
                else:
                    return features.sum(dim=0, keepdim=True)
            else:
                # Batched graphs
                if self.pooling == "max":
                    return global_max_pool(features, batch)
                elif self.pooling == "mean":
                    return global_mean_pool(features, batch)
                else:
                    return global_add_pool(features, batch)
                    
    def forward(self, batch: Union[Data, Batch, TensorDict, Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass through the unified model.
        
        Handles PyG Data/Batch and TensorDict inputs seamlessly.
        """
        
        # Convert to unified format
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        elif isinstance(batch, dict) and not isinstance(batch, TensorDict):
            td = TensorDict({"data": batch}, batch_size=[])
        elif isinstance(batch, TensorDict):
            td = TensorDict({"data": batch}, batch_size=batch.batch_size)
        else:
            raise ValueError(f"Unsupported input type: {type(batch)}")
            
        # Check if we need chunked processing
        positions = td["data"].get("pos", td["data"].get("positions", None))
        if positions is not None and positions.size(0) > self.max_points_per_batch:
            return self._forward_chunked(td)
            
        # Standard forward pass
        td = self.td_encode(td)
        td = self.td_process(td)
        
        if hasattr(self, "td_pool"):
            td = self.td_pool(td)
            
        td = self.td_output(td)
        
        return td["logits"]
        
    def _pyg_to_tensordict(self, pyg_data: Union[Data, Batch]) -> TensorDict:
        """Convert PyG data to TensorDict format."""
        
        data_dict = {}
        
        # Positions
        if hasattr(pyg_data, "pos"):
            data_dict["pos"] = pyg_data.pos
        elif hasattr(pyg_data, "x") and pyg_data.x.shape[-1] >= 3:
            data_dict["pos"] = pyg_data.x[:, :3]
            
        # Features
        if hasattr(pyg_data, "x"):
            data_dict["features"] = pyg_data.x
            
        # Batch indices
        if hasattr(pyg_data, "batch"):
            data_dict["batch"] = pyg_data.batch
            
        # Labels
        if hasattr(pyg_data, "y"):
            data_dict["y"] = pyg_data.y
            
        # Multi-modal features
        if hasattr(pyg_data, "photometric"):
            data_dict["photometric"] = pyg_data.photometric
        if hasattr(pyg_data, "spectral"):
            data_dict["spectral"] = pyg_data.spectral
            
        return TensorDict({"data": data_dict}, batch_size=[])
        
    def get_critical_points(self, batch: Union[Data, Batch, TensorDict]) -> Dict[str, torch.Tensor]:
        """
        Identify critical points that contribute most to the prediction.
        
        Useful for astronomical interpretation.
        """
        
        # Process input
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        else:
            td = TensorDict({"data": batch}, batch_size=[])
            
        # Get features at each stage
        td = self.td_encode(td)
        encoded_features = td["encoded_features"]
        positions = td["positions"]
        
        td = self.td_process(td)
        processed_features = td["processed_features"]
        
        # Calculate importance scores
        # Using gradient-based importance
        processed_features.requires_grad_(True)
        
        # Get predictions
        if hasattr(self, "td_pool"):
            pooled = self._pool_features(processed_features, td["data"].get("batch"))
            logits = self.output_head(pooled)
        else:
            logits = self.output_head(processed_features)
            
        # Compute gradients
        if logits.dim() > 1:
            # Multi-class: use max logit
            max_logit = logits.max(dim=-1)[0].sum()
        else:
            max_logit = logits.sum()
            
        gradients = torch.autograd.grad(max_logit, processed_features)[0]
        
        # Importance as gradient magnitude
        importance = gradients.norm(dim=-1)
        
        # Get top-k critical points
        k = min(100, importance.size(0))
        values, indices = torch.topk(importance, k)
        
        return {
            "critical_indices": indices,
            "importance_scores": values,
            "critical_positions": positions[indices],
            "all_positions": positions,
            "feature_gradients": gradients,
        }
        
    def visualize_cosmic_web(
        self,
        batch: Union[Data, Batch, TensorDict],
        backend: str = "plotly",
    ) -> Any:
        """
        Visualize cosmic web predictions with critical points.
        
        Leverages the VisualizationMixin capabilities.
        """
        
        # Get predictions and critical points
        with torch.no_grad():
            logits = self.forward(batch)
            predictions = F.softmax(logits, dim=-1)
            
        critical_points = self.get_critical_points(batch)
        
        # Prepare visualization data
        if isinstance(batch, (Data, Batch)):
            positions = batch.pos if hasattr(batch, "pos") else batch.x[:, :3]
        else:
            positions = batch.get("pos", batch.get("positions"))
            
        # Structure names for cosmic web
        structure_names = ["void", "sheet", "filament", "node"]
        pred_classes = predictions.argmax(dim=-1)
        
        # Use mixin visualization with cosmic web specific settings
        viz_data = {
            "positions": positions.cpu().numpy(),
            "predictions": pred_classes.cpu().numpy(),
            "importance": critical_points["importance_scores"].cpu().numpy(),
            "labels": [structure_names[i] for i in pred_classes],
        }
        
        return self.create_visualization(
            viz_data,
            viz_type="3d_scatter",
            backend=backend,
            title="Cosmic Web Structure Predictions",
            color_map={
                0: "blue",    # void
                1: "green",   # sheet
                2: "orange",  # filament
                3: "red",     # node
            }
        )


class SpatialTransformNet(nn.Module):
    """
    Spatial transformation network (T-Net) from PointNet.
    
    Learns optimal rotation/translation for input coordinates.
    """
    
    def __init__(self, k: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.k = k
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(k, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
        )
        
        # Decoder to transformation matrix
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k * k),
        )
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute transformation matrix.
        
        Args:
            x: Positions [N, k] or [B, N, k]
            batch: Batch indices [N]
            
        Returns:
            Transformation matrices [k, k] or [B, k, k]
        """
        
        if x.dim() == 2:
            # Add batch dimension
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Transpose for conv1d
        x = x.transpose(1, 2)  # [B, k, N]
        
        # Encode
        features = self.encoder(x)
        
        # Global max pooling
        features = features.max(dim=2)[0]  # [B, C]
        
        # Decode to transformation
        transform = self.decoder(features)  # [B, k*k]
        transform = transform.view(-1, self.k, self.k)
        
        # Add identity
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype)
        identity = identity.unsqueeze(0).expand_as(transform)
        transform = transform + identity
        
        if squeeze_output:
            transform = transform.squeeze(0)
            
        return transform


class LearnedPooling(nn.Module):
    """Learned pooling layer for point clouds."""
    
    def __init__(self, in_channels: int, num_pools: int = 3):
        super().__init__()
        self.num_pools = num_pools
        
        # Learn pooling weights
        self.pool_weights = nn.Parameter(torch.randn(num_pools, in_channels))
        self.pool_bias = nn.Parameter(torch.zeros(num_pools))
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply learned pooling."""
        
        # Compute attention scores
        scores = torch.matmul(x, self.pool_weights.t()) + self.pool_bias  # [N, num_pools]
        attention = F.softmax(scores, dim=0)
        
        # Weighted pooling
        if batch is None:
            # Single graph
            pooled = torch.matmul(attention.t(), x)  # [num_pools, D]
            return pooled.flatten()
        else:
            # Batched graphs
            pooled_list = []
            for b in torch.unique(batch):
                mask = batch == b
                x_b = x[mask]
                att_b = attention[mask]
                pooled_b = torch.matmul(att_b.t(), x_b)
                pooled_list.append(pooled_b.flatten())
            return torch.stack(pooled_list)


# Factory function
def create_unified_point_cloud_model(
    num_features: int,
    num_classes: int,
    task: str = "cosmic_web_classification",
    scale: str = "medium",  # "small", "medium", "large", "xlarge"
    **kwargs,
) -> AstroUnifiedPointCloud:
    """
    Create a unified point cloud model optimized for the given scale.
    
    Args:
        num_features: Number of input features
        num_classes: Number of output classes  
        task: Task type
        scale: Data scale (affects architecture choices)
        **kwargs: Additional model arguments
        
    Returns:
        Configured AstroUnifiedPointCloud model
    """
    
    # Scale-based defaults
    scale_configs = {
        "small": {  # < 100k objects
            "architecture": "hybrid",
            "encoder_type": "multi_scale",
            "use_hierarchical": False,
            "max_points_per_batch": 100_000,
            "pooling": "adaptive",
        },
        "medium": {  # 100k - 1M objects
            "architecture": "hybrid",
            "encoder_type": "multi_scale",
            "use_hierarchical": False,
            "max_points_per_batch": 500_000,
            "pooling": "statistical",
        },
        "large": {  # 1M - 10M objects
            "architecture": "modern",
            "encoder_type": "hierarchical",
            "use_hierarchical": True,
            "max_points_per_batch": 100_000,
            "downsample_ratios": [0.5, 0.25, 0.1],
            "pooling": "learned",
        },
        "xlarge": {  # 10M+ objects
            "architecture": "modern",
            "encoder_type": "hierarchical", 
            "use_hierarchical": True,
            "max_points_per_batch": 50_000,
            "downsample_ratios": [0.2, 0.1, 0.05],
            "pooling": "mean",  # Simple but efficient
            "layer_types": ["dynamic_edge", "gravnet"],  # Skip expensive layers
        },
    }
    
    # Apply scale defaults
    config = scale_configs.get(scale, scale_configs["medium"])
    config.update(kwargs)
    
    # Task-specific settings
    if task == "cosmic_web_classification":
        config.setdefault("num_classes", 4)
        config.setdefault("output_head", "cosmic_web")
        config.setdefault("use_photometric", True)
    elif task == "stellar_classification":
        config.setdefault("use_photometric", True)
        config.setdefault("use_spectral", True)
        config.setdefault("output_head", "classification")
    elif task == "galaxy_morphology":
        config.setdefault("output_head", "morphology")
        config.setdefault("use_photometric", True)
        
    return AstroUnifiedPointCloud(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **config,
    )
