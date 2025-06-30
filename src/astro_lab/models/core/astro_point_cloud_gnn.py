"""
Advanced Point Cloud GNN for Astronomical Data
==============================================

Flexible architecture supporting all modern PyG point cloud operators
optimized for cosmic web analysis with 50M+ objects.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ..components.layers.point_cloud import (
    AdaptivePointCloudLayer,
    HierarchicalPointCloudProcessor,
    MultiScalePointCloudEncoder,
    create_point_cloud_encoder,
)
from .base_model import AstroBaseModel


class AstroPointCloudGNN(AstroBaseModel):
    """
    Advanced Point Cloud GNN supporting multiple operators.
    
    Features:
    - All modern PyG point cloud operators
    - Adaptive layer selection based on data
    - Hierarchical processing for 50M+ objects
    - TensorDict integration
    - Multi-modal astronomical features
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        encoder_type: str = "multi_scale",
        layer_types: Optional[List[str]] = None,
        use_adaptive: bool = True,
        num_scales: int = 3,
        k_neighbors: List[int] = [20, 40, 80],
        pooling: str = "mean",
        dropout: float = 0.1,
        task: str = "cosmic_web_classification",
        # Astronomical modalities
        use_photometric: bool = True,
        use_spectral: bool = False,
        use_temporal: bool = False,
        # Large-scale options
        max_objects_per_batch: int = 100_000,
        use_hierarchical: bool = True,
        downsample_ratios: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize Advanced Point Cloud GNN.
        
        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            encoder_type: Type of encoder ("multi_scale", "adaptive", "hierarchical")
            layer_types: List of layer types to use
            use_adaptive: Use adaptive layer selection
            num_scales: Number of processing scales
            k_neighbors: Number of neighbors at each scale
            pooling: Pooling method
            dropout: Dropout rate
            task: Task type
            use_photometric: Use photometric features
            use_spectral: Use spectral features
            use_temporal: Use temporal features
            max_objects_per_batch: Maximum objects per batch
            use_hierarchical: Use hierarchical processing for large data
            downsample_ratios: Downsampling ratios for hierarchical processing
        """
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            **kwargs,
        )
        
        self.encoder_type = encoder_type
        self.use_adaptive = use_adaptive
        self.pooling = pooling
        self.dropout = dropout
        self.max_objects_per_batch = max_objects_per_batch
        self.use_hierarchical = use_hierarchical
        
        # Default layer types - mix of modern operators
        if layer_types is None:
            layer_types = ["dynamic_edge", "point_transformer", "gravnet"]
        self.layer_types = layer_types
        
        # Modality flags
        self.use_photometric = use_photometric
        self.use_spectral = use_spectral
        self.use_temporal = use_temporal
        
        # Build architecture
        self._build_model(
            num_scales=num_scales,
            k_neighbors=k_neighbors,
            downsample_ratios=downsample_ratios or [0.5, 0.25, 0.1],
        )
        
    def _build_model(
        self,
        num_scales: int,
        k_neighbors: List[int],
        downsample_ratios: List[float],
    ):
        """Build the model architecture."""
        
        # Input projection for multi-modal features
        input_dim = 3  # Base spatial coordinates
        if self.use_photometric:
            input_dim += 5  # Magnitudes in different bands
        if self.use_spectral:
            input_dim += 10  # Spectral features
        if self.use_temporal:
            input_dim += 5  # Temporal features
            
        self.input_projection = nn.Sequential(
            nn.Linear(self.num_features, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        
        # Main encoder
        encoder_kwargs = {
            "layer_types": self.layer_types[:num_scales],
            "k_values": k_neighbors,
            "dropout": self.dropout,
            "pooling": None,  # We'll handle pooling separately
        }
        
        if self.encoder_type == "hierarchical" or self.use_hierarchical:
            encoder_kwargs["downsample_ratios"] = downsample_ratios
            
        self.encoder = create_point_cloud_encoder(
            input_dim=input_dim,
            output_dim=self.hidden_dim,
            encoder_type=self.encoder_type,
            **encoder_kwargs,
        )
        
        # Adaptive layer (optional)
        if self.use_adaptive:
            self.adaptive_layer = AdaptivePointCloudLayer(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                candidate_layers=self.layer_types,
                use_gating=True,
            )
        
        # Task-specific heads
        self._build_task_head()
        
        # TensorDict modules for seamless integration
        self._build_tensordict_modules()
        
    def _build_task_head(self):
        """Build task-specific output head."""
        
        if "classification" in self.task:
            self.task_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_classes),
            )
        elif "regression" in self.task:
            self.task_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_classes),
            )
        elif self.task == "cosmic_web_classification":
            # Specialized head for cosmic web structures
            self.task_head = nn.Sequential(
                nn.Linear(self.hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 4),  # void, sheet, filament, node
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
    def _build_tensordict_modules(self):
        """Build TensorDict-compatible modules."""
        
        # Feature extraction
        self.td_encoder = TensorDictModule(
            lambda pos, features: self._encode_features(pos, features),
            in_keys=["pos", "features"],
            out_keys=["encoded_features", "hierarchical_features"],
        )
        
        # Adaptive processing (optional)
        if self.use_adaptive:
            self.td_adaptive = TensorDictModule(
                lambda features, pos, batch: self.adaptive_layer(features, pos, batch),
                in_keys=["encoded_features", "pos", "batch"],
                out_keys=["adapted_features"],
            )
        
        # Pooling
        self.td_pool = TensorDictModule(
            lambda features, batch: self._pool_features(features, batch),
            in_keys=["encoded_features", "batch"] if not self.use_adaptive 
                   else ["adapted_features", "batch"],
            out_keys=["pooled_features"],
        )
        
        # Task head
        self.td_task = TensorDictModule(
            self.task_head,
            in_keys=["pooled_features"],
            out_keys=["logits"],
        )
        
    def _encode_features(self, pos: torch.Tensor, features: torch.Tensor):
        """Encode features with the point cloud encoder."""
        
        # Project features
        projected = self.input_projection(features)
        
        # Encode
        if isinstance(self.encoder, HierarchicalPointCloudProcessor):
            encoded, hierarchical = self.encoder(projected, pos)
            return encoded, hierarchical
        else:
            encoded, intermediate = self.encoder(projected, pos)
            return encoded, intermediate
            
    def _pool_features(self, features: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Pool features for graph-level tasks."""
        
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
                
    def forward(self, batch: Union[Data, Batch, TensorDict, Dict]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Input data (PyG Data/Batch or TensorDict)
            
        Returns:
            Output predictions
        """
        
        # Convert to TensorDict if needed
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        elif isinstance(batch, dict) and not isinstance(batch, TensorDict):
            td = TensorDict(batch, batch_size=[])
        else:
            td = batch
            
        # Check if we need chunked processing
        n_objects = td["pos"].size(0)
        if n_objects > self.max_objects_per_batch:
            return self._forward_chunked(td)
            
        # Standard forward pass
        td = self.td_encoder(td)
        
        if self.use_adaptive:
            td = self.td_adaptive(td)
            
        # Pool for graph-level tasks
        if "graph" in self.task:
            td = self.td_pool(td)
            
        # Task head
        td = self.td_task(td)
        
        return td["logits"]
        
    def _forward_chunked(self, td: TensorDict) -> torch.Tensor:
        """Process very large point clouds in chunks."""
        
        pos = td["pos"]
        features = td["features"]
        batch_idx = td.get("batch", None)
        
        n_objects = pos.size(0)
        n_chunks = (n_objects + self.max_objects_per_batch - 1) // self.max_objects_per_batch
        
        chunk_outputs = []
        
        for i in range(n_chunks):
            start_idx = i * self.max_objects_per_batch
            end_idx = min((i + 1) * self.max_objects_per_batch, n_objects)
            
            # Create chunk
            chunk_td = TensorDict({
                "pos": pos[start_idx:end_idx],
                "features": features[start_idx:end_idx],
                "batch": batch_idx[start_idx:end_idx] if batch_idx is not None else None,
            }, batch_size=[])
            
            # Process chunk
            chunk_output = self.forward(chunk_td)
            chunk_outputs.append(chunk_output)
            
        # Aggregate outputs
        if "graph" in self.task:
            # Average for graph-level predictions
            return torch.stack(chunk_outputs).mean(dim=0)
        else:
            # Concatenate for node-level predictions
            return torch.cat(chunk_outputs, dim=0)
            
    def _pyg_to_tensordict(self, pyg_data: Union[Data, Batch]) -> TensorDict:
        """Convert PyG data to TensorDict."""
        
        td_dict = {
            "pos": pyg_data.pos if hasattr(pyg_data, "pos") else pyg_data.x[:, :3],
            "features": pyg_data.x,
        }
        
        if hasattr(pyg_data, "batch"):
            td_dict["batch"] = pyg_data.batch
            
        if hasattr(pyg_data, "y"):
            td_dict["y"] = pyg_data.y
            
        if hasattr(pyg_data, "edge_index"):
            td_dict["edge_index"] = pyg_data.edge_index
            
        return TensorDict(td_dict, batch_size=[])
        
    def get_layer_importance(self, batch: Union[Data, Batch, TensorDict]) -> Dict[str, torch.Tensor]:
        """
        Analyze which layers are most important for the prediction.
        
        Useful for understanding which point cloud operators work best
        for different astronomical structures.
        """
        
        if not self.use_adaptive or not hasattr(self, "adaptive_layer"):
            return {}
            
        # Convert to TensorDict
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        else:
            td = batch
            
        # Get features
        td = self.td_encoder(td)
        
        # Get gating weights from adaptive layer
        x = td["encoded_features"]
        pos = td["pos"]
        batch_idx = td.get("batch", None)
        
        # Compute global features for gating
        if batch_idx is not None:
            x_global = global_mean_pool(x, batch_idx)
            pos_global = global_mean_pool(pos, batch_idx)
        else:
            x_global = x.mean(0, keepdim=True)
            pos_global = pos.mean(0, keepdim=True)
            
        gate_input = torch.cat([x_global, pos_global], dim=-1)
        gate_weights = self.adaptive_layer.gate_net(gate_input)
        
        # Create importance dictionary
        importance = {}
        for i, layer_name in enumerate(self.adaptive_layer.layers.keys()):
            importance[layer_name] = gate_weights[:, i].mean().item()
            
        return importance
        
    def analyze_cosmic_web_features(self, batch: Union[Data, Batch, TensorDict]) -> Dict[str, torch.Tensor]:
        """
        Specialized analysis for cosmic web structures.
        
        Returns feature importance and structure predictions.
        """
        
        # Get predictions
        logits = self.forward(batch)
        probs = F.softmax(logits, dim=-1)
        
        # Structure names
        structures = ["void", "sheet", "filament", "node"]
        
        # Get layer importance if available
        layer_importance = self.get_layer_importance(batch)
        
        # Analyze predictions
        pred_classes = torch.argmax(probs, dim=-1)
        
        analysis = {
            "predictions": pred_classes,
            "probabilities": probs,
            "structure_distribution": {
                structures[i]: (pred_classes == i).float().mean().item()
                for i in range(4)
            },
            "confidence": probs.max(dim=-1)[0].mean().item(),
            "layer_importance": layer_importance,
        }
        
        return analysis


def create_astro_point_cloud_gnn(
    num_features: int,
    num_classes: int,
    task: str = "cosmic_web_classification",
    num_objects: int = 1_000_000,
    **kwargs,
) -> AstroPointCloudGNN:
    """
    Factory function for creating AstroPointCloudGNN models.
    
    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type
        num_objects: Expected number of objects
        **kwargs: Additional arguments
        
    Returns:
        Configured AstroPointCloudGNN model
    """
    
    # Optimize based on scale
    if num_objects > 50_000_000:
        # Ultra-large scale
        kwargs.setdefault("encoder_type", "hierarchical")
        kwargs.setdefault("use_hierarchical", True)
        kwargs.setdefault("downsample_ratios", [0.1, 0.05, 0.01])
        kwargs.setdefault("layer_types", ["dynamic_edge", "gravnet", "point_transformer"])
        kwargs.setdefault("max_objects_per_batch", 100_000)
    elif num_objects > 10_000_000:
        # Very large scale
        kwargs.setdefault("encoder_type", "hierarchical")
        kwargs.setdefault("downsample_ratios", [0.2, 0.1, 0.05])
        kwargs.setdefault("layer_types", ["dynamic_edge", "point_transformer", "gravnet"])
        kwargs.setdefault("max_objects_per_batch", 500_000)
    elif num_objects > 1_000_000:
        # Large scale
        kwargs.setdefault("encoder_type", "multi_scale")
        kwargs.setdefault("layer_types", ["dynamic_edge", "point_transformer", "xconv"])
        kwargs.setdefault("max_objects_per_batch", 1_000_000)
    else:
        # Medium scale - can use all features
        kwargs.setdefault("encoder_type", "adaptive")
        kwargs.setdefault("use_adaptive", True)
        kwargs.setdefault("layer_types", ["pointnet", "dynamic_edge", "point_transformer", "gravnet"])
        
    # Task-specific configurations
    if task == "cosmic_web_classification":
        kwargs.setdefault("num_classes", 4)
        kwargs.setdefault("pooling", "mean")
        kwargs.setdefault("use_photometric", True)
    elif task == "stellar_classification":
        kwargs.setdefault("use_photometric", True)
        kwargs.setdefault("use_spectral", True)
        kwargs.setdefault("pooling", "max")
    elif task == "galaxy_morphology":
        kwargs.setdefault("use_photometric", True)
        kwargs.setdefault("layer_types", ["point_transformer", "gravnet", "xconv"])
        
    return AstroPointCloudGNN(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )
