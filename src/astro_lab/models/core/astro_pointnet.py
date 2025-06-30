"""
AstroLab PointNet with TensorDict Integration
============================================

PointNet architecture optimized for astronomical point cloud analysis
with full TensorDict support and multi-modal capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool, fps
from typing import Optional, Dict, Any, Union, List
from torch_geometric.data import Data, Batch

from .base_model import AstroBaseModel


class PointNetTransform(nn.Module):
    """
    Spatial transformation network (T-Net) for PointNet.
    
    Learns an optimal transformation matrix for input coordinates
    to achieve rotation/translation invariance.
    """
    
    def __init__(self, k: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.k = k
        
        # Feature extraction
        self.conv1 = nn.Conv1d(k, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, k * k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates [B, k, N] or [N, k]
        
        Returns:
            Transformation matrix [B, k, k] or [k, k]
        """
        # Handle both batched and unbatched input
        if x.dim() == 2:
            x = x.t().unsqueeze(0)  # [N, k] -> [1, k, N]
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(batch_size, self.k, self.k)
        
        if squeeze_output:
            x = x.squeeze(0)
            
        return x


class PointNetFeatureExtractor(nn.Module):
    """
    Standard PointNet feature extraction.
    
    Extracts local and global features from point clouds.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_tnet: bool = True,
        use_feature_tnet: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_tnet = use_tnet
        self.use_feature_tnet = use_feature_tnet
        
        # Spatial transform
        if use_tnet:
            self.stn = PointNetTransform(k=3, hidden_dim=64)
            
        # Feature transform (optional)
        if use_feature_tnet:
            self.fstn = PointNetTransform(k=64, hidden_dim=64)
            
        # Shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
    def forward(
        self, 
        coords: torch.Tensor, 
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from point cloud.
        
        Args:
            coords: [N, 3] or [B, N, 3] coordinates
            features: Optional additional features (unused in standard PointNet)
            
        Returns:
            Dictionary with point_features and spatial_transform
        """
        # Handle batched vs unbatched input
        if coords.dim() == 2:
            # [N, 3] -> [1, 3, N]
            coords = coords.t().unsqueeze(0)
            unbatched = True
        else:
            # [B, N, 3] -> [B, 3, N]
            coords = coords.transpose(1, 2)
            unbatched = False
            
        batch_size = coords.size(0)
        n_points = coords.size(2)
        
        # Apply spatial transform
        if self.use_tnet:
            trans = self.stn(coords)
            coords = torch.bmm(trans, coords)
        else:
            trans = None
            
        # First layer
        x = F.relu(self.bn1(self.conv1(coords)))
        
        # Feature transform (optional)
        if self.use_feature_tnet:
            trans_feat = self.fstn(x)
            x = torch.bmm(trans_feat, x)
        else:
            trans_feat = None
            
        # Continue processing
        point_features = F.relu(self.bn2(self.conv2(x)))
        point_features = self.bn3(self.conv3(point_features))
        
        # Transpose back to [B, N, F]
        point_features = point_features.transpose(1, 2)
        
        if unbatched:
            point_features = point_features.squeeze(0)
            
        return {
            "point_features": point_features,
            "spatial_transform": trans,
            "feature_transform": trans_feat,
        }


class ScalablePointNetFeatureExtractor(nn.Module):
    """
    Scalable PointNet feature extraction for 50M+ points.
    
    Features:
    - Hierarchical processing with spatial pooling
    - Memory-efficient operations
    - Multi-scale feature extraction
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 128, 256],
        use_tnet: bool = True,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_tnet = use_tnet
        
        # Spatial transform
        if use_tnet:
            self.stn = PointNetTransform(k=3, hidden_dim=64)
            
        # Multi-scale feature extraction
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        
        in_channels = 3
        for hidden_dim in hidden_dims:
            self.conv_layers.append(nn.Conv1d(in_channels, hidden_dim, 1))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            in_channels = hidden_dim
            
        # Final projection
        self.final_conv = nn.Conv1d(hidden_dims[-1], output_dim, 1)
        if use_batch_norm:
            self.final_bn = nn.BatchNorm1d(output_dim)
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        coords: torch.Tensor, 
        features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features with optional batching.
        
        Args:
            coords: [N, 3] or [B, N, 3] coordinates
            features: Optional [N, F] or [B, N, F] features
            batch: Optional [N] batch indices for minibatching
            
        Returns:
            Dictionary with extracted features
        """
        # Handle batched vs unbatched input
        if coords.dim() == 2:
            # [N, 3] -> [1, 3, N]
            coords_t = coords.t().unsqueeze(0)
            unbatched = True
        else:
            # [B, N, 3] -> [B, 3, N]
            coords_t = coords.transpose(1, 2)
            unbatched = False
            
        # Apply spatial transform
        if self.use_tnet:
            trans = self.stn(coords_t)
            coords_t = torch.bmm(trans, coords_t)
        else:
            trans = None
            
        # Multi-scale feature extraction
        x = coords_t
        intermediate_features = []
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.bn_layers is not None:
                x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            intermediate_features.append(x)
            
        # Final features
        x = self.final_conv(x)
        if hasattr(self, 'final_bn'):
            x = self.final_bn(x)
            
        # Transpose back
        point_features = x.transpose(1, 2)  # [B, N, F] or [1, N, F]
        
        if unbatched:
            point_features = point_features.squeeze(0)
            
        return {
            "point_features": point_features,
            "intermediate_features": intermediate_features,
            "spatial_transform": trans,
        }


class AstroPointNet(AstroBaseModel):
    """
    PointNet for astronomical point cloud analysis.
    
    Features:
    - Full TensorDict integration
    - Multi-modal support (spatial, photometric, spectral)
    - Flexible pooling strategies
    - Task-specific heads (classification, regression, segmentation)
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        feature_dim: int = 1024,
        use_tnet: bool = True,
        use_feature_tnet: bool = False,
        pooling: str = "max",
        task: str = "node_classification",
        dropout: float = 0.3,
        # Multi-modal options
        use_spatial: bool = True,
        use_photometric: bool = False,
        use_spectral: bool = False,
        # TensorDict options
        in_keys: Optional[list] = None,
        out_keys: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            **kwargs,
        )
        
        self.feature_dim = feature_dim
        self.use_tnet = use_tnet
        self.use_feature_tnet = use_feature_tnet
        self.pooling = pooling
        self.dropout = dropout
        
        # Multi-modal configuration
        self.use_spatial = use_spatial
        self.use_photometric = use_photometric
        self.use_spectral = use_spectral
        
        # TensorDict keys
        self.in_keys = in_keys or ["features", "pos"]
        self.out_keys = out_keys or ["logits"]
        
        # Build architecture
        self._build_model()
        
    def _build_model(self):
        """Build the PointNet architecture."""
        
        # Feature extractor
        self.feature_extractor = PointNetFeatureExtractor(
            input_dim=3,  # Spatial coordinates
            output_dim=self.feature_dim,
            use_tnet=self.use_tnet,
            use_feature_tnet=self.use_feature_tnet,
        )
        
        # Additional feature processing
        if self.use_photometric or self.use_spectral or self.num_features > 3:
            # Combine spatial features with other modalities
            self.feature_fusion = nn.Sequential(
                nn.Linear(self.feature_dim + self.num_features, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
            )
        else:
            self.feature_fusion = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
            )
        
        # Task-specific heads
        if self.task in ["node_classification", "node_regression"]:
            # Per-point prediction
            self.output_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_classes),
            )
        else:
            # Graph-level prediction (needs pooling)
            self.output_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_classes),
            )
            
        # Build TensorDict modules
        self._build_tensordict_modules()
        
    def _build_tensordict_modules(self):
        """Build TensorDict-compatible modules."""
        
        # Feature extraction module
        self.td_feature_extractor = TensorDictModule(
            lambda pos, features=None: self._extract_features(pos, features),
            in_keys=["pos", "features"] if self.num_features > 3 else ["pos"],
            out_keys=["point_features", "spatial_transform"],
        )
        
        # Fusion module
        self.td_fusion = TensorDictModule(
            lambda point_features, features=None: self._fuse_features(point_features, features),
            in_keys=["point_features", "features"] if self.num_features > 3 else ["point_features"],
            out_keys=["fused_features"],
        )
        
        # Pooling module (for graph tasks)
        if self.task in ["graph_classification", "graph_regression"]:
            self.td_pooling = TensorDictModule(
                lambda features, batch: self._pool_features(features, batch),
                in_keys=["fused_features", "batch"],
                out_keys=["pooled_features"],
            )
        
        # Output module
        output_in_key = "pooled_features" if hasattr(self, "td_pooling") else "fused_features"
        self.td_output = TensorDictModule(
            self.output_head,
            in_keys=[output_in_key],
            out_keys=["logits"],
        )
        
    def _extract_features(self, pos: torch.Tensor, features: Optional[torch.Tensor] = None):
        """Extract features using PointNet."""
        result = self.feature_extractor(pos, features)
        return result["point_features"], result.get("spatial_transform", torch.eye(3))
        
    def _fuse_features(self, point_features: torch.Tensor, features: Optional[torch.Tensor] = None):
        """Fuse spatial features with other modalities."""
        if features is not None and features.size(-1) > 3:
            # Concatenate all features
            combined = torch.cat([point_features, features], dim=-1)
        else:
            combined = point_features
            
        # Apply fusion network
        # Handle both batched and unbatched input
        if combined.dim() == 2:
            return self.feature_fusion(combined)
        else:
            batch_size, n_points, feat_dim = combined.shape
            combined = combined.view(-1, feat_dim)
            fused = self.feature_fusion(combined)
            return fused.view(batch_size, n_points, -1)
            
    def _pool_features(self, features: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Pool features for graph-level tasks."""
        if batch is None:
            # Single graph - pool over all points
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
        Forward pass through PointNet.
        
        Args:
            batch: Input data (PyG Data/Batch or TensorDict)
            
        Returns:
            Predictions tensor
        """
        # Convert to TensorDict if needed
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        elif isinstance(batch, dict) and not isinstance(batch, TensorDict):
            td = TensorDict(batch, batch_size=[])
        else:
            td = batch
            
        # Extract features
        td = self.td_feature_extractor(td)
        
        # Fuse features
        td = self.td_fusion(td)
        
        # Pool if needed
        if hasattr(self, "td_pooling"):
            td = self.td_pooling(td)
            
        # Generate output
        td = self.td_output(td)
        
        return td["logits"]
        
    def _pyg_to_tensordict(self, pyg_data: Union[Data, Batch]) -> TensorDict:
        """Convert PyG data to TensorDict format."""
        td_dict = {}
        
        # Positions (required)
        if hasattr(pyg_data, "pos") and pyg_data.pos is not None:
            td_dict["pos"] = pyg_data.pos
        elif hasattr(pyg_data, "x") and pyg_data.x is not None and pyg_data.x.size(-1) >= 3:
            td_dict["pos"] = pyg_data.x[:, :3]
        else:
            raise ValueError("No position data found in input")
            
        # Features
        if hasattr(pyg_data, "x") and pyg_data.x is not None:
            td_dict["features"] = pyg_data.x
            
        # Batch indices (for batched graphs)
        if hasattr(pyg_data, "batch") and pyg_data.batch is not None:
            td_dict["batch"] = pyg_data.batch
            
        # Labels
        if hasattr(pyg_data, "y") and pyg_data.y is not None:
            td_dict["y"] = pyg_data.y
            
        # Edge information (if needed)
        if hasattr(pyg_data, "edge_index") and pyg_data.edge_index is not None:
            td_dict["edge_index"] = pyg_data.edge_index
            
        return TensorDict(td_dict, batch_size=[])
        
    def get_embeddings(self, batch: Union[Data, Batch, TensorDict]) -> torch.Tensor:
        """Get point embeddings without classification head."""
        # Convert to TensorDict
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        else:
            td = batch
            
        # Extract and fuse features
        td = self.td_feature_extractor(td)
        td = self.td_fusion(td)
        
        return td["fused_features"]
        
    def analyze_critical_points(self, batch: Union[Data, Batch, TensorDict]) -> Dict[str, torch.Tensor]:
        """
        Analyze critical points for the prediction.
        
        Returns:
            Dictionary with:
                - critical_indices: Indices of most important points
                - point_importance: Importance score for each point
                - spatial_transform: Applied spatial transformation
        """
        # Get embeddings
        embeddings = self.get_embeddings(batch)
        
        # Calculate importance as L2 norm of embeddings
        point_importance = torch.norm(embeddings, p=2, dim=-1)
        
        # Get top-k critical points
        k = min(10, point_importance.size(0))
        values, indices = torch.topk(point_importance, k)
        
        # Get spatial transform if available
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        else:
            td = batch
            
        td = self.td_feature_extractor(td)
        spatial_transform = td.get("spatial_transform", torch.eye(3))
        
        return {
            "critical_indices": indices,
            "critical_values": values,
            "point_importance": point_importance,
            "spatial_transform": spatial_transform,
        }
        
    def visualize_critical_regions(self, batch: Union[Data, Batch, TensorDict]) -> Dict[str, Any]:
        """
        Get visualization data for critical regions.
        
        Returns:
            Dictionary with visualization-ready data
        """
        analysis = self.analyze_critical_points(batch)
        
        # Get positions
        if isinstance(batch, (Data, Batch)):
            pos = batch.pos if hasattr(batch, "pos") else batch.x[:, :3]
        else:
            pos = batch.get("pos", batch.get("features", torch.zeros(1, 3))[:, :3])
            
        critical_pos = pos[analysis["critical_indices"]]
        
        return {
            "all_positions": pos.detach().cpu().numpy(),
            "critical_positions": critical_pos.detach().cpu().numpy(),
            "importance_scores": analysis["point_importance"].detach().cpu().numpy(),
            "critical_indices": analysis["critical_indices"].detach().cpu().numpy(),
        }


class ScalableAstroPointNet(AstroPointNet):
    """
    Scalable AstroPointNet for 50M+ astronomical objects.
    
    Features:
    - Hierarchical point sampling and pooling
    - Memory-efficient batch processing
    - Multi-GPU support through Lightning
    - Gradient checkpointing for memory savings
    - Dynamic batching based on available memory
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        feature_dim: int = 1024,
        num_samples: List[int] = [10000, 5000, 1000],  # Hierarchical sampling
        sampling_method: str = "fps",  # or "random"
        use_gradient_checkpointing: bool = True,
        max_points_per_batch: int = 100000,
        **kwargs,
    ):
        # Initialize parent with modified feature extractor
        super().__init__(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            **kwargs,
        )
        
        self.num_samples = num_samples
        self.sampling_method = sampling_method
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_points_per_batch = max_points_per_batch
        
        # Replace feature extractor with scalable version
        self.feature_extractor = ScalablePointNetFeatureExtractor(
            input_dim=3,
            output_dim=feature_dim,
            hidden_dims=[64, 128, 256, 512],
            use_tnet=self.use_tnet,
            dropout=self.dropout,
        )
        
        # Hierarchical pooling modules
        self.pooling_layers = nn.ModuleList()
        for i in range(len(num_samples)):
            self.pooling_layers.append(
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                )
            )
            
        # Multi-scale fusion
        fusion_dim = hidden_dim * len(num_samples)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
    def forward(self, batch: Union[Data, Batch, TensorDict, Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass with hierarchical processing for large point clouds.
        """
        # Convert to TensorDict
        if isinstance(batch, (Data, Batch)):
            td = self._pyg_to_tensordict(batch)
        elif isinstance(batch, dict) and not isinstance(batch, TensorDict):
            td = TensorDict(batch, batch_size=[])
        else:
            td = batch
            
        # Get positions and features
        pos = td["pos"]
        features = td.get("features", None)
        batch_idx = td.get("batch", None)
        
        # Check if we need to process in chunks
        n_points = pos.size(0)
        if n_points > self.max_points_per_batch:
            return self._forward_chunked(td)
            
        # Hierarchical sampling and feature extraction
        multi_scale_features = []
        
        for i, n_samples in enumerate(self.num_samples):
            # Sample points
            if n_points > n_samples:
                if self.sampling_method == "fps":
                    # Farthest point sampling
                    sample_idx = fps(pos, ratio=n_samples/n_points, batch=batch_idx)
                else:
                    # Random sampling
                    perm = torch.randperm(n_points, device=pos.device)
                    sample_idx = perm[:n_samples]
                    
                sampled_pos = pos[sample_idx]
                sampled_features = features[sample_idx] if features is not None else None
                sampled_batch = batch_idx[sample_idx] if batch_idx is not None else None
            else:
                sampled_pos = pos
                sampled_features = features
                sampled_batch = batch_idx
                
            # Extract features at this scale
            if self.use_gradient_checkpointing and self.training:
                feat_dict = torch.utils.checkpoint.checkpoint(
                    self.feature_extractor.forward,
                    sampled_pos,
                    sampled_features,
                    sampled_batch,
                )
            else:
                feat_dict = self.feature_extractor(sampled_pos, sampled_features, sampled_batch)
                
            scale_features = feat_dict["point_features"]
            
            # Pool features
            if sampled_batch is not None:
                if self.pooling == "max":
                    pooled = global_max_pool(scale_features, sampled_batch)
                elif self.pooling == "mean":
                    pooled = global_mean_pool(scale_features, sampled_batch)
                else:
                    pooled = global_add_pool(scale_features, sampled_batch)
            else:
                # Single graph
                if self.pooling == "max":
                    pooled = scale_features.max(dim=0, keepdim=True)[0]
                elif self.pooling == "mean":
                    pooled = scale_features.mean(dim=0, keepdim=True)
                else:
                    pooled = scale_features.sum(dim=0, keepdim=True)
                    
            # Process pooled features
            pooled = self.pooling_layers[i](pooled)
            multi_scale_features.append(pooled)
            
        # Fuse multi-scale features
        fused = torch.cat(multi_scale_features, dim=-1)
        fused = self.fusion_layer(fused)
        
        # Final classification
        return self.output_head(fused)
        
    def _forward_chunked(self, td: TensorDict) -> torch.Tensor:
        """
        Process very large point clouds in chunks.
        """
        pos = td["pos"]
        features = td.get("features", None)
        batch_idx = td.get("batch", None)
        
        n_points = pos.size(0)
        n_chunks = (n_points + self.max_points_per_batch - 1) // self.max_points_per_batch
        
        # Process each chunk
        chunk_outputs = []
        
        for i in range(n_chunks):
            start_idx = i * self.max_points_per_batch
            end_idx = min((i + 1) * self.max_points_per_batch, n_points)
            
            # Create chunk TensorDict
            chunk_td = TensorDict({
                "pos": pos[start_idx:end_idx],
                "features": features[start_idx:end_idx] if features is not None else None,
                "batch": batch_idx[start_idx:end_idx] if batch_idx is not None else None,
            }, batch_size=[])
            
            # Process chunk
            chunk_output = self.forward(chunk_td)
            chunk_outputs.append(chunk_output)
            
        # Aggregate chunk outputs
        if self.task in ["graph_classification", "graph_regression"]:
            # For graph tasks, average the outputs
            return torch.stack(chunk_outputs).mean(dim=0)
        else:
            # For node tasks, concatenate
            return torch.cat(chunk_outputs, dim=0)
            
    def configure_sharded_model(self):
        """Configure model for multi-GPU training with FSDP."""
        # This is called by Lightning for model parallelism
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        
        # Wrap layers larger than 10M parameters
        return size_based_auto_wrap_policy(min_num_params=10_000_000)
        

def create_scalable_astro_pointnet(
    num_features: int,
    num_classes: int,
    task: str = "cosmic_web_classification",
    num_objects: int = 50_000_000,
    **kwargs
) -> ScalableAstroPointNet:
    """
    Factory function for creating scalable AstroPointNet models.
    
    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type
        num_objects: Expected number of objects (for optimization)
        **kwargs: Additional arguments
        
    Returns:
        Configured ScalableAstroPointNet model
    """
    # Adjust default parameters based on scale
    if num_objects > 10_000_000:
        # Very large scale
        kwargs.setdefault("num_samples", [100000, 50000, 10000])
        kwargs.setdefault("use_gradient_checkpointing", True)
        kwargs.setdefault("max_points_per_batch", 500000)
        kwargs.setdefault("feature_dim", 512)  # Smaller features for memory
    elif num_objects > 1_000_000:
        # Large scale
        kwargs.setdefault("num_samples", [50000, 10000, 5000])
        kwargs.setdefault("use_gradient_checkpointing", True)
        kwargs.setdefault("max_points_per_batch", 1000000)
        kwargs.setdefault("feature_dim", 1024)
    else:
        # Medium scale
        kwargs.setdefault("num_samples", [10000, 5000, 1000])
        kwargs.setdefault("use_gradient_checkpointing", False)
        kwargs.setdefault("max_points_per_batch", 2000000)
        kwargs.setdefault("feature_dim", 1024)
        
    # Task-specific configurations
    if task == "cosmic_web_classification":
        kwargs.setdefault("num_classes", 4)  # void, sheet, filament, node
        kwargs.setdefault("pooling", "mean")
        kwargs.setdefault("sampling_method", "fps")
    elif task == "stellar_classification":
        kwargs.setdefault("use_photometric", True)
        kwargs.setdefault("pooling", "max")
        
    return ScalableAstroPointNet(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs
    )
