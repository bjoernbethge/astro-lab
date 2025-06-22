"""
Centralized Layer Factory for AstroLab Models
============================================

Provides unified layer creation for graph neural networks with:
- Consistent interface across all models
- Optimized implementations
- Type safety and validation
- Easy extensibility
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
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

from astro_lab.models.config import ConvType, ActivationType, PoolingType
from astro_lab.models.utils import get_activation, initialize_weights


class LayerFactory:
    """Factory for creating neural network layers with consistent interface."""
    
    @staticmethod
    def create_conv_layer(
        conv_type: ConvType,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ) -> nn.Module:
        """Create a graph convolution layer."""
        # Filter kwargs for torch_geometric Layer
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        
        # Explicitly do not take output_head, concat, normalize, etc. from kwargs
        if conv_type == "gcn":
            return GCNConv(in_channels, out_channels, **filtered_kwargs)
        elif conv_type == "gat":
            head_dim = out_channels // heads
            if head_dim <= 0:
                raise ValueError(f"out_channels ({out_channels}) must be >= heads ({heads})")
            return GATConv(
                in_channels, head_dim, heads=heads, dropout=dropout, **filtered_kwargs
            )
        elif conv_type == "sage":
            return SAGEConv(in_channels, out_channels, **filtered_kwargs)
        elif conv_type == "transformer":
            head_dim = out_channels // heads
            if head_dim <= 0:
                raise ValueError(f"out_channels ({out_channels}) must be >= heads ({heads})")
            return TransformerConv(
                in_channels, head_dim, heads=heads, dropout=dropout, **filtered_kwargs
            )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
    
    @staticmethod
    def create_mlp(
        input_dim: int,
        hidden_dims_or_output_dim: Union[List[int], int],
        output_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,  # For backward compatibility
        activation: ActivationType = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ) -> nn.Module:
        """Create a multi-layer perceptron."""
        
        # Handle backward compatibility with hidden_dims keyword argument
        if hidden_dims is not None:
            hidden_dims_or_output_dim = hidden_dims
            output_dim = None
        
        # Handle case where second parameter is output_dim (backward compatibility)
        if output_dim is None:
            if isinstance(hidden_dims_or_output_dim, int):
                # Single layer: input_dim -> output_dim
                hidden_dims = []
                output_dim = hidden_dims_or_output_dim
            else:
                # hidden_dims provided, need output_dim
                hidden_dims = hidden_dims_or_output_dim
                output_dim = hidden_dims[-1] if hidden_dims else input_dim
        else:
            # Both parameters provided
            if isinstance(hidden_dims_or_output_dim, int):
                hidden_dims = []
            else:
                hidden_dims = hidden_dims_or_output_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_attention_pooling(
        input_dim: int,
        attention_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> nn.Module:
        """Create attention-based pooling layer."""
        
        if attention_dim is None:
            attention_dim = input_dim
        
        return AttentionPooling(input_dim, attention_dim, dropout)
    
    @staticmethod
    def get_pooling_function(pooling: PoolingType):
        """Get pooling function by name."""
        
        pooling_fns = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "add": global_add_pool,
        }
        
        if pooling not in pooling_fns:
            raise ValueError(f"Unknown pooling: {pooling}. Available: {list(pooling_fns.keys())}")
        
        return pooling_fns[pooling]


class AttentionPooling(nn.Module):
    """Attention-based pooling for graph-level tasks."""
    
    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention_net = nn.Sequential(
            Linear(input_dim, attention_dim),
            nn.Tanh(),
            Linear(attention_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention pooling."""
        
        # Compute attention weights
        attention_weights = self.attention_net(x)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        weighted_x = x * attention_weights
        
        # Global pooling
        return global_add_pool(weighted_x, batch)


class ResidualBlock(nn.Module):
    """Residual block for graph neural networks."""
    
    def __init__(
        self,
        conv_layer: nn.Module,
        norm_layer: nn.Module,
        activation: nn.Module,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv = conv_layer
        self.norm = norm_layer
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with residual connection."""
        
        identity = x
        
        # Main path
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection (if dimensions match)
        if out.size(-1) == identity.size(-1):
            out = out + identity
        
        return out


class FeatureFusion(nn.Module):
    """Advanced feature fusion with attention mechanism."""
    
    def __init__(
        self,
        input_dims: List[int],
        output_dim: int,
        fusion_type: Literal["concat", "attention", "weighted"] = "concat",
        dropout: float = 0.1,
    ):
        super().__init__()
        # Always set fusion_type as attribute, default to "concat" if None
        self.fusion_type = fusion_type if fusion_type is not None else "concat"
        self.input_dims = input_dims
        self.output_dim = output_dim
        if self.fusion_type == "concat":
            total_dim = sum(input_dims)
            self.fusion = nn.Sequential(
                Linear(total_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif self.fusion_type == "attention":
            self.attention_weights = nn.Parameter(torch.ones(len(input_dims)))
            self.projection = nn.Linear(max(input_dims), output_dim)
        elif self.fusion_type == "weighted":
            self.weights = nn.Parameter(torch.ones(len(input_dims)))
            self.projection = nn.Linear(max(input_dims), output_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature tensors."""
        
        if len(features) != len(self.input_dims):
            raise ValueError(f"Expected {len(self.input_dims)} features, got {len(features)}")
        
        if self.fusion_type == "concat":
            concatenated = torch.cat(features, dim=-1)
            return self.fusion(concatenated)
        
        elif self.fusion_type == "attention":
            # Normalize attention weights
            attention_weights = F.softmax(self.attention_weights, dim=0)
            
            # Weighted combination
            weighted_features = []
            for feat, weight in zip(features, attention_weights):
                # Pad or truncate to max dimension
                if feat.size(-1) < max(self.input_dims):
                    padding = max(self.input_dims) - feat.size(-1)
                    feat = F.pad(feat, (0, padding))
                elif feat.size(-1) > max(self.input_dims):
                    feat = feat[..., :max(self.input_dims)]
                
                weighted_features.append(feat * weight)
            
            fused = sum(weighted_features)
            return self.projection(fused)
        
        elif self.fusion_type == "weighted":
            # Simple weighted combination
            weighted_features = []
            for feat, weight in zip(features, self.weights):
                # Pad or truncate to max dimension
                if feat.size(-1) < max(self.input_dims):
                    padding = max(self.input_dims) - feat.size(-1)
                    feat = F.pad(feat, (0, padding))
                elif feat.size(-1) > max(self.input_dims):
                    feat = feat[..., :max(self.input_dims)]
                
                weighted_features.append(feat * weight)
            
            fused = sum(weighted_features)
            return self.projection(fused)
        
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")


class LayerRegistry:
    """Registry for custom layer types."""
    
    _layers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register custom layers."""
        
        def decorator(layer_class):
            cls._layers[name] = layer_class
            return layer_class
        
        return decorator
    
    @classmethod
    def create(cls, layer_type: str, **kwargs) -> nn.Module:
        """Create a layer by type."""
        
        if layer_type not in cls._layers:
            available = list(cls._layers.keys())
            raise ValueError(f"Unknown layer type: {layer_type}. Available: {available}")
        
        return cls._layers[layer_type](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available layer types."""
        return list(cls._layers.keys()) 