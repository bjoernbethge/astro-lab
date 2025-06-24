"""Simple base components and mixins for AstroLab models."""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .layers import create_conv_layer


class DeviceMixin:
    """Simple device management mixin."""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
    def to_device(self, *tensors):
        """Move tensors to device."""
        return [t.to(self.device) if t is not None else None for t in tensors]


class GraphProcessor(nn.Module):
    """Simple graph processing component."""
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_layers: int, 
        conv_type: str = 'gcn',
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            create_conv_layer(conv_type, hidden_dim, hidden_dim, **kwargs)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process graph through conv layers."""
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        return h


class FeatureProcessor(nn.Module):
    """Simple feature processing component."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_photometry: bool = True,
        use_astrometry: bool = True,
        use_spectroscopy: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_photometry = use_photometry
        self.use_astrometry = use_astrometry
        self.use_spectroscopy = use_spectroscopy
        
        # We'll create the projection layer dynamically based on actual input
        self._projection = None
        self.norm = nn.LayerNorm(hidden_dim)
        
    def _get_projection(self, actual_dim: int) -> nn.Linear:
        """Get or create projection layer for actual input dimension."""
        if self._projection is None or self._projection.in_features != actual_dim:
            self._projection = nn.Linear(actual_dim, self.hidden_dim)
            # Move to same device as norm layer
            self._projection = self._projection.to(self.norm.weight.device)
        return self._projection
        
    def forward(self, data) -> torch.Tensor:
        """Process features from data object."""
        # Handle direct tensor input
        if isinstance(data, torch.Tensor):
            x = data
        else:
            # Handle data objects with attributes
            features = []
            
            if self.use_photometry and hasattr(data, 'photometry'):
                features.append(data.photometry)
            if self.use_astrometry and hasattr(data, 'astrometry'):
                features.append(data.astrometry)
            if self.use_spectroscopy and hasattr(data, 'spectroscopy'):
                features.append(data.spectroscopy)
                
            # Concatenate all features
            if features:
                x = torch.cat(features, dim=-1)
            else:
                # Fallback to x attribute or data attribute
                if hasattr(data, 'x'):
                    x = data.x
                elif hasattr(data, 'data'):
                    x = data.data
                else:
                    raise ValueError("Cannot extract features from input data")
            
        # Get projection layer for actual input dimension
        projection = self._get_projection(x.size(-1))
        
        # Project to hidden dimension
        x = projection(x)
        x = self.norm(x)
        return F.relu(x)


class PoolingModule(nn.Module):
    """Simple pooling module for graph-level features."""
    
    def __init__(self, pooling_type: str = 'mean'):
        super().__init__()
        self.pooling_type = pooling_type
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool node features to graph level."""
        if batch is None:
            # If no batch, assume single graph
            if self.pooling_type == 'mean':
                return x.mean(dim=0, keepdim=True)
            elif self.pooling_type == 'max':
                return x.max(dim=0)[0].unsqueeze(0)
            elif self.pooling_type == 'sum':
                return x.sum(dim=0, keepdim=True)
        else:
            # Use PyG pooling
            if self.pooling_type == 'mean':
                return global_mean_pool(x, batch)
            elif self.pooling_type == 'max':
                from torch_geometric.nn import global_max_pool
                return global_max_pool(x, batch)
            elif self.pooling_type == 'sum':
                from torch_geometric.nn import global_add_pool
                return global_add_pool(x, batch) 