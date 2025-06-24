"""Simplified AstroPhot GNN for galaxy modeling."""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import (
    GraphProcessor,
    PoolingModule,
    create_mlp,
    create_output_head,
)


class AstroPhotGNN(nn.Module):
    """Simplified GNN with AstroPhot integration for galaxy modeling."""
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        output_dim: int = 12,  # Typical Sersic + disk parameters
        num_layers: int = 3,
        model_components: List[str] = None,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        super().__init__()
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if model_components is None:
            model_components = ["sersic", "disk"]
        self.model_components = model_components
        
        # Default input dimension for galaxy features
        if input_dim is None:
            input_dim = kwargs.get('galaxy_features_dim', 20)
            
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph processor
        self.graph_processor = GraphProcessor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=kwargs.get('conv_type', 'gcn'),
            dropout=dropout,
            **kwargs
        )
        
        # Pooling
        self.pooling = PoolingModule(kwargs.get('pooling', 'mean'))
        
        # Component-specific heads
        self.component_heads = nn.ModuleDict()
        for component in model_components:
            if component == "sersic":
                # Sersic parameters: [Re, n, I_e, PA]
                self.component_heads[component] = create_mlp(
                    hidden_dim, 4, [hidden_dim // 2], dropout=dropout
                )
            elif component == "disk":
                # Disk parameters: [Rd, I0, PA]
                self.component_heads[component] = create_mlp(
                    hidden_dim, 3, [hidden_dim // 2], dropout=dropout
                )
            elif component == "bulge":
                # Bulge parameters: [Rb, Ib, q]
                self.component_heads[component] = create_mlp(
                    hidden_dim, 3, [hidden_dim // 2], dropout=dropout
                )
                
        # Global galaxy parameters
        self.global_head = create_mlp(
            hidden_dim, output_dim, [hidden_dim, hidden_dim // 2], dropout=dropout
        )
        
        self.to(self.device)
        
    def apply_parameter_constraints(self, params: torch.Tensor, component: str) -> torch.Tensor:
        """Apply physical constraints to component parameters."""
        if component == "sersic":
            # [Re, n, I_e, PA]
            re = F.softplus(params[..., 0:1])  # Effective radius > 0
            n = torch.clamp(params[..., 1:2], 0.1, 8.0)  # Sersic index
            ie = F.softplus(params[..., 2:3])  # Surface brightness > 0
            pa = torch.remainder(params[..., 3:4], 180.0)  # Position angle [0, 180)
            return torch.cat([re, n, ie, pa], dim=-1)
            
        elif component == "disk":
            # [Rd, I0, PA]
            rd = F.softplus(params[..., 0:1])  # Scale radius > 0
            i0 = F.softplus(params[..., 1:2])  # Central surface brightness > 0
            pa = torch.remainder(params[..., 2:3], 180.0)  # Position angle
            return torch.cat([rd, i0, pa], dim=-1)
            
        elif component == "bulge":
            # [Rb, Ib, q]
            rb = F.softplus(params[..., 0:1])  # Bulge radius > 0
            ib = F.softplus(params[..., 1:2])  # Bulge intensity > 0
            q = torch.sigmoid(params[..., 2:3])  # Axis ratio [0, 1]
            return torch.cat([rb, ib, q], dim=-1)
            
        return params
        
    def forward(
        self,
        data,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for galaxy parameter prediction."""
        
        # Handle different input formats
        if hasattr(data, 'x'):
            x = data.x
            if edge_index is None and hasattr(data, 'edge_index'):
                edge_index = data.edge_index
            if batch is None and hasattr(data, 'batch'):
                batch = data.batch
        elif isinstance(data, torch.Tensor):
            x = data
        else:
            # Try to extract features
            x = data
            
        # Move to device
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        if edge_index is not None:
            edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)
            
        # Project input
        h = self.input_projection(x)
        
        # Process through graph layers
        if edge_index is not None:
            h = self.graph_processor(h, edge_index)
        
        # Pool to galaxy level
        h = self.pooling(h, batch)
        
        if return_components:
            # Return component-specific predictions
            results = {}
            for component, head in self.component_heads.items():
                params = head(h)
                results[component] = self.apply_parameter_constraints(params, component)
            results["global"] = self.global_head(h)
            return results
        else:
            # Return global parameters only
            return self.global_head(h) 