"""Simplified survey GNN using composition."""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import (
    DeviceMixin,
    FeatureProcessor,
    GraphProcessor,
    PoolingModule,
    create_output_head,
)


class AstroSurveyGNN(nn.Module):
    """Simplified survey GNN using composition."""
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        conv_type: str = 'gcn',
        task: str = 'classification',
        use_photometry: bool = True,
        use_astrometry: bool = True,
        use_spectroscopy: bool = False,
        pooling: str = 'mean',
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Device management
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task = task
        self.pooling = pooling
        
        # Calculate input dimension if not provided
        if input_dim is None:
            input_dim = 0
            if use_photometry:
                input_dim += kwargs.get('photometry_dim', 5)
            if use_astrometry:
                input_dim += kwargs.get('astrometry_dim', 5)
            if use_spectroscopy:
                input_dim += kwargs.get('spectroscopy_dim', 3)
            
            # Default fallback
            if input_dim == 0:
                input_dim = 13  # Default total features
        
        # Use composition instead of complex inheritance
        self.feature_processor = FeatureProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_photometry=use_photometry,
            use_astrometry=use_astrometry,
            use_spectroscopy=use_spectroscopy,
            **kwargs
        )
        
        self.graph_processor = GraphProcessor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout,
            **kwargs
        )
        
        # Pooling for graph-level tasks
        if 'graph' in task:
            self.pooling_module = PoolingModule(pooling)
        else:
            self.pooling_module = None
            
        # Output head based on task
        head_type = task.replace('graph_', '').replace('node_', '')
        self.output_head = create_output_head(
            head_type,
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            **kwargs
        )
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self, 
        data,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Simple, clear forward pass."""
        
        # Handle different input formats
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # PyG Data object
            x = data.x
            if edge_index is None:
                edge_index = data.edge_index
            if batch is None and hasattr(data, 'batch'):
                batch = data.batch
        elif isinstance(data, torch.Tensor):
            # Direct tensor input
            x = data
        elif isinstance(data, dict):
            # Dictionary input
            x = data.get('x', data)
            if edge_index is None:
                edge_index = data.get('edge_index')
        else:
            # Try to use the data object for feature extraction
            x = data
            
        # Move to device
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        if edge_index is not None:
            edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)
            
        # Process features
        features = self.feature_processor(x)
        
        # Process through graph layers
        if edge_index is not None:
            embeddings = self.graph_processor(features, edge_index)
        else:
            # No graph structure, just use features
            embeddings = features
        
        # Pool if needed for graph-level tasks
        if self.pooling_module is not None and batch is not None:
            pooled = self.pooling_module(embeddings, batch)
        else:
            pooled = embeddings
            
        # Output head
        output = self.output_head(pooled)
        
        if return_embeddings:
            return {'logits': output, 'embeddings': embeddings}
        return output
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 