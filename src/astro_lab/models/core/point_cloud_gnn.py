"""Simplified ALCDEF Temporal GNN for lightcurve analysis."""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from ..components import (
    create_conv_layer,
    create_mlp,
    create_output_head,
    PeriodDetectionHead,
    ShapeModelingHead,
    ClassificationHead,
)


class ALCDEFTemporalGNN(nn.Module):
    """Simplified Temporal GNN for ALCDEF lightcurve data."""
    
    def __init__(
        self,
        input_dim: int = 1,  # Typically magnitude values
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        task: str = "period_detection",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        super().__init__()
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.hidden_dim = hidden_dim
        self.task = task
        
        # Lightcurve encoder - simple LSTM
        self.lightcurve_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Graph convolutions for temporal relationships
        self.convs = nn.ModuleList([
            create_conv_layer("gcn", hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific output heads
        if task == "period_detection":
            self.output_head = PeriodDetectionHead(hidden_dim, dropout)
        elif task == "shape_modeling":
            num_harmonics = kwargs.get("num_harmonics", 10)
            self.output_head = ShapeModelingHead(hidden_dim, num_harmonics, dropout)
        elif task == "classification":
            num_classes = kwargs.get("num_classes", 2)
            self.output_head = ClassificationHead(hidden_dim, num_classes, dropout)
        else:
            # Default to simple output head
            self.output_head = create_output_head(
                "regression",
                input_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout
            )
            
        self.to(self.device)
        
    def encode_lightcurve(self, lightcurve_data: torch.Tensor) -> torch.Tensor:
        """Encode lightcurve sequence with LSTM."""
        # lightcurve_data shape: (batch, seq_len, features)
        if lightcurve_data.dim() == 2:
            # Add feature dimension if needed
            lightcurve_data = lightcurve_data.unsqueeze(-1)
            
        # Encode with LSTM
        lstm_out, (h_n, _) = self.lightcurve_encoder(lightcurve_data)
        
        # Use last hidden state
        # h_n shape: (num_layers, batch, hidden_dim)
        encoded = h_n[-1]  # Take last layer: (batch, hidden_dim)
        
        return encoded
        
    def forward(
        self,
        lightcurve,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for lightcurve analysis."""
        
        # Move to device
        edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)
            
        # Handle different input formats
        if hasattr(lightcurve, 'data'):
            # LightcurveTensor
            lc_data = lightcurve.data.to(self.device)
        elif isinstance(lightcurve, torch.Tensor):
            lc_data = lightcurve.to(self.device)
        else:
            raise ValueError("Unsupported lightcurve format")
            
        # Encode lightcurve
        h = self.encode_lightcurve(lc_data)
        
        # If we have multiple nodes per graph, expand the encoding
        if edge_index.size(1) > 0:
            num_nodes = edge_index.max().item() + 1
            if h.size(0) == 1 and num_nodes > 1:
                # Broadcast to all nodes
                h = h.expand(num_nodes, -1)
                
        # Apply graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            h_prev = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            
            # Residual connection
            if h_prev.shape == h.shape:
                h = h + h_prev
                
        # Pool if needed
        if batch is not None:
            pooled = global_mean_pool(h, batch)
        else:
            pooled = h.mean(dim=0, keepdim=True)
            
        # Task-specific output
        output = self.output_head(pooled)
        
        if return_embeddings:
            return {"predictions": output, "embeddings": pooled}
        return output 