"""Simple layer creation functions for AstroLab models."""

from typing import List, Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv


def create_conv_layer(
    conv_type: str, 
    in_channels: int, 
    out_channels: int, 
    **kwargs
) -> nn.Module:
    """Simple function to create conv layers."""
    conv_type = conv_type.lower()
    
    if conv_type == "gcn":
        return GCNConv(in_channels, out_channels)
    elif conv_type == "gat":
        heads = kwargs.get('heads', 8)
        return GATConv(in_channels, out_channels // heads, heads=heads)
    elif conv_type == "sage":
        return SAGEConv(in_channels, out_channels)
    elif conv_type == "transformer":
        heads = kwargs.get('heads', 8)
        return TransformerConv(in_channels, out_channels // heads, heads=heads)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}. Available: gcn, gat, sage, transformer")


def create_mlp(
    input_dim: int, 
    output_dim: int, 
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
    activation: str = 'relu',
    batch_norm: bool = False
) -> nn.Module:
    """Create simple MLP."""
    if hidden_dims is None:
        hidden_dims = [input_dim // 2]
    
    layers = []
    prev_dim = input_dim
    
    # Hidden layers
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
            
        layers.append(get_activation(activation))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        
    return activations[name]


class ResidualBlock(nn.Module):
    """Simple residual block for deeper networks."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return torch.relu(x + residual) 