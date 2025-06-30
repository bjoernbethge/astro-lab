"""
AstroNodeGNN - Node-Level Neural Network for Astronomical Data
===================================================================

Simplified implementation focusing on core functionality.
"""

from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# PyTorch Geometric
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
)

from .base_model import AstroBaseModel


class AstroNodeGNN(AstroBaseModel):
    """
    Simplified node-level GNN for astronomical object classification and analysis.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        task: str = "node_classification",
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

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Graph layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = hidden_dim if i == 0 else hidden_dim
            layer_out = hidden_dim

            if conv_type == "gcn":
                conv = GCNConv(layer_in, layer_out)
            elif conv_type == "gat":
                conv = GATConv(layer_in, layer_out // heads, heads=heads)
            elif conv_type == "sage":
                conv = SAGEConv(layer_in, layer_out)
            elif conv_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(layer_in, 2 * layer_out),
                    nn.ReLU(),
                    nn.Linear(2 * layer_out, layer_out),
                )
                conv = GINConv(mlp)
            elif conv_type == "transformer":
                conv = TransformerConv(layer_in, layer_out, heads=heads)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.layers.append(conv)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch: Union[Data, HeteroData, Batch]) -> Tensor:
        """Forward pass through the network."""
        x = getattr(batch, "x", None)
        edge_index = getattr(batch, "edge_index", None)
        if x is None or edge_index is None:
            raise ValueError("Batch must have 'x' (node features) and 'edge_index'!")
        edge_attr = getattr(batch, "edge_attr", None)

        # Input projection
        x = self.input_proj(x)
        x = F.gelu(x)

        # Graph layers
        for layer in self.layers:
            # Handle different conv types
            if isinstance(layer, GCNConv):
                # GCN doesn't support edge_attr
                x = layer(x, edge_index)
            elif isinstance(layer, GATConv):
                # GAT supports edge_attr
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif isinstance(layer, SAGEConv):
                # SAGE supports edge_attr
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif isinstance(layer, GINConv):
                # GIN doesn't support edge_attr
                x = layer(x, edge_index)
            elif isinstance(layer, TransformerConv):
                # Transformer supports edge_attr
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                # Default fallback
                x = layer(x, edge_index)

        # Output projection
        return self.output_head(x)

    def get_node_embeddings(self, batch: Union[Data, HeteroData, Batch]) -> Tensor:
        """Get node embeddings without final classification layers."""
        x = getattr(batch, "x", None)
        edge_index = getattr(batch, "edge_index", None)
        if x is None or edge_index is None:
            raise ValueError("Batch must have 'x' (node features) and 'edge_index'!")
        edge_attr = getattr(batch, "edge_attr", None)

        # Input projection
        x = self.input_proj(x)
        x = F.gelu(x)

        # Graph layers
        for layer in self.layers:
            # Handle different conv types
            if isinstance(layer, GCNConv):
                # GCN doesn't support edge_attr
                x = layer(x, edge_index)
            elif isinstance(layer, GATConv):
                # GAT supports edge_attr
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif isinstance(layer, SAGEConv):
                # SAGE supports edge_attr
                x = layer(x, edge_index, edge_attr=edge_attr)
            elif isinstance(layer, GINConv):
                # GIN doesn't support edge_attr
                x = layer(x, edge_index)
            elif isinstance(layer, TransformerConv):
                # Transformer supports edge_attr
                x = layer(x, edge_index, edge_attr=edge_attr)
            else:
                # Default fallback
                x = layer(x, edge_index)

        return x


def create_stellar_classification_model(
    num_features: int, hidden_dim: int = 128, **kwargs
) -> AstroNodeGNN:
    """Create a specialized model for stellar classification."""
    return AstroNodeGNN(
        num_features=num_features,
        num_classes=7,  # O, B, A, F, G, K, M
        hidden_dim=hidden_dim,
        task="node_classification",
        **kwargs,
    )


def create_galaxy_analysis_model(
    num_features: int, hidden_dim: int = 128, **kwargs
) -> AstroNodeGNN:
    """Create a specialized model for galaxy analysis."""
    return AstroNodeGNN(
        num_features=num_features,
        num_classes=6,  # E, S0, Sa, Sb, Sc, Irr
        hidden_dim=hidden_dim,
        task="node_classification",
        **kwargs,
    )


def create_variable_star_detector(
    num_features: int, hidden_dim: int = 128, **kwargs
) -> AstroNodeGNN:
    """Create a specialized model for variable star detection."""
    return AstroNodeGNN(
        num_features=num_features,
        num_classes=2,  # Variable/non-variable
        hidden_dim=hidden_dim,
        task="node_classification",
        **kwargs,
    )
