"""
Base Graph Neural Network Classes für AstroLab

Gemeinsame Base-Klassen für alle astronomischen GNN-Modelle mit:
- Einheitliche Graph-Convolution-Layer
- Feature-Fusion-Module
- Output-Head-Registry
- Robuste Error-Handling
"""

from typing import Any, Dict, List, Literal, Optional, Union

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

from astro_lab.models.utils import get_activation, initialize_weights
from astro_lab.models.layers import LayerFactory

ConvType = Literal["gcn", "gat", "sage", "transformer"]
TaskType = Literal["node_classification", "node_regression", "graph_classification"]


class FeatureFusion(nn.Module):
    """Unified feature fusion module for combining multiple feature types."""

    def __init__(self, input_dims: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        total_dim = sum(input_dims)

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature tensors."""
        concatenated = torch.cat(features, dim=-1)
        return self.fusion(concatenated)


class BaseAstroGNN(nn.Module):
    """Base class for all astronomical GNN models."""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: ConvType = "gcn",
        dropout: float = 0.1,
        activation: str = "relu",
        use_residual: bool = True,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.dropout = dropout
        self.activation = activation
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Build graph convolution layers
        self.convs = self._build_conv_layers()

        # Normalization layers
        if use_layer_norm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
            )
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # Activation function
        self.act_fn = get_activation(activation)

        # Apply weight initialization
        self.apply(initialize_weights)

    def _build_conv_layers(self) -> nn.ModuleList:
        """Build graph convolution layers based on conv_type."""
        convs = nn.ModuleList()

        for i in range(self.num_layers):
            conv = LayerFactory.create_conv_layer(
                self.conv_type, self.hidden_dim, self.hidden_dim
            )
            convs.append(conv)
        return convs

    def graph_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """Standard graph convolution forward pass."""
        h = x
        intermediate_outputs = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h

            # Graph convolution
            h = conv(h, edge_index)

            # Normalization
            h = norm(h)

            # Activation
            h = self.act_fn(h)

            # Dropout
            if self.training:
                h = F.dropout(h, p=self.dropout)

            # Residual connection (skip first layer to avoid dimension mismatch)
            if self.use_residual and i > 0 and h.size(-1) == h_prev.size(-1):
                h = h + h_prev

            if return_intermediate:
                intermediate_outputs.append(h)

        return intermediate_outputs if return_intermediate else h

    def get_pooling_fn(self, pooling: str):
        """Get pooling function by name."""
        pooling_fns = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "add": global_add_pool,
        }
        return pooling_fns.get(pooling, global_mean_pool)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Default forward pass - can be overridden by subclasses."""
        # Project input to hidden dimension
        h = self.input_projection(x)

        # Apply graph convolutions
        h = self.graph_forward(h, edge_index)

        # Ensure we return a tensor, not a list
        if isinstance(h, list):
            h = h[-1]  # Take the last layer output

        return h


class BaseTemporalGNN(BaseAstroGNN):
    """Base class for all temporal GNN models."""

    def __init__(
        self, recurrent_type: str = "lstm", recurrent_layers: int = 2, **kwargs
    ):
        super().__init__(**kwargs)

        self.recurrent_type = recurrent_type
        self.recurrent_layers = recurrent_layers

        # Build RNN layers
        self.rnn = self._build_rnn()

    def _build_rnn(self) -> nn.Module:
        """Build RNN layers based on recurrent_type."""
        if self.recurrent_type == "lstm":
            return nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.recurrent_layers,
                dropout=self.dropout if self.recurrent_layers > 1 else 0,
                batch_first=True,
            )
        elif self.recurrent_type == "gru":
            return nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.recurrent_layers,
                dropout=self.dropout if self.recurrent_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown recurrent_type: {self.recurrent_type}")

    def temporal_forward(self, snapshot_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Unified temporal processing."""
        if len(snapshot_embeddings) == 1:
            return snapshot_embeddings[0]

        # Stack embeddings for sequence processing
        sequence = torch.stack(
            snapshot_embeddings, dim=1
        )  # [batch, seq_len, hidden_dim]

        # Apply RNN
        rnn_out, _ = self.rnn(sequence)

        # Return last output
        return rnn_out[:, -1, :]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with temporal processing."""
        # Project input to hidden dimension
        h = self.input_projection(x)

        # Apply graph convolutions
        h = self.graph_forward(h, edge_index)

        # Ensure we return a tensor, not a list
        if isinstance(h, list):
            h = h[-1]

        # If temporal features provided, combine them
        if temporal_features is not None:
            # Simple temporal processing - can be overridden
            snapshots = [h, temporal_features]
            h = self.temporal_forward(snapshots)

        return h


class BaseTNGModel(BaseTemporalGNN):
    """Base class for all TNG simulation models."""

    def __init__(
        self,
        cosmological_features: bool = True,
        redshift_encoding: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cosmological_features = cosmological_features
        self.redshift_encoding = redshift_encoding

        # Redshift encoder
        if redshift_encoding:
            self.redshift_encoder = self._build_redshift_encoder()
            self.time_projection = nn.Linear(
                self.hidden_dim + self.hidden_dim // 4, self.hidden_dim
            )

        # Cosmological parameter head
        if cosmological_features:
            self.cosmo_head = self._build_cosmo_head()

    def _build_redshift_encoder(self) -> nn.Module:
        """Build redshift encoding module."""
        return nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 4),
        )

    def _build_cosmo_head(self) -> nn.Module:
        """Build cosmological parameter prediction head."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 6),  # Omega_m, Omega_L, h, sigma_8, n_s, w
        )

    def encode_with_cosmology(
        self, x: torch.Tensor, edge_index: torch.Tensor, redshift: float
    ) -> torch.Tensor:
        """Standard cosmological encoding."""
        # Graph forward pass
        h = self.graph_forward(self.input_projection(x), edge_index)

        # Ensure we have a tensor
        if isinstance(h, list):
            h = h[-1]

        # Add redshift encoding if enabled
        if self.redshift_encoding:
            z_tensor = torch.full((h.size(0), 1), redshift, device=h.device)
            z_encoded = self.redshift_encoder(z_tensor)
            h_combined = torch.cat([h, z_encoded], dim=1)
            h = self.time_projection(h_combined)

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        redshift: Optional[float] = None,
    ) -> torch.Tensor:
        """Forward pass with cosmological features."""
        if redshift is not None:
            return self.encode_with_cosmology(x, edge_index, redshift)
        else:
            # Standard temporal forward
            return super().forward(x, edge_index)
