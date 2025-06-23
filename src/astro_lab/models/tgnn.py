"""
Temporal Graph Convolutional Networks

Base classes for temporal graph neural networks in astronomical applications.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, Linear, global_mean_pool

from astro_lab.tensors.lightcurve import LightcurveTensor
from astro_lab.models.encoders import LightcurveEncoder
from astro_lab.models.utils import initialize_weights
from astro_lab.models.layers import LayerFactory

logger = logging.getLogger(__name__)

__all__ = ["TemporalGCN", "TemporalGATCNN"]

class TemporalGCN(nn.Module):
    """
    Base Temporal Graph Convolutional Network for astronomical time-series data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        graph_layers: int = 3,
        recurrent_layers: int = 2,
        recurrent_type: str = "lstm",
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Initialize Temporal GCN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            graph_layers: Number of graph convolution layers
            recurrent_layers: Number of recurrent layers
            recurrent_type: Type of recurrent layer ('lstm', 'gru')
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_layers = graph_layers
        self.recurrent_layers = recurrent_layers
        self.recurrent_type = recurrent_type
        self.dropout = dropout

        # Input projection
        self.input_projection = LayerFactory.create_mlp(input_dim, hidden_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(graph_layers):
            self.convs.append(LayerFactory.create_conv_layer("gcn", hidden_dim, hidden_dim))

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(graph_layers)]
        )

        # Recurrent layer for temporal processing
        if recurrent_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=recurrent_layers,
                dropout=dropout if recurrent_layers > 1 else 0,
                batch_first=True,
            )
        elif recurrent_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=recurrent_layers,
                dropout=dropout if recurrent_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown recurrent_type: {recurrent_type}")

        # Output layer
        self.output_layer = nn.Sequential(
            LayerFactory.create_mlp(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            LayerFactory.create_mlp(hidden_dim // 2, output_dim),
        )

        self.apply(initialize_weights)

        logger.info(
            f"Initialized TemporalGCN with {graph_layers} graph layers and {recurrent_layers} {recurrent_type} layers"
        )

    def encode_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a single graph snapshot.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            Graph embedding
        """
        # Input projection
        h = self.input_projection(x)

        # Apply graph convolutions
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual connection for deeper layers
            if i > 0:
                h = h + h_prev

        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = torch.mean(h, dim=0, keepdim=True)

        return h

    def forward(
        self, snapshot_sequence: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through temporal sequence.

        Args:
            snapshot_sequence: List of graph snapshots

        Returns:
            Temporal predictions
        """
        # Encode each snapshot
        graph_embeddings = []

        for snapshot in snapshot_sequence:
            h = self.encode_snapshot(
                snapshot["x"], snapshot["edge_index"], snapshot.get("batch")
            )
            graph_embeddings.append(h)

        # Stack for temporal processing
        if len(graph_embeddings) > 1:
            graph_sequence = torch.stack(
                graph_embeddings, dim=1
            )  # [batch, seq_len, hidden_dim]

            # Apply recurrent processing
            rnn_out, hidden = self.rnn(graph_sequence)
            final_embedding = rnn_out[:, -1, :]  # Use last output
        else:
            # Single snapshot case
            final_embedding = graph_embeddings[0]

        # Final prediction
        predictions = self.output_layer(final_embedding)

        return {"predictions": predictions, "embeddings": final_embedding}

class TemporalGATCNN(TemporalGCN):
    """
    Temporal Graph Attention Network with attention-based processing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        heads: int = 8,
        attention_dropout: float = 0.1,
        **kwargs,
    ):
        """
        Initialize Temporal GAT.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            heads: Number of attention heads
            attention_dropout: Attention dropout rate
        """
        super().__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
        )

        self.heads = heads
        self.attention_dropout = attention_dropout

        # Replace GCN layers with GAT layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=attention_dropout,
                concat=True,
            )
        )

        # Hidden layers
        for _ in range(self.graph_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=attention_dropout,
                    concat=True,
                )
            )

        # Final layer
        if self.graph_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=1,
                    dropout=attention_dropout,
                    concat=False,
                )
            )

        logger.info(f"Initialized TemporalGATCNN with {heads} attention heads")

    def encode_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode snapshot with attention mechanism.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            Attention-enhanced graph embedding
        """
        # Input projection
        h = self.input_projection(x)

        # Apply GAT convolutions
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            # Apply activation except for last layer
            if i < len(self.convs) - 1:
                h = F.elu(h)  # ELU works better with GAT
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = torch.mean(h, dim=0, keepdim=True)

        return h

# TNG-specific models moved to tng_models.py

class ALCDEFTemporalGNN(nn.Module):
    """Temporal Graph Neural Network for ALCDEF lightcurve data with native tensor support."""

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        task: str = "period_detection",  # period_detection, shape_modeling, classification
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.task = task
        self.dropout = dropout

        # Encoders for node features
        self.lightcurve_encoder = LightcurveEncoder(
            input_dim=config.get("lightcurve_features", 1),
            hidden_dim=hidden_dim, 
            output_dim=hidden_dim
        )
        if self.use_static_features:
            self.static_encoder = nn.Linear(
                config.get("static_features_dim", 1), hidden_dim)

        # Graph convolution layers for temporal relationships
        self.convs = nn.ModuleList(
            [LayerFactory.create_conv_layer("gcn", hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Task-specific output heads
        if task == "period_detection":
            self.output_head = PeriodDetectionHead(hidden_dim, output_dim)
        elif task == "shape_modeling":
            self.output_head = ShapeModelingHead(hidden_dim, output_dim)
        elif task == "classification":
            self.output_head = ClassificationHead(hidden_dim, output_dim)
        else:
            # Generic regression head
            self.output_head = nn.Sequential(
                LayerFactory.create_mlp(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                LayerFactory.create_mlp(hidden_dim // 2, output_dim),
            )

        self.apply(initialize_weights)

    def forward(
        self,
        lightcurve: LightcurveTensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with native lightcurve support."""
        # Extract features using existing LightcurveEncoder
        h = self.lightcurve_encoder(lightcurve)

        # Graph convolutions for temporal relationships
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)

            if self.training:
                h = F.dropout(h, p=self.dropout)

            if i > 0 and h.size(-1) == h_prev.size(-1):
                h = h + h_prev

        embeddings = h

        # Global pooling for sequence-level tasks
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        pooled = global_mean_pool(h, batch)
        output = self.output_head(pooled)

        if return_embeddings:
            return {"output": output, "embeddings": embeddings}
        return output

class PeriodDetectionHead(nn.Module):
    """Output head for rotation period detection."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            LayerFactory.create_mlp(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            LayerFactory.create_mlp(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            LayerFactory.create_mlp(hidden_dim // 4, output_dim),
            nn.Softplus(),  # Ensure positive period values
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class ShapeModelingHead(nn.Module):
    """Output head for shape modeling parameters."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            LayerFactory.create_mlp(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            LayerFactory.create_mlp(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            LayerFactory.create_mlp(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class ClassificationHead(nn.Module):
    """Output head for asteroid classification."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            LayerFactory.create_mlp(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            LayerFactory.create_mlp(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
