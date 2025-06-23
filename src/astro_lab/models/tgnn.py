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
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.task = task
        self.use_static_features = kwargs.get("use_static_features", False)
        self.use_residual = kwargs.get("use_residual", True)

        # Encoder for lightcurve data
        self.encoder = LightcurveEncoder(
            input_dim=kwargs.get("lightcurve_features", 1),
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=self.device
        )

        # Optional encoder for static (non-temporal) features
        if self.use_static_features:
            static_input_dim = kwargs.get("static_features_dim", 16)
            self.static_encoder = nn.Sequential(
                Linear(static_input_dim, hidden_dim // 2),
                nn.ReLU(),
                Linear(hidden_dim // 2, hidden_dim // 2),
            ).to(self.device)
            fusion_dim = hidden_dim + hidden_dim // 2
        else:
            self.static_encoder = None
            fusion_dim = hidden_dim

        # Graph convolutions for temporal relationships (if needed)
        self.convs = nn.ModuleList(
            [GCNConv(fusion_dim, fusion_dim) for _ in range(num_layers)]
        )
        
        # Add missing norms attribute
        self.norms = nn.ModuleList([
            nn.LayerNorm(fusion_dim) for _ in range(num_layers)
        ])

        # Task-specific output heads
        if self.task == "period_detection":
            self.output_head = PeriodDetectionHead(fusion_dim, output_dim)
        elif self.task == "shape_modeling":
            self.output_head = ShapeModelingHead(fusion_dim, output_dim)
        elif self.task == "classification":
            num_classes = kwargs.get("num_classes", 2)
            self.output_head = ClassificationHead(fusion_dim, num_classes)
        else:
            raise ValueError(f"Unknown task for ALCDEFTemporalGNN: {self.task}")

        self.apply(initialize_weights)
        self.to(self.device)

    def forward(
        self,
        lightcurve: LightcurveTensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the ALCDEF Temporal GNN.
        """
        # Move inputs to correct device
        if edge_index.device != self.device:
            edge_index = edge_index.to(self.device)
        
        # 1. ENCODE
        # h shape: (batch_size, num_timesteps, hidden_dim)
        h = self.encoder(lightcurve)

        # If we processed a single lightcurve, squeeze the batch dimension but preserve sequence
        if h.shape[0] == 1:
            h = h.squeeze(0)  # Now (seq_len, hidden_dim)

        # 2. CONVOLVE
        # Apply graph convolutions over the timesteps
        for i, conv in enumerate(self.convs):
            h_prev = h
            h = conv(h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.use_residual and h_prev.shape == h.shape:
                h = h + h_prev

        # 3. POOL
        # Pool the node (timestep) embeddings into a single graph embedding
        # If we have a batch, use torch_geometric's pooling
        if batch is not None:
             pooled_h = global_mean_pool(h, batch)
        else:
             # Otherwise, just take the mean
             pooled_h = torch.mean(h, dim=0)

        # 4. PREDICT
        # Pass through the appropriate output head based on the task
        output = self.output_head(pooled_h)

        if return_embeddings:
            return {"predictions": output, "embeddings": pooled_h}

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
