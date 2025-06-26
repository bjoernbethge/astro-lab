"""
AstroTemporalGNN - Unified Temporal Graph Neural Network
======================================================

Single model for all temporal astronomical tasks:
- Lightcurve classification
- Transient detection
- Time series forecasting
- Anomaly detection
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ..components import AstroLightningMixin, create_mlp, create_output_head


class AstroTemporalGNN(AstroLightningMixin, LightningModule):
    """
    Unified Graph Neural Network for temporal astronomical tasks.

    Supports all temporal tasks with a single, flexible architecture:
    - Time Series Classification: Lightcurve types, transient types
    - Time Series Forecasting: Future lightcurve prediction
    - Anomaly Detection: Outlier detection in time series
    - Temporal Segmentation: Event detection in lightcurves

    Args:
        num_features: Number of input features per time step
        num_classes: Number of output classes (for classification)
        hidden_dim: Base hidden dimension
        num_layers: Number of GNN layers
        conv_type: GNN convolution type ('gcn', 'gat', 'sage', 'transformer')
        temporal_model: Temporal model type ('lstm', 'gru', 'transformer', 'tcn')
        task: Task type ('time_series_classification', 'forecasting', 'anomaly_detection')
        sequence_length: Length of input time series
        learning_rate: Learning rate for training
        optimizer: Optimizer type ('adam', 'adamw')
        scheduler: Scheduler type ('cosine', 'onecycle')
        weight_decay: Weight decay for regularization
        dropout: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        num_features: int = 64,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gcn",
        temporal_model: str = "lstm",
        task: str = "time_series_classification",
        sequence_length: int = 100,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        # Initialize Lightning mixin first
        super().__init__(
            task=task,
            learning_rate=learning_rate,
            optimizer=optimizer,
            scheduler=scheduler,
            weight_decay=weight_decay,
            **kwargs,
        )

        # Initialize LightningModule
        LightningModule.__init__(self)
        self.save_hyperparameters()

        # Setup criterion after LightningModule initialization
        self._setup_criterion()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.temporal_model = temporal_model
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroTemporalGNN] Init: num_features={num_features}, hidden_dim={hidden_dim}"
        )

        # Feature encoder
        if num_features != hidden_dim:
            self.feature_encoder = create_mlp(
                input_dim=num_features,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim // 2],
                activation="relu",
                batch_norm=use_batch_norm,
                dropout=dropout,
            )
        else:
            self.feature_encoder = nn.Identity()

        # Temporal encoder
        if temporal_model == "lstm":
            self.temporal_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if 2 > 1 else 0,
                bidirectional=True,
            )
            temporal_output_dim = hidden_dim * 2  # Bidirectional
        elif temporal_model == "gru":
            self.temporal_encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout if 2 > 1 else 0,
                bidirectional=True,
            )
            temporal_output_dim = hidden_dim * 2  # Bidirectional
        elif temporal_model == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            temporal_output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}")

        # GNN layers for spatial relationships
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = temporal_output_dim
            else:
                in_dim = hidden_dim

            if conv_type == "gcn":
                conv = GCNConv(in_dim, hidden_dim)
            elif conv_type == "gat":
                heads = kwargs.get("num_heads", 8)
                conv = GATConv(in_dim, hidden_dim // heads, heads=heads)
            elif conv_type == "sage":
                conv = SAGEConv(in_dim, hidden_dim)
            elif conv_type == "transformer":
                heads = kwargs.get("num_heads", 8)
                conv = TransformerConv(in_dim, hidden_dim // heads, heads=heads)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

            self.conv_layers.append(conv)

            if use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())

        # Task-specific output head
        if task == "time_series_classification":
            self.output_head = create_output_head(
                "classification",
                input_dim=hidden_dim,
                output_dim=num_classes,
                dropout=dropout,
            )
        elif task == "forecasting":
            # For forecasting, predict future time steps
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),  # num_classes = forecast_steps
            )
        elif task == "anomaly_detection":
            # For anomaly detection, output anomaly scores
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Anomaly probability
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, data) -> Tensor:
        """
        Forward pass through the network.

        Args:
            data: PyG Data object with x, edge_index, batch
                  x shape: [N, sequence_length, num_features]

        Returns:
            Predictions based on task type
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroTemporalGNN] Forward: x.shape={x.shape}, expected num_features={self.num_features}"
        )
        assert x.shape[-1] == self.num_features, (
            f"Input feature mismatch: x.shape[-1]={x.shape[-1]}, expected={self.num_features}"
        )

        # Reshape for temporal processing
        N = x.size(0)
        x = x.view(N, self.sequence_length, -1)  # [N, seq_len, features]

        # Feature encoding
        x = self.feature_encoder(x)  # [N, seq_len, hidden_dim]

        # Temporal encoding
        if self.temporal_model in ["lstm", "gru"]:
            # Pack sequence if needed
            temporal_output, _ = self.temporal_encoder(x)
            # Use last output for each sequence
            x = temporal_output[:, -1, :]  # [N, temporal_output_dim]
        elif self.temporal_model == "transformer":
            # Add positional encoding if needed
            temporal_output = self.temporal_encoder(x)
            # Use mean pooling over time dimension
            x = temporal_output.mean(dim=1)  # [N, hidden_dim]

        # GNN layers for spatial relationships
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Task-specific output
        return self.output_head(x)

    def get_temporal_embeddings(self, data) -> Tensor:
        """
        Get temporal embeddings from the last GNN layer.

        Args:
            data: PyG Data object

        Returns:
            Temporal embeddings [N, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index

        # Reshape for temporal processing
        N = x.size(0)
        x = x.view(N, self.sequence_length, -1)

        # Feature encoding
        x = self.feature_encoder(x)

        # Temporal encoding
        if self.temporal_model in ["lstm", "gru"]:
            temporal_output, _ = self.temporal_encoder(x)
            x = temporal_output[:, -1, :]
        elif self.temporal_model == "transformer":
            temporal_output = self.temporal_encoder(x)
            x = temporal_output.mean(dim=1)

        # GNN layers (without final output head)
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


def create_astro_temporal_gnn(
    num_features: int = 64,
    num_classes: int = 2,
    task: str = "time_series_classification",
    **kwargs,
) -> AstroTemporalGNN:
    """
    Factory function to create AstroTemporalGNN model.

    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        task: Task type
        **kwargs: Additional model parameters

    Returns:
        Configured AstroTemporalGNN model
    """
    return AstroTemporalGNN(
        num_features=num_features,
        num_classes=num_classes,
        task=task,
        **kwargs,
    )
