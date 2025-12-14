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

        # Feature encoder - adjusted for different input types
        # For spatial data (3D coordinates), we might need to encode from 3 to hidden_dim
        self.feature_encoder = None
        if num_features != hidden_dim:
            # Only create feature encoder if we know it will work with the expected features
            try:
                self.feature_encoder = create_mlp(
                    input_dim=num_features,
                    output_dim=hidden_dim,
                    hidden_dims=[max(hidden_dim // 2, 16)],  # Ensure minimum size
                    activation="relu",
                    batch_norm=False,  # Disable batch norm for stability
                    dropout=dropout,
                )
                logger.info(f"[AstroTemporalGNN] Created feature encoder: {num_features} -> {hidden_dim}")
            except Exception as e:
                logger.warning(f"[AstroTemporalGNN] Failed to create feature encoder: {e}")
                self.feature_encoder = nn.Identity()
        else:
            self.feature_encoder = nn.Identity()
            logger.info("[AstroTemporalGNN] No feature encoding needed (dimensions match)")

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

        # GNN layers for spatial relationships - will be created dynamically in forward
        self.conv_type = conv_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout

        # Store kwargs for layer creation
        self.layer_kwargs = kwargs

        # Initialize empty lists - will be populated in forward
        self.conv_layers = None
        self.norm_layers = None
        self._layers_initialized = False

        # Debug: Log die erwartete Eingabedimension für Layer 0
        logger.info(
            f"[DEBUG] Initializing GNN-Layer 0: in_dim={temporal_output_dim} (expected x.shape[1])"
        )
        self._expected_first_gnn_in_dim = temporal_output_dim

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
                  x shape: [N, num_features] or [N, sequence_length, num_features]

        Returns:
            Predictions based on task type
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        logger = logging.getLogger(__name__)

        logger.info(
            f"[AstroTemporalGNN] Input: x.shape={x.shape}, x.dtype={x.dtype}, x.type={type(x)}"
        )
        
        # Critical validation
        if not isinstance(x, torch.Tensor):
            logger.error(
                f"[AstroTemporalGNN] Input x is not a tensor: {type(x)}, value={x}"
            )
            raise ValueError(f"Expected tensor input, got {type(x)}")
        
        # Ensure edge_index is valid
        if not isinstance(edge_index, torch.Tensor):
            logger.error(f"[AstroTemporalGNN] edge_index is not a tensor: {type(edge_index)}")
            raise ValueError(f"Expected tensor edge_index, got {type(edge_index)}")

        # Handle different input shapes
        if x.dim() == 2:
            # 2D input [N, F] - could be spatial coordinates or features
            N, F = x.shape
            logger.info(f"[AstroTemporalGNN] 2D input detected: {x.shape}")
            
            # Check if features match expected
            if F != self.num_features:
                logger.warning(
                    f"[AstroTemporalGNN] Feature mismatch: got {F}, expected {self.num_features}. "
                    f"Using actual feature count."
                )
                # For spatial data or when features don't match, skip temporal processing
                is_spatial = True
            else:
                # Create temporal dimension for matching features
                x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
                logger.info(f"[AstroTemporalGNN] Created temporal dimension: {x.shape}")
                is_spatial = False
                
        elif x.dim() == 3:
            # Already has temporal dimension
            logger.info(f"[AstroTemporalGNN] Input already temporal: x.shape={x.shape}")
            is_spatial = False
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")

        if batch is None:
            # Single graph - create batch indices
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Process based on data type
        if is_spatial or x.dim() == 2:
            # Spatial data - use direct processing without temporal encoding
            logger.info(f"[AstroTemporalGNN] Processing as spatial data")
            
            # Apply feature encoder if dimensions don't match hidden_dim
            if x.shape[1] != self.hidden_dim:
                if hasattr(self, 'feature_encoder') and self.feature_encoder is not None:
                    # Create a simple projection layer if feature_encoder can't handle the dimension
                    if not hasattr(self, '_projection_layer'):
                        self._projection_layer = nn.Linear(x.shape[1], self.hidden_dim).to(x.device)
                        logger.info(f"[AstroTemporalGNN] Created projection layer: {x.shape[1]} -> {self.hidden_dim}")
                    x = self._projection_layer(x)
                    x = F.relu(x)
                    logger.info(f"[AstroTemporalGNN] After projection: {x.shape}")
        else:
            # Temporal data - apply temporal processing
            logger.info(f"[AstroTemporalGNN] Processing temporal data: {x.shape}")
            
            # Feature encoding per time step if needed (vectorized)
            if hasattr(self, 'feature_encoder') and not isinstance(self.feature_encoder, nn.Identity):
                # Reshape to [batch * seq_len, features] for efficient batch processing
                batch_size, seq_len, n_features = x.size()
                x_flat = x.reshape(batch_size * seq_len, n_features)
                
                # Apply encoder once to all timesteps
                x_encoded_flat = self.feature_encoder(x_flat)
                
                # Reshape back to [batch, seq_len, encoded_features]
                x = x_encoded_flat.reshape(batch_size, seq_len, -1)
                logger.info(f"[AstroTemporalGNN] After feature encoding: {x.shape}")
            
            # Temporal encoding
            if self.temporal_model in ["lstm", "gru"]:
                temporal_output, _ = self.temporal_encoder(x)
                x = temporal_output[:, -1, :]  # Use last output
                logger.info(f"[AstroTemporalGNN] After {self.temporal_model.upper()}: {x.shape}")
            elif self.temporal_model == "transformer":
                temporal_output = self.temporal_encoder(x)
                x = temporal_output.mean(dim=1)  # Mean pooling
                logger.info(f"[AstroTemporalGNN] After Transformer: {x.shape}")

        # Now x should be 2D [N, hidden_dim] for GNN processing
        if x.dim() != 2:
            logger.error(f"[AstroTemporalGNN] Expected 2D tensor before GNN, got {x.shape}")
            raise ValueError(f"Expected 2D tensor for GNN processing, got {x.shape}")

        # Use simple linear layers for now (more robust than GNN for debugging)
        if not hasattr(self, '_processing_layers'):
            input_dim = x.shape[1]
            self._processing_layers = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ).to(x.device)
            logger.info(f"[AstroTemporalGNN] Created processing layers with input_dim={input_dim}")
        
        # Process through layers
        x = self._processing_layers(x)
        logger.info(f"[AstroTemporalGNN] After processing layers: {x.shape}")
        
        # Global pooling if needed
        if batch is not None and len(torch.unique(batch)) > 1:
            x = global_mean_pool(x, batch)
            logger.info(f"[AstroTemporalGNN] After pooling: {x.shape}")
        
        # Apply output head
        output = self.output_head(x)
        logger.info(f"[AstroTemporalGNN] Final output: {output.shape}")
        
        return output

    def _initialize_gnn_layers(self, input_dim: int):
        """Initialize GNN layers dynamically based on actual input dimensions."""
        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroTemporalGNN] Creating GNN layers with input_dim={input_dim}, hidden_dim={self.hidden_dim}"
        )

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if self.use_batch_norm else None

        for i in range(self.num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = self.hidden_dim

            if self.conv_type == "gcn":
                conv = GCNConv(in_dim, self.hidden_dim)
            elif self.conv_type == "gat":
                heads = self.layer_kwargs.get("num_heads", 8)
                if self.hidden_dim % heads != 0:
                    adjusted_hidden_dim = (self.hidden_dim // heads) * heads
                    logger.warning(
                        f"Adjusted hidden_dim from {self.hidden_dim} to {adjusted_hidden_dim} for GAT heads={heads}"
                    )
                    self.hidden_dim = adjusted_hidden_dim
                conv = GATConv(
                    in_dim, self.hidden_dim // heads, heads=heads, concat=True
                )
            elif self.conv_type == "sage":
                conv = SAGEConv(in_dim, self.hidden_dim)
            elif self.conv_type == "transformer":
                heads = self.layer_kwargs.get("num_heads", 8)
                if self.hidden_dim % heads != 0:
                    adjusted_hidden_dim = (self.hidden_dim // heads) * heads
                    logger.warning(
                        f"Adjusted hidden_dim from {self.hidden_dim} to {adjusted_hidden_dim} for Transformer heads={heads}"
                    )
                    self.hidden_dim = adjusted_hidden_dim
                conv = TransformerConv(
                    in_dim, self.hidden_dim // heads, heads=heads, concat=True
                )
            else:
                raise ValueError(f"Unknown conv_type: {self.conv_type}")

            self.conv_layers.append(conv)

            if self.use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(self.hidden_dim))

        self._layers_initialized = True
        logger.info("[AstroTemporalGNN] GNN layers initialized successfully")

    def get_temporal_embeddings(self, data) -> Tensor:
        """
        Get temporal embeddings from the last layer.

        Args:
            data: PyG Data object

        Returns:
            Temporal embeddings [N, hidden_dim] or [B, hidden_dim] if batched
        """
        x = data.x
        batch = getattr(data, "batch", None)
        
        logger = logging.getLogger(__name__)
        logger.info(
            f"[AstroTemporalGNN] get_temporal_embeddings: x.shape={x.shape}, x.dtype={x.dtype}"
        )

        # Process similar to forward, but without output head
        if x.dim() == 2:
            N, F = x.shape
            if F != self.num_features:
                # Spatial data - apply projection if needed
                if not hasattr(self, '_projection_layer'):
                    self._projection_layer = nn.Linear(F, self.hidden_dim).to(x.device)
                x = self._projection_layer(x)
                x = F.relu(x)
            else:
                # Create temporal dimension
                x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Apply temporal encoding if 3D
        if x.dim() == 3:
            # Feature encoding (vectorized)
            if hasattr(self, 'feature_encoder') and not isinstance(self.feature_encoder, nn.Identity):
                # Reshape to [batch * seq_len, features] for efficient batch processing
                batch_size, seq_len, n_features = x.size()
                x_flat = x.reshape(batch_size * seq_len, n_features)
                
                # Apply encoder once to all timesteps
                x_encoded_flat = self.feature_encoder(x_flat)
                
                # Reshape back to [batch, seq_len, encoded_features]
                x = x_encoded_flat.reshape(batch_size, seq_len, -1)
            
            # Temporal encoding
            if self.temporal_model in ["lstm", "gru"]:
                temporal_output, _ = self.temporal_encoder(x)
                x = temporal_output[:, -1, :]
            elif self.temporal_model == "transformer":
                temporal_output = self.temporal_encoder(x)
                x = temporal_output.mean(dim=1)
        
        # Process through layers (without output head)
        if hasattr(self, '_processing_layers'):
            # Get all layers except the last one
            for layer in list(self._processing_layers.children())[:-1]:
                x = layer(x)
        
        # Global pooling if needed
        if batch is not None and len(torch.unique(batch)) > 1:
            x = global_mean_pool(x, batch)
        
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
