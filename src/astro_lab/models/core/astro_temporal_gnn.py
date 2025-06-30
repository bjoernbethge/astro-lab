"""
AstroTemporalGNN - Temporal Neural Network
==========================================

temporal neural network for time series tasks.
"""

import torch

from ..components.base import AdvancedTemporalEncoder, create_output_head
from .base_model import AstroBaseModel


class AstroTemporalGNN(AstroBaseModel):
    """Temporal neural network for astronomical time series tasks."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
        task: str = "time_series_classification",
        **kwargs,
    ):
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        # Temporal encoder
        self.encoder = AdvancedTemporalEncoder(
            in_dim=num_features,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            encoder_type="lstm",
            bidirectional=True,
            attention=True,
            dropout=dropout,
        )

        # Output head
        self.output_head = create_output_head(
            task_type=task, in_dim=hidden_dim, out_dim=num_classes, dropout=dropout
        )

    def forward(self, batch):
        """Forward pass through the network."""
        # Handle different input formats
        x = getattr(batch, "x", None)
        if x is None:
            x = batch
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a tensor or batch with attribute 'x'.")
        # Ensure 3D shape [batch, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # Encode sequence
        x = self.encoder(x)
        # Output
        return self.output_head(x)
