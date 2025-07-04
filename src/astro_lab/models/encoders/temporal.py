"""
Temporal Encoders
================

TensorDict-based encoders for time-series astronomical data.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule


class TemporalEncoderModule(TensorDictModule):
    """
    TensorDict module for encoding temporal/time-series data.

    Features:
    - Variable-length sequence handling
    - Period detection support
    - Multi-band light curves
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        use_attention: bool = True,
        max_sequence_length: int = 1000,
        in_keys: List[str] = ["times", "values"],
        out_keys: List[str] = ["temporal_features"],
        errors_key: Optional[str] = "value_errors",
    ):
        output_dim = output_dim or hidden_dim

        encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_attention=use_attention,
            max_sequence_length=max_sequence_length,
        )

        # Add errors to input keys if provided
        if errors_key:
            in_keys = in_keys + [errors_key]

        super().__init__(
            module=encoder,
            in_keys=in_keys,
            out_keys=out_keys,
        )

        self.errors_key = errors_key


class TemporalEncoder(nn.Module):
    """Core temporal encoding network."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        use_attention: bool = True,
        max_sequence_length: int = 1000,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length

        # Time encoding
        self.time_encoder = TimeEncoding(hidden_dim // 4)

        # Input projection
        self.input_proj = nn.Linear(
            input_dim + hidden_dim // 4,  # values + time encoding
            hidden_dim,
        )

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # bidirectional
                num_heads=4,
                batch_first=True,
            )
            self.attention_pool = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.attention = None

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass expecting (times, values) or (times, values, errors).
        """
        times = args[0]
        values = args[1]
        errors = args[2] if len(args) > 2 else None

        batch_size, seq_len = times.shape

        # Ensure values have feature dimension
        if values.dim() == 2:
            values = values.unsqueeze(-1)

        # Time encoding
        time_features = self.time_encoder(times)

        # Combine time and value features
        combined = torch.cat([values, time_features], dim=-1)

        # Input projection
        lstm_input = self.input_proj(combined)

        # Weight by inverse variance if errors provided
        if errors is not None:
            if errors.dim() == 2:
                errors = errors.unsqueeze(-1)
            weights = 1.0 / (errors**2 + 1e-6)
            weights = weights / weights.mean(dim=1, keepdim=True)
            lstm_input = lstm_input * weights

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # Attention pooling
        if self.attention is not None:
            attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Global average pooling
            pooled = attended.mean(dim=1)
        else:
            # Use final hidden states
            h_n = h_n.transpose(0, 1).contiguous()  # [batch, layers*directions, hidden]
            h_n = h_n.view(batch_size, -1)  # [batch, layers*directions*hidden]
            pooled = h_n

        # Output projection
        return self.output_net(pooled)


class TimeEncoding(nn.Module):
    """Learnable time encoding for astronomical time series."""

    def __init__(self, encoding_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim

        # Learnable frequency components
        self.frequencies = nn.Parameter(torch.randn(encoding_dim // 2) * 0.1)

        # Phase shifts
        self.phases = nn.Parameter(torch.randn(encoding_dim // 2) * 0.1)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Apply time encoding."""
        # Normalize times to [0, 1] range
        time_min = times.min(dim=1, keepdim=True)[0]
        time_max = times.max(dim=1, keepdim=True)[0]
        norm_times = (times - time_min) / (time_max - time_min + 1e-6)

        encoding = []

        for i in range(self.encoding_dim // 2):
            # Sin and cos components with learnable frequency and phase
            angle = 2 * torch.pi * self.frequencies[i] * norm_times + self.phases[i]
            encoding.append(torch.sin(angle))
            encoding.append(torch.cos(angle))

        return torch.stack(encoding, dim=-1)


class MultiTimescaleEncoderModule(TensorDictModule):
    """
    TensorDict module for multi-timescale analysis.

    Useful for transient detection across different timescales.
    """

    def __init__(
        self,
        timescales: List[float] = [1.0, 10.0, 100.0],  # days
        hidden_dim: int = 128,
        fusion_dim: int = 256,
        in_keys: List[str] = ["times", "values"],
        out_key: str = "multiscale_temporal_features",
    ):
        # Create encoder for each timescale
        encoders = []
        for timescale in timescales:
            encoder = TemporalEncoder(
                input_dim=1,
                hidden_dim=hidden_dim,
                output_dim=fusion_dim,
            )
            encoders.append(encoder)

        # Build multi-scale module
        module = MultiTimescaleEncoder(encoders, timescales, fusion_dim)

        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=[out_key],
        )


class MultiTimescaleEncoder(nn.Module):
    """Core multi-timescale encoder."""

    def __init__(
        self, encoders: List[nn.Module], timescales: List[float], fusion_dim: int
    ):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.timescales = timescales

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(encoders), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, times: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Process at multiple timescales."""
        encoded_features = []

        for encoder, timescale in zip(self.encoders, self.timescales):
            # Scale times by timescale
            scaled_times = times / timescale
            encoded = encoder(scaled_times, values)
            encoded_features.append(encoded)

        # Fuse timescales
        combined = torch.cat(encoded_features, dim=-1)
        return self.fusion(combined)
