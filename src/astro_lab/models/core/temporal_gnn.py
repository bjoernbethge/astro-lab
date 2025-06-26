"""
TensorDict-Native Temporal GNN Models
====================================

Time-series GNN models for astronomical lightcurve analysis
using native TensorDict methods and time properties.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

# Import our TensorDict classes to use their methods
from astro_lab.tensors.tensordict_astro import LightcurveTensorDict

from ..components.base import BaseGNNLayer, TensorDictFeatureProcessor
from ..encoders import LightcurveEncoder


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network using native LightcurveTensorDict methods.

    Processes time-series astronomical data through temporal modeling and GNN layers,
    utilizing the native time-domain methods of LightcurveTensorDict.
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_temporal_layers: int = 2,
        num_gnn_layers: int = 2,
        temporal_model: str = "lstm",
        dropout: float = 0.1,
        use_attention: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_temporal_layers = num_temporal_layers
        self.num_gnn_layers = num_gnn_layers
        self.temporal_model = temporal_model
        self.dropout = dropout
        self.use_attention = use_attention

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Native LightcurveTensorDict encoder
        self.lightcurve_encoder = LightcurveEncoder(
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout,
            device=device,
            **kwargs,
        )

        # Temporal modeling layers
        if temporal_model == "lstm":
            self.temporal_processor = nn.LSTM(
                input_size=2,  # time + magnitude
                hidden_size=hidden_dim,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0,
            )
        elif temporal_model == "gru":
            self.temporal_processor = nn.GRU(
                input_size=2,
                hidden_size=hidden_dim,
                num_layers=num_temporal_layers,
                batch_first=True,
                dropout=dropout if num_temporal_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unsupported temporal model: {temporal_model}")

        # Attention mechanism for temporal features
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )

        # GNN layers for spatial relationships
        self.gnn_layers = nn.ModuleList(
            [
                BaseGNNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    layer_type="gcn",
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        # Global pooling
        from ..components.base import PoolingModule

        self.pooling = PoolingModule(pooling_type="mean")

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.to(self.device)

    def forward(
        self,
        data: LightcurveTensorDict,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using native LightcurveTensorDict methods.

        Args:
            data: LightcurveTensorDict with native time-domain access
            edge_index: Graph edge indices for spatial relationships
            batch: Batch assignment for lightcurves

        Returns:
            Encoded temporal features
        """
        if not isinstance(data, LightcurveTensorDict):
            raise ValueError("TemporalGCN requires LightcurveTensorDict input")

        # Use native LightcurveTensorDict methods
        times = data["times"].to(self.device)
        magnitudes = data["magnitudes"].to(self.device)

        # Use time_span property for normalization
        if hasattr(data, "time_span"):
            time_span = data.time_span
            # Normalize times to [0, 1] range
            times_normalized = (times - times.min()) / (time_span + 1e-8)
        else:
            times_normalized = times

        # Prepare sequence data for temporal processing
        if times.dim() == 1:
            # Single lightcurve
            sequence = torch.stack(
                [times_normalized, magnitudes.squeeze(-1)], dim=-1
            ).unsqueeze(0)
        else:
            # Multiple lightcurves in batch
            if magnitudes.dim() == 3 and magnitudes.shape[-1] > 1:
                # Use first band only for simplicity
                magnitudes = magnitudes[..., 0]
            sequence = torch.stack([times_normalized, magnitudes], dim=-1)

        # Process through temporal model
        if self.temporal_model in ["lstm", "gru"]:
            temporal_out, (h_n, _) = self.temporal_processor(sequence)

            # Use last hidden state as node features
            if self.temporal_model == "lstm":
                node_features = h_n[-1]
            else:  # GRU
                node_features = h_n[-1]

            # Apply temporal attention if enabled
            if self.use_attention:
                attended_out, _ = self.temporal_attention(
                    temporal_out, temporal_out, temporal_out
                )
                # Combine last hidden state with attended features
                attention_pooled = attended_out.mean(dim=1)
                node_features = node_features + attention_pooled

        # Create edge index if not provided
        if edge_index is None:
            num_nodes = node_features.shape[0]
            edge_index = self._create_temporal_graph(times, num_nodes)

        edge_index = edge_index.to(self.device)

        # Process through GNN layers
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling
        graph_embedding = self.pooling(h, batch)

        # Final projection
        output = self.output_projection(graph_embedding)

        return output

    def _create_temporal_graph(
        self, times: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Create graph based on temporal proximity using native time methods."""
        if num_nodes <= 1:
            return torch.tensor([[0], [0]], device=self.device, dtype=torch.long)

        # Connect lightcurves that are temporally close
        if times.dim() > 1:
            # Use mean observation time for each lightcurve
            mean_times = times.mean(dim=-1)
        else:
            # Single lightcurve case
            mean_times = times.unsqueeze(0) if num_nodes == 1 else times[:num_nodes]

        # Create temporal proximity graph
        time_distances = torch.abs(mean_times.unsqueeze(1) - mean_times.unsqueeze(0))

        # Connect nodes within temporal threshold
        threshold = time_distances.std() * 0.5  # Adaptive threshold
        adjacency = (time_distances < threshold) & (time_distances > 0)

        # Convert to edge index
        edge_indices = adjacency.nonzero(as_tuple=False).T

        if edge_indices.shape[1] == 0:
            # Fallback: connect consecutive observations
            source = torch.arange(num_nodes - 1, device=self.device)
            target = torch.arange(1, num_nodes, device=self.device)
            edge_indices = torch.stack([source, target])

        return edge_indices

    def extract_period_features(
        self, data: LightcurveTensorDict
    ) -> Dict[str, torch.Tensor]:
        """Extract period-related features using native time methods."""
        if not isinstance(data, LightcurveTensorDict):
            raise ValueError("Requires LightcurveTensorDict input")

        features = {}

        # Basic time statistics
        times = data["times"]
        magnitudes = data["magnitudes"]

        if hasattr(data, "time_span"):
            features["time_span"] = data.time_span

        # Observation statistics
        features["n_observations"] = torch.tensor(len(times), dtype=torch.float32)
        features["time_range"] = times.max() - times.min()
        features["mean_magnitude"] = magnitudes.mean()
        features["magnitude_std"] = magnitudes.std()

        # Sampling statistics
        if len(times) > 1:
            time_diffs = torch.diff(torch.sort(times)[0])
            features["mean_cadence"] = time_diffs.mean()
            features["cadence_std"] = time_diffs.std()

        return features


class ALCDEFTemporalGNN(TemporalGCN):
    """
    Advanced Lightcurve Classification and Detection using Extended Features.

    Specialized temporal GNN for asteroid lightcurve analysis with period detection
    and classification using native LightcurveTensorDict methods.
    """

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_period_candidates: int = 100,
        min_period: float = 0.1,  # hours
        max_period: float = 100.0,  # hours
        use_fourier_features: bool = True,
        **kwargs,
    ):
        super().__init__(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.num_period_candidates = num_period_candidates
        self.min_period = min_period
        self.max_period = max_period
        self.use_fourier_features = use_fourier_features

        # Period search grid
        self.register_buffer(
            "period_grid",
            torch.logspace(
                torch.log10(torch.tensor(min_period)),
                torch.log10(torch.tensor(max_period)),
                num_period_candidates,
            ),
        )

        # Fourier feature extractor
        if use_fourier_features:
            self.fourier_features = nn.Sequential(
                nn.Linear(num_period_candidates, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
            )

        # Period regression head
        self.period_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output normalized period
        )

        # Classification head for lightcurve types
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, 5),  # 5 lightcurve classes
        )

    def forward(
        self,
        data: LightcurveTensorDict,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with period detection using native methods.

        Returns:
            Dict with 'embedding', 'period', 'classification' keys
        """
        if not isinstance(data, LightcurveTensorDict):
            raise ValueError("ALCDEFTemporalGNN requires LightcurveTensorDict input")

        # Get base temporal features
        base_embedding = super().forward(data, edge_index, batch)

        # Period analysis using native time methods
        period_features = self._analyze_periods(data)

        # Combine features
        if self.use_fourier_features and period_features is not None:
            fourier_feats = self.fourier_features(period_features)
            combined_embedding = torch.cat([base_embedding, fourier_feats], dim=-1)

            # Adjust projection to handle concatenated features
            if not hasattr(self, "_adjusted_projection"):
                combined_dim = base_embedding.shape[-1] + fourier_feats.shape[-1]
                self.final_projection = nn.Linear(combined_dim, self.hidden_dim).to(
                    self.device
                )
                self._adjusted_projection = True

            final_embedding = self.final_projection(combined_embedding)
        else:
            final_embedding = base_embedding

        # Generate outputs
        results = {
            "embedding": final_embedding,
            "period": self.period_head(final_embedding),
            "classification": self.classification_head(final_embedding),
        }

        return results

    def _analyze_periods(self, data: LightcurveTensorDict) -> Optional[torch.Tensor]:
        """Analyze periods using native LightcurveTensorDict time methods."""
        times = data["times"].to(self.device)
        magnitudes = data["magnitudes"].to(self.device)

        if len(times) < 10:  # Not enough data for period analysis
            return None

        # Handle multiple bands by using first band
        if magnitudes.dim() > 1 and magnitudes.shape[-1] > 1:
            magnitudes = magnitudes[..., 0]

        # Lomb-Scargle periodogram approximation using FFT
        period_powers = []

        for period in self.period_grid:
            # Phase fold the lightcurve
            phases = (times % period) / period

            # Sort by phase
            sorted_indices = torch.argsort(phases)
            sorted_mags = magnitudes[sorted_indices]

            # Compute variance as period strength measure
            if len(sorted_mags) > 1:
                power = sorted_mags.var()
            else:
                power = torch.tensor(0.0, device=self.device)

            period_powers.append(power)

        period_powers = torch.stack(period_powers)

        # Normalize powers
        if period_powers.max() > period_powers.min():
            period_powers = (period_powers - period_powers.min()) / (
                period_powers.max() - period_powers.min()
            )

        return period_powers.unsqueeze(0)  # Add batch dimension
