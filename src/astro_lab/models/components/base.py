"""
TensorDict-Native Base Components for AstroLab Models
===================================================

This module provides foundational neural network components designed
specifically for the AstroLab TensorDict system. All components expect
TensorDict inputs and are optimized for astronomical data processing.

Key Components:
- TensorDictFeatureProcessor: Extract features from TensorDict structures
- BaseGNNLayer: Graph neural network layer with TensorDict support
- AstroGCNLayer: Specialized GCN for astronomical graphs
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch_geometric.nn import global_mean_pool

from .layers import create_conv_layer


class DeviceMixin:
    """Simple device management mixin."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def to_device(self, *tensors):
        """Move tensors to device."""
        return [t.to(self.device) if t is not None else None for t in tensors]


class GraphProcessor(nn.Module):
    """Simple graph processing component."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        conv_type: str = "gcn",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                create_conv_layer(conv_type, hidden_dim, hidden_dim, **kwargs)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Process graph through conv layers."""
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        return h


class TensorDictFeatureProcessor(nn.Module):
    """
    TensorDict-native feature extraction processor.

    Designed to extract and process features from various TensorDict types
    used in astronomical data analysis.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim * 2

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Feature extraction networks for different TensorDict types
        self.feature_processors = nn.ModuleDict()

        # Will be built dynamically based on input TensorDict structure
        self.is_initialized = False

        self.to(self.device)

    def _initialize_processors(self, data: TensorDict):
        """Initialize processors based on TensorDict structure."""
        if self.is_initialized:
            return

        feature_dim = 0

        # Survey data processor
        if "photometric" in data:
            phot_data = data["photometric"]
            if isinstance(phot_data, TensorDict) and "magnitudes" in phot_data:
                mag_dim = phot_data["magnitudes"].shape[-1]
                # Add space for colors and errors
                phot_feature_dim = mag_dim * 2 if mag_dim > 1 else mag_dim
                if "errors" in phot_data:
                    phot_feature_dim += mag_dim

                self.feature_processors["photometric"] = nn.Sequential(
                    nn.Linear(phot_feature_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                feature_dim += self.hidden_dim // 2

        if "spatial" in data:
            spatial_data = data["spatial"]
            if isinstance(spatial_data, TensorDict) and "coordinates" in spatial_data:
                coord_dim = spatial_data["coordinates"].shape[-1]
                self.feature_processors["spatial"] = nn.Sequential(
                    nn.Linear(coord_dim, self.hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                feature_dim += self.hidden_dim // 4

        if "spectral" in data:
            spectral_data = data["spectral"]
            if isinstance(spectral_data, TensorDict) and "flux" in spectral_data:
                flux_dim = spectral_data["flux"].shape[-1]
                self.feature_processors["spectral"] = nn.Sequential(
                    nn.Linear(flux_dim, self.hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                feature_dim += self.hidden_dim // 4

        # Lightcurve data processor
        if "times" in data and "magnitudes" in data:
            # For lightcurve data - use LSTM
            self.feature_processors["lightcurve"] = nn.LSTM(
                input_size=2,  # time + magnitude
                hidden_size=self.hidden_dim // 2,
                batch_first=True,
            )
            feature_dim += self.hidden_dim // 2

        # Final projection layer
        if feature_dim > 0:
            self.final_projection = nn.Sequential(
                nn.Linear(feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            # Fallback for unknown structure
            self.final_projection = nn.Linear(1, self.output_dim)

        self.feature_processors = self.feature_processors.to(self.device)
        self.final_projection = self.final_projection.to(self.device)
        self.is_initialized = True

    def forward(self, data: TensorDict) -> torch.Tensor:
        """Extract features from TensorDict."""
        if not isinstance(data, TensorDict):
            raise ValueError("TensorDictFeatureProcessor requires TensorDict input")

        # Initialize processors if needed
        if not self.is_initialized:
            self._initialize_processors(data)

        features = []

        # Process photometric data
        if "photometric" in data and "photometric" in self.feature_processors:
            phot_data = data["photometric"]
            if isinstance(phot_data, TensorDict) and "magnitudes" in phot_data:
                magnitudes = phot_data["magnitudes"].to(self.device)

                # Prepare photometric features
                phot_features = [magnitudes]

                # Add colors if multiple bands
                if magnitudes.shape[-1] > 1:
                    colors = magnitudes[..., :-1] - magnitudes[..., 1:]
                    phot_features.append(colors)

                # Add errors if available
                if "errors" in phot_data:
                    errors = phot_data["errors"].to(self.device)
                    phot_features.append(errors)

                combined_phot = torch.cat(phot_features, dim=-1)

                # Handle NaN values
                if torch.isnan(combined_phot).any():
                    combined_phot = torch.nan_to_num(combined_phot, nan=0.0)

                phot_encoded = self.feature_processors["photometric"](combined_phot)
                features.append(phot_encoded)

        # Process spatial data
        if "spatial" in data and "spatial" in self.feature_processors:
            spatial_data = data["spatial"]
            if isinstance(spatial_data, TensorDict) and "coordinates" in spatial_data:
                coordinates = spatial_data["coordinates"].to(self.device)
                spatial_encoded = self.feature_processors["spatial"](coordinates)
                features.append(spatial_encoded)

        # Process spectral data
        if "spectral" in data and "spectral" in self.feature_processors:
            spectral_data = data["spectral"]
            if isinstance(spectral_data, TensorDict) and "flux" in spectral_data:
                flux = spectral_data["flux"].to(self.device)
                # Handle different flux dimensions
                if flux.dim() > 2:
                    flux = flux.mean(dim=-1)
                spectral_encoded = self.feature_processors["spectral"](flux)
                features.append(spectral_encoded)

        # Process lightcurve data
        if (
            "times" in data
            and "magnitudes" in data
            and "lightcurve" in self.feature_processors
        ):
            times = data["times"].to(self.device)
            magnitudes = data["magnitudes"].to(self.device)

            # Prepare sequence
            if times.dim() == 1:
                times = times.unsqueeze(0).unsqueeze(-1)
                magnitudes = magnitudes.unsqueeze(0).unsqueeze(-1)
            elif times.dim() == 2:
                times = times.unsqueeze(-1)
                magnitudes = magnitudes.unsqueeze(-1)

            sequence = torch.cat([times, magnitudes], dim=-1)
            lstm_out, (h_n, _) = self.feature_processors["lightcurve"](sequence)
            lc_encoded = h_n[-1]  # Use final hidden state
            features.append(lc_encoded)

        if not features:
            # Fallback for unknown TensorDict structure
            dummy_input = torch.zeros(1, 1, device=self.device)
            if hasattr(data, "batch_size"):
                dummy_input = dummy_input.expand(data.batch_size[0], -1)
            return self.final_projection(dummy_input)

        # Combine all features
        combined_features = torch.cat(features, dim=-1)

        # Final projection
        return self.final_projection(combined_features)


class BaseGNNLayer(nn.Module):
    """
    Base Graph Neural Network layer with TensorDict support.

    Provides common functionality for graph-based processing of
    astronomical data stored in TensorDict format.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        normalization: str = "batch",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim, output_dim)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        # Normalization
        if normalization == "batch":
            self.norm = nn.BatchNorm1d(output_dim)
        elif normalization == "layer":
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Main transformation
        self.transform = nn.Linear(input_dim, output_dim)

        self.to(self.device)

    def forward(
        self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through GNN layer."""
        # Apply transformation
        x = self.transform(x.to(self.device))

        # Apply normalization
        if hasattr(self.norm, "weight"):  # BatchNorm
            if x.dim() == 2:
                x = self.norm(x)
            elif x.dim() == 3:
                # (batch, seq, features) -> (batch, features, seq)
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)
        else:  # LayerNorm or Identity
            x = self.norm(x)

        # Apply activation
        x = self.activation(x)

        # Apply dropout
        x = self.dropout(x)

        return x


class AstroGCNLayer(BaseGNNLayer):
    """
    Astronomical Graph Convolutional Network layer.

    Specialized GCN layer designed for astronomical graph data
    with support for various edge types and node features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_edge_types: int = 1,
        use_edge_features: bool = False,
        self_loops: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim, output_dim, **kwargs)

        self.num_edge_types = num_edge_types
        self.use_edge_features = use_edge_features
        self.self_loops = self_loops

        # Multiple transformation matrices for different edge types
        if num_edge_types > 1:
            self.edge_transforms = nn.ModuleList(
                [nn.Linear(input_dim, output_dim) for _ in range(num_edge_types)]
            )
        else:
            self.edge_transforms = nn.ModuleList([self.transform])

        # Edge feature processing
        if use_edge_features:
            self.edge_processor = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, output_dim),
            )

        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with graph convolution."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        if edge_type is not None:
            edge_type = edge_type.to(self.device)
        if edge_features is not None:
            edge_features = edge_features.to(self.device)

        # Initialize output
        out = torch.zeros(x.size(0), self.output_dim, device=self.device)

        # Process different edge types
        for i, transform in enumerate(self.edge_transforms):
            if edge_type is not None:
                # Filter edges by type
                edge_mask = edge_type == i
                if not edge_mask.any():
                    continue
                current_edges = edge_index[:, edge_mask]
            else:
                current_edges = edge_index

            if current_edges.size(1) == 0:
                continue

            # Message passing
            row, col = current_edges
            messages = transform(x[col])

            # Include edge features if available
            if self.use_edge_features and edge_features is not None:
                if edge_type is not None:
                    edge_feats = edge_features[edge_mask]
                else:
                    edge_feats = edge_features
                edge_processed = self.edge_processor(edge_feats)
                messages = messages + edge_processed

            # Aggregate messages
            out.index_add_(0, row, messages)

        # Add self-loops
        if self.self_loops:
            out = out + self.transform(x)

        # Apply normalization, activation, dropout
        if hasattr(self.norm, "weight"):  # BatchNorm
            out = self.norm(out.unsqueeze(0)).squeeze(0)
        else:  # LayerNorm or Identity
            out = self.norm(out)

        out = self.activation(out)
        out = self.dropout(out)

        return out


class PoolingModule(nn.Module):
    """Simple pooling module for graph-level features."""

    def __init__(self, pooling_type: str = "mean"):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(
        self, x: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool node features to graph level."""
        # Handle edge cases for input tensor
        if x.numel() == 0:
            # Empty tensor - return a valid shaped tensor
            return torch.zeros(
                1, x.size(-1) if x.dim() > 0 else 1, device=x.device, dtype=x.dtype
            )

        if batch is None:
            # If no batch, assume single graph
            # Ensure x has at least 2 dimensions for pooling
            if x.dim() == 0:
                # Scalar - convert to [1, 1]
                x = x.view(1, 1)
            elif x.dim() == 1:
                # 1D tensor - convert to [1, features] or [nodes, 1]
                if x.size(0) == 1:
                    x = x.view(1, 1)  # Single value
                else:
                    x = x.unsqueeze(-1)  # Multiple nodes, single feature

            # Apply pooling
            if self.pooling_type == "mean":
                result = x.mean(dim=0, keepdim=True)
            elif self.pooling_type == "max":
                result = x.max(dim=0)[0].unsqueeze(0)
            elif self.pooling_type == "sum":
                result = x.sum(dim=0, keepdim=True)
            else:
                # Default to mean pooling
                result = x.mean(dim=0, keepdim=True)

            # Ensure result has proper dimensions
            if result.dim() == 0:
                result = result.unsqueeze(0)
            if result.dim() == 1 and result.size(0) > 1:
                result = result.unsqueeze(0)

            return result
        else:
            # Use PyG pooling with error handling
            try:
                if self.pooling_type == "mean":
                    from torch_geometric.nn import global_mean_pool

                    result = global_mean_pool(x, batch)
                elif self.pooling_type == "max":
                    from torch_geometric.nn import global_max_pool

                    result = global_max_pool(x, batch)
                elif self.pooling_type == "sum":
                    from torch_geometric.nn import global_add_pool

                    result = global_add_pool(x, batch)
                else:
                    # Default to mean pooling
                    from torch_geometric.nn import global_mean_pool

                    result = global_mean_pool(x, batch)

                # Ensure result has proper batch dimension
                if result.dim() == 1:
                    result = result.unsqueeze(0)
                elif result.dim() == 0:
                    result = result.view(1, 1)

                return result
            except Exception:
                # Fallback to manual pooling if PyG pooling fails
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return x.mean(dim=0, keepdim=True)
