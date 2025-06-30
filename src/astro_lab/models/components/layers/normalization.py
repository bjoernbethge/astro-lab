"""
Normalization Layers for AstroLab
=================================

Advanced normalization techniques for astronomical graph data.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import BatchNorm, GraphNorm, InstanceNorm, LayerNorm


class AdaptiveNormalization(nn.Module):
    """
    Adaptive normalization that switches between different norm types.

    Useful for heterogeneous astronomical data where different
    normalization strategies work better for different features.
    """

    def __init(
        self,
        num_features: int,
        norm_types: list[str] = ["batch", "layer", "instance"],
        learnable_weights: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.norm_types = norm_types

        # Create normalization layers
        self.norms = nn.ModuleDict()
        for norm_type in norm_types:
            if norm_type == "batch":
                self.norms[norm_type] = BatchNorm(num_features)
            elif norm_type == "layer":
                self.norms[norm_type] = LayerNorm(num_features)
            elif norm_type == "instance":
                self.norms[norm_type] = InstanceNorm(num_features)
            elif norm_type == "graph":
                self.norms[norm_type] = GraphNorm(num_features)
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")

        # Learnable weights for combining normalizations
        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(len(norm_types)) / len(norm_types))
        else:
            self.register_buffer(
                "weights", torch.ones(len(norm_types)) / len(norm_types)
            )

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """Apply adaptive normalization."""

        # Apply each normalization
        normalized_outputs = []
        for i, norm_type in enumerate(self.norm_types):
            if norm_type == "graph" and batch is not None:
                norm_out = self.norms[norm_type](x, batch)
            elif norm_type == "layer":
                # LayerNorm never gets batch argument
                norm_out = self.norms[norm_type](x)
            else:
                norm_out = self.norms[norm_type](x)
            normalized_outputs.append(norm_out)

        # Combine with learnable weights
        weights = F.softmax(self.weights, dim=0)
        output = sum(w * out for w, out in zip(weights, normalized_outputs))

        return output

    def reset_parameters(self):
        """Reset all normalization parameters."""
        for norm in self.norms.values():
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if isinstance(self.weights, nn.Parameter):
            nn.init.uniform_(self.weights)


class AstronomicalFeatureNorm(nn.Module):
    """
    Feature-specific normalization for astronomical data.

    Different features (magnitudes, colors, positions) may need
    different normalization strategies.
    """

    def __init__(
        self,
        num_features: int,
        feature_groups: Optional[dict[str, list[int]]] = None,
        default_norm: str = "layer",
    ):
        super().__init__()

        self.num_features = num_features
        self.feature_groups = feature_groups or {}
        self.default_norm = default_norm

        # Create norm for each feature group
        self.group_norms = nn.ModuleDict()

        # Track which features belong to which group
        self.feature_to_group = {}

        for group_name, feature_indices in self.feature_groups.items():
            group_size = len(feature_indices)

            if group_name == "magnitudes":
                # Magnitude features often benefit from instance norm
                self.group_norms[group_name] = InstanceNorm(group_size)
            elif group_name == "positions":
                # Positional features often benefit from batch norm
                self.group_norms[group_name] = BatchNorm(group_size)
            elif group_name == "velocities":
                # Velocity features may benefit from layer norm
                self.group_norms[group_name] = LayerNorm(group_size)
            else:
                # Default normalization
                self.group_norms[group_name] = self._create_default_norm(group_size)

            for idx in feature_indices:
                self.feature_to_group[idx] = group_name

        # Default norm for ungrouped features
        ungrouped_features = [
            i for i in range(num_features) if i not in self.feature_to_group
        ]
        if ungrouped_features:
            self.default_norm_layer = self._create_default_norm(len(ungrouped_features))
            self.ungrouped_indices = ungrouped_features

    def _create_default_norm(self, num_features: int) -> nn.Module:
        """Create default normalization layer."""
        if self.default_norm == "batch":
            return BatchNorm(num_features)
        elif self.default_norm == "layer":
            return LayerNorm(num_features)
        elif self.default_norm == "instance":
            return InstanceNorm(num_features)
        else:
            return nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply feature-specific normalization."""

        # Initialize output
        output = torch.zeros_like(x)

        # Apply group-specific normalization
        for group_name, norm in self.group_norms.items():
            feature_indices = self.feature_groups[group_name]
            group_features = x[:, feature_indices]
            normalized = norm(group_features)
            output[:, feature_indices] = normalized

        # Apply default norm to ungrouped features
        if hasattr(self, "ungrouped_indices"):
            ungrouped_features = x[:, self.ungrouped_indices]
            normalized = self.default_norm_layer(ungrouped_features)
            output[:, self.ungrouped_indices] = normalized

        return output

    def reset_parameters(self):
        """Reset all normalization parameters."""
        for norm in self.group_norms.values():
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if hasattr(self, "default_norm_layer"):
            if hasattr(self.default_norm_layer, "reset_parameters"):
                self.default_norm_layer.reset_parameters()


class RobustNormalization(nn.Module):
    """
    Robust normalization for astronomical data with outliers.

    Uses percentile-based normalization to handle extreme values
    common in astronomical measurements.
    """

    def __init__(
        self,
        num_features: int,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.num_features = num_features
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.eps = eps

        # Running statistics
        self.register_buffer("running_lower", torch.zeros(num_features))
        self.register_buffer("running_upper", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

        # Momentum for running stats
        self.momentum = 0.1

    def forward(self, x: Tensor) -> Tensor:
        """Apply robust normalization."""

        if self.training:
            # Compute percentiles
            lower = torch.quantile(x, self.lower_percentile / 100.0, dim=0)
            upper = torch.quantile(x, self.upper_percentile / 100.0, dim=0)

            # Update running statistics
            if self.num_batches_tracked == 0:
                self.running_lower.copy_(lower)
                self.running_upper.copy_(upper)
            else:
                self.running_lower.mul_(1 - self.momentum).add_(lower * self.momentum)
                self.running_upper.mul_(1 - self.momentum).add_(upper * self.momentum)

            self.num_batches_tracked += 1

            # Use current batch statistics
            scale = (upper - lower).clamp(min=self.eps)
            shift = lower
        else:
            # Use running statistics
            scale = (self.running_upper - self.running_lower).clamp(min=self.eps)
            shift = self.running_lower

        # Normalize
        x_normalized = (x - shift) / scale

        # Clip to reasonable range
        x_normalized = torch.clamp(x_normalized, -10, 10)

        return x_normalized

    def reset_parameters(self):
        """Reset running statistics."""
        self.running_lower.zero_()
        self.running_upper.fill_(1.0)
        self.num_batches_tracked.zero_()
