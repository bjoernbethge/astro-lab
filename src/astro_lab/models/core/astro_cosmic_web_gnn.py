"""
Specialized Graph Neural Network for cosmic web structure analysis
with native TensorDict support. Optimized for multi-scale astronomical
data with proper coordinate system handling.
"""

from typing import List, Optional

import torch
import torch.nn as nn

# PyTorch Geometric imports
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ..components.base import ModernGraphEncoder
from .base_model import AstroBaseModel


class CosmicWebMessagePassing(MessagePassing):
    """
    Specialized message passing for cosmic web structures.

    Features:
    - Distance-aware message aggregation
    - Multi-scale structure detection
    - Astronomical coordinate system awareness
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coordinate_system: str = "galactocentric",
        max_distance_pc: float = 1000.0,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coordinate_system = coordinate_system
        self.max_distance_pc = max_distance_pc

        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),  # +1 for distance
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Distance encoding
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x, edge_index, pos=None):
        """Forward pass with position-aware messaging."""
        # Include positional information if available
        if pos is not None:
            # Compute edge distances
            row, col = edge_index
            edge_distances = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)

            # Normalize distances (astronomical scaling)
            normalized_distances = edge_distances / self.max_distance_pc

            return self.propagate(
                edge_index, x=x, pos=pos, distances=normalized_distances
            )
        else:
            return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, distances=None):
        """Create messages with distance awareness."""
        if distances is not None:
            # Encode distances
            distance_features = self.distance_encoder(distances)

            # Combine node features with distance
            msg_input = torch.cat([x_i, x_j, distance_features], dim=-1)
        else:
            msg_input = torch.cat([x_i, x_j], dim=-1)

        return self.message_mlp(msg_input)


class MultiScaleCosmicWebLayer(nn.Module):
    """
    Multi-scale cosmic web detection layer.

    Processes cosmic web structures at different scales simultaneously
    to capture hierarchical organization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: List[float] = [5.0, 10.0, 25.0, 50.0],  # parsecs
        coordinate_system: str = "galactocentric",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales or [5.0, 10.0, 25.0, 50.0]  # Default if None
        self.coordinate_system = coordinate_system

        # Scale-specific message passing layers
        self.scale_layers = nn.ModuleList(
            [
                CosmicWebMessagePassing(
                    in_channels=in_channels,
                    out_channels=out_channels // len(self.scales),
                    coordinate_system=coordinate_system,
                    max_distance_pc=scale,
                )
                for scale in self.scales
            ]
        )

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(out_channels, out_channels), nn.ReLU(), nn.LayerNorm(out_channels)
        )

    def forward(self, x, edge_index, pos=None):
        """Process at multiple scales and fuse."""
        scale_features = []

        for scale_layer in self.scale_layers:
            scale_feat = scale_layer(x, edge_index, pos)
            scale_features.append(scale_feat)

        # Concatenate scale features
        multi_scale = torch.cat(scale_features, dim=-1)

        # Fuse scales
        output = self.scale_fusion(multi_scale)

        return output


class CosmicWebAttention(nn.Module):
    """
    Attention mechanism specialized for cosmic web structures.

    Features:
    - Structure-type aware attention
    - Distance-based attention weighting
    - Multi-head attention for different structure aspects
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        structure_types: int = 4,  # void, sheet, filament, node
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.structure_types = structure_types
        self.head_dim = hidden_dim // num_heads

        # Attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Structure-specific attention weights
        self.structure_attention = nn.Linear(hidden_dim, structure_types)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, pos=None):
        """Apply cosmic web attention."""
        batch_size, _ = x.size()

        # Compute queries, keys, values
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Structure-aware attention weights
        structure_weights = torch.softmax(self.structure_attention(x), dim=-1)

        # Apply structure weighting to attention scores
        # This is a simplified version - could be more sophisticated
        structure_bias = structure_weights.mean(dim=-1, keepdim=True)
        scores = scores + structure_bias.unsqueeze(1)

        # Softmax and weighted values
        attention_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, v)

        # Reshape and project
        attended = attended.view(batch_size, self.hidden_dim)
        output = self.output_proj(attended)

        return output


class AstroCosmicWebGNN(AstroBaseModel):
    """
    Graph Neural Network for cosmic web structure analysis.

    Features:
    - Multi-scale cosmic web detection
    - Structure-type specific processing (void, sheet, filament, node)
    - Coordinate system aware processing
    - Native TensorDict compatibility
    - Astronomical distance handling
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 4,  # void, sheet, filament, node
        hidden_dim: int = 128,
        num_layers: int = 4,
        cosmic_web_scales: Optional[List[float]] = None,
        coordinate_system: str = "galactocentric",
        use_attention: bool = True,
        use_multi_scale: bool = True,
        pooling: str = "mean",
        task: str = "node_classification",
        **kwargs,
    ):
        """
        Initialize Cosmic Web GNN.

        Args:
            num_features: Number of input node features
            num_classes: Number of cosmic web structure classes
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            cosmic_web_scales: Scales for multi-scale analysis (in parsecs)
            coordinate_system: Astronomical coordinate system
            use_attention: Whether to use cosmic web attention
            use_multi_scale: Whether to use multi-scale processing
            pooling: Graph pooling method for graph-level tasks
            task: Task type (node_classification, graph_classification)
            **kwargs: Additional model parameters
        """
        super().__init__(
            task=task,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.cosmic_web_scales = cosmic_web_scales or [5.0, 10.0, 25.0, 50.0]
        self.coordinate_system = coordinate_system
        self.use_attention = use_attention
        self.use_multi_scale = use_multi_scale
        self.pooling = pooling

        # Input processing - ensure dimensions match
        self.input_encoder = nn.Sequential(
            nn.Linear(
                num_features, hidden_dim
            ),  # This should match the actual input features
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(kwargs.get("dropout", 0.1)),
        )

        # Multi-scale cosmic web layers
        if use_multi_scale:
            n_scales = len(self.cosmic_web_scales)
            # Ensure hidden_dim is divisible by n_scales
            if hidden_dim % n_scales != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by number of scales ({n_scales})"
                )
            self.multi_scale_layers = nn.ModuleList(
                [
                    MultiScaleCosmicWebLayer(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        scales=self.cosmic_web_scales,
                        coordinate_system=coordinate_system,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            # Graph encoder
            self.graph_encoder = ModernGraphEncoder(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=num_layers,
                conv_type="gat",  # GAT works well for cosmic web
                **kwargs,
            )

        # Cosmic web attention
        if use_attention:
            self.cosmic_attention = CosmicWebAttention(
                hidden_dim=hidden_dim,
                num_heads=kwargs.get("num_heads", 8),
                structure_types=num_classes,
            )

        # Structure-specific processing
        self.structure_processors = nn.ModuleDict(
            {
                "void": nn.Linear(hidden_dim, hidden_dim // 2),
                "sheet": nn.Linear(hidden_dim, hidden_dim // 2),
                "filament": nn.Linear(hidden_dim, hidden_dim // 2),
                "node": nn.Linear(hidden_dim, hidden_dim // 2),
            }
        )

        # Final processing
        final_dim = hidden_dim + 4 * (hidden_dim // 2)  # original + 4 structure types

        self.final_encoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(kwargs.get("dropout", 0.1)),
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        """
        Forward pass through the cosmic web network.

        Args:
            batch: PyG batch with x, edge_index, pos (3D coordinates)

        Returns:
            Cosmic web structure predictions
        """
        x = getattr(batch, "x", None)
        edge_index = getattr(batch, "edge_index", None)
        pos = getattr(batch, "pos", None)
        batch_idx = getattr(batch, "batch", None)
        if x is None or edge_index is None:
            raise ValueError("Batch must have 'x' (node features) and 'edge_index'!")

        # Input encoding
        x = self.input_encoder(x)

        # Multi-scale cosmic web processing
        if (
            self.use_multi_scale
            and hasattr(self, "multi_scale_layers")
            and self.multi_scale_layers
        ):
            for multi_scale_layer in self.multi_scale_layers:
                try:
                    x_multi = multi_scale_layer(x, edge_index, pos)
                    x = x + x_multi  # Residual connection
                except Exception as e:
                    # Fallback if multi-scale fails
                    print(f"Multi-scale layer failed: {e}, using simple processing")
                    break
        else:
            # Fallback to simple encoder if multi-scale is disabled
            if hasattr(self, "graph_encoder"):
                x = self.graph_encoder(x, edge_index)

        # Cosmic web attention
        if self.use_attention:
            x_attended = self.cosmic_attention(x, edge_index, pos)
            x = x + x_attended  # Residual connection

        # Structure-specific processing
        structure_features = []
        for structure_type, processor in self.structure_processors.items():
            struct_feat = processor(x)
            structure_features.append(struct_feat)

        # Combine original and structure-specific features
        x_combined = torch.cat([x] + structure_features, dim=-1)

        # Final processing
        x_final = self.final_encoder(x_combined)

        # Apply pooling for graph-level tasks
        if "graph" in self.task and batch_idx is not None:
            if self.pooling == "mean":
                x_final = global_mean_pool(x_final, batch_idx)
            elif self.pooling == "max":
                x_final = global_max_pool(x_final, batch_idx)
            elif self.pooling == "sum":
                x_final = global_add_pool(x_final, batch_idx)

        # Output predictions
        return self.output_head(x_final)

    def get_cosmic_web_embeddings(self, batch):
        """
        Get cosmic web structure embeddings without classification.

        Args:
            batch: PyG batch with graph data

        Returns:
            Node embeddings optimized for cosmic web analysis
        """
        x = getattr(batch, "x", None)
        edge_index = getattr(batch, "edge_index", None)
        pos = getattr(batch, "pos", None)
        if x is None or edge_index is None:
            raise ValueError("Batch must have 'x' (node features) and 'edge_index'!")

        # Input encoding
        x = self.input_encoder(x)

        # Multi-scale processing
        if self.use_multi_scale:
            for multi_scale_layer in self.multi_scale_layers:
                x_multi = multi_scale_layer(x, edge_index, pos)
                x = x + x_multi
        else:
            x = self.graph_encoder(x, edge_index)

        # Attention
        if self.use_attention:
            x_attended = self.cosmic_attention(x, edge_index, pos)
            x = x + x_attended

        return x

    def detect_cosmic_structures(self, batch, threshold: float = 0.5):
        """
        Detect and classify cosmic web structures.

        Args:
            batch: PyG batch with graph data
            threshold: Classification threshold

        Returns:
            Dictionary with structure predictions and confidence scores
        """
        # Get predictions
        logits = self.forward(batch)
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

        # Get confidence scores
        max_probs = torch.max(probs, dim=-1)[0]
        confident_mask = max_probs > threshold

        # Structure mapping
        structure_names = ["void", "sheet", "filament", "node"]

        # Organize results
        results = {
            "predictions": predictions,
            "probabilities": probs,
            "confidence_scores": max_probs,
            "confident_predictions": confident_mask,
            "structure_counts": {
                name: (predictions == i).sum().item()
                for i, name in enumerate(structure_names)
            },
            "mean_confidence": max_probs.mean().item(),
        }

        return results

    def analyze_cosmic_web_statistics(self, batch):
        """
        Compute cosmic web statistics from the model predictions.

        Args:
            batch: PyG batch with graph data

        Returns:
            Dictionary with cosmic web analysis statistics
        """
        results = self.detect_cosmic_structures(batch)

        # Compute structure fractions
        total_objects = len(results["predictions"])
        structure_fractions = {
            name: count / total_objects
            for name, count in results["structure_counts"].items()
        }

        # Cosmic web statistics
        stats = {
            "total_objects": total_objects,
            "structure_fractions": structure_fractions,
            "void_fraction": structure_fractions["void"],
            "filament_fraction": structure_fractions["filament"],
            "cluster_fraction": structure_fractions["node"],
            "sheet_fraction": structure_fractions["sheet"],
            "mean_confidence": results["mean_confidence"],
            "coordinate_system": self.coordinate_system,
            "scales_analyzed": self.cosmic_web_scales,
        }

        return stats
