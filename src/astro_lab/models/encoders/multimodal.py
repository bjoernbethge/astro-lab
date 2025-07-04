"""
Multimodal Encoders
==================

TensorDict-based fusion of multiple astronomical data modalities.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule


class MultiModalFusionModule(TensorDictModule):
    """
    TensorDict module for fusing multiple data modalities.

    Combines spatial, photometric, spectral, and temporal features.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int = 256,
        output_dim: Optional[int] = None,
        fusion_type: str = "attention",
        dropout: float = 0.1,
        in_key_suffix: str = "_features",
        out_key: str = "multimodal_features",
    ):
        output_dim = output_dim or fusion_dim

        # Build input keys from modality names
        in_keys = [f"{modality}{in_key_suffix}" for modality in modality_dims.keys()]

        # Create fusion module
        fusion = MultiModalFusion(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            fusion_type=fusion_type,
            dropout=dropout,
        )

        super().__init__(
            module=fusion,
            in_keys=in_keys,
            out_keys=[out_key],
        )


class MultiModalFusion(nn.Module):
    """Core multimodal fusion network."""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int = 256,
        output_dim: int = 256,
        fusion_type: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.modality_names = list(modality_dims.keys())

        # Modality-specific projections
        self.modality_projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name, dim in modality_dims.items()
            }
        )

        # Fusion mechanism
        if fusion_type == "attention":
            self.fusion = CrossModalAttention(fusion_dim, num_heads=8, dropout=dropout)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(fusion_dim, len(modality_dims), dropout=dropout)
        elif fusion_type == "hierarchical":
            self.fusion = HierarchicalFusion(
                fusion_dim, len(modality_dims), dropout=dropout
            )
        else:
            # Simple concatenation + MLP
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim * len(modality_dims), fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, *modality_features) -> torch.Tensor:
        """Fuse multiple modality features."""
        # Project each modality
        projected = []

        for i, (name, features) in enumerate(
            zip(self.modality_names, modality_features)
        ):
            if features is not None:
                proj = self.modality_projections[name](features)
                projected.append(proj)

        if not projected:
            raise ValueError("At least one modality must be provided")

        # Apply fusion
        if self.fusion_type == "attention":
            fused = self.fusion(projected)
        elif self.fusion_type in ["gated", "hierarchical"]:
            fused = self.fusion(torch.stack(projected, dim=1))
        else:
            fused = self.fusion(torch.cat(projected, dim=-1))

        # Output projection
        return self.output_net(fused)


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Self-attention over modalities
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(10, dim) * 0.1  # Max 10 modalities
        )

    def forward(self, modality_list: List[torch.Tensor]) -> torch.Tensor:
        """Apply cross-modal attention."""
        # Stack modalities [batch, num_modalities, dim]
        stacked = torch.stack(modality_list, dim=1)
        batch_size, num_modalities, dim = stacked.shape

        # Add modality embeddings
        embeddings = self.modality_embeddings[:num_modalities].unsqueeze(0)
        stacked = stacked + embeddings

        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)

        # Average pool over modalities
        return attended.mean(dim=1)


class GatedFusion(nn.Module):
    """Gated fusion with learnable weights per modality."""

    def __init__(self, dim: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.num_modalities = num_modalities

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Sigmoid(),
        )

        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        """
        Apply gated fusion.

        Args:
            modalities: [batch, num_modalities, dim]
        """
        batch_size, num_mod, dim = modalities.shape

        # Compute gates
        flattened = modalities.view(batch_size, -1)
        gates = self.gate_net(flattened).unsqueeze(-1)  # [batch, num_mod, 1]

        # Apply gates and transform
        gated = modalities * gates

        # Sum over modalities
        fused = gated.sum(dim=1)

        # Transform
        return self.transform(fused)


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion that combines modalities in stages."""

    def __init__(self, dim: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.num_modalities = num_modalities

        # Pairwise fusion layers
        self.pairwise_fusions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_modalities - 1)
            ]
        )

    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical fusion.

        Args:
            modalities: [batch, num_modalities, dim]
        """
        batch_size, num_mod, dim = modalities.shape

        # Start with first modality
        fused = modalities[:, 0]

        # Hierarchically fuse remaining modalities
        for i in range(1, num_mod):
            if i - 1 < len(self.pairwise_fusions):
                combined = torch.cat([fused, modalities[:, i]], dim=-1)
                fused = self.pairwise_fusions[i - 1](combined)
            else:
                # Simple average if we run out of fusion layers
                fused = (fused + modalities[:, i]) / 2

        return fused


class AstronomicalContextFusionModule(TensorDictModule):
    """
    Specialized fusion module that incorporates astronomical context.

    Considers physical relationships between modalities.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int = 256,
        use_redshift: bool = True,
        use_distance: bool = True,
        in_keys: Optional[List[str]] = None,
        out_key: str = "contextualized_features",
    ):
        # Default input keys
        if in_keys is None:
            in_keys = [f"{mod}_features" for mod in modality_dims.keys()]
            if use_redshift:
                in_keys.append("redshift")
            if use_distance:
                in_keys.append("distance")

        # Create context-aware fusion
        fusion = AstronomicalContextFusion(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            use_redshift=use_redshift,
            use_distance=use_distance,
        )

        super().__init__(
            module=fusion,
            in_keys=in_keys,
            out_keys=[out_key],
        )


class AstronomicalContextFusion(nn.Module):
    """Context-aware fusion using astronomical priors."""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        fusion_dim: int = 256,
        use_redshift: bool = True,
        use_distance: bool = True,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.use_redshift = use_redshift
        self.use_distance = use_distance

        # Base multimodal fusion
        self.base_fusion = MultiModalFusion(
            modality_dims=modality_dims,
            fusion_dim=fusion_dim,
            output_dim=fusion_dim,
            fusion_type="attention",
        )

        # Context encoders
        context_dim = 0
        if use_redshift:
            self.redshift_encoder = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Linear(32, 64),
            )
            context_dim += 64

        if use_distance:
            self.distance_encoder = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Linear(32, 64),
            )
            context_dim += 64

        # Context integration
        self.context_integration = nn.Sequential(
            nn.Linear(fusion_dim + context_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, *args) -> torch.Tensor:
        """Fuse with astronomical context."""
        # Separate modalities and context
        n_modalities = len(self.modality_dims)
        modality_features = args[:n_modalities]

        # Base fusion
        fused = self.base_fusion(*modality_features)

        # Add context
        context_features = []
        idx = n_modalities

        if self.use_redshift and idx < len(args):
            redshift = args[idx].unsqueeze(-1) if args[idx].dim() == 1 else args[idx]
            z_features = self.redshift_encoder(redshift)
            context_features.append(z_features)
            idx += 1

        if self.use_distance and idx < len(args):
            distance = args[idx].unsqueeze(-1) if args[idx].dim() == 1 else args[idx]
            d_features = self.distance_encoder(torch.log(distance + 1))
            context_features.append(d_features)

        # Integrate context
        if context_features:
            all_features = [fused] + context_features
            combined = torch.cat(all_features, dim=-1)
            return self.context_integration(combined)
        else:
            return fused
