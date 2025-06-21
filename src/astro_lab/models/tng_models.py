"""
TNG-specific Temporal Models

Specialized temporal GNN models for IllustrisTNG cosmological simulations.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from astro_lab.models.tgnn import TemporalGATCNN, TemporalGCN
from astro_lab.models.layers import LayerFactory


class CosmicEvolutionGNN(TemporalGCN):
    """
    Temporal GNN for cosmic evolution in TNG simulations.
    Analyzes galaxy formation, halo growth, and large-scale structure.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        cosmological_features: bool = True,
        redshift_encoding: bool = True,
        **kwargs,
    ):
        """Initialize cosmic evolution GNN with cosmological time encoding."""
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs,
        )

        self.cosmological_features = cosmological_features
        self.redshift_encoding = redshift_encoding

        # Cosmological time encoding
        if redshift_encoding:
            self.redshift_encoder = nn.Sequential(
                LayerFactory.create_mlp(1, hidden_dim // 4),
                nn.ReLU(),
                LayerFactory.create_mlp(hidden_dim // 4, hidden_dim // 4),
            )
            self.time_projection = LayerFactory.create_mlp(hidden_dim + hidden_dim // 4, hidden_dim)

        # Cosmological parameter prediction head
        if cosmological_features:
            self.cosmo_head = nn.Sequential(
                LayerFactory.create_mlp(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                LayerFactory.create_mlp(hidden_dim // 2, 6),  # Omega_m, Omega_L, h, sigma_8, n_s, w
            )

    def encode_snapshot_with_redshift(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        redshift: float,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode snapshot with redshift information."""
        h = self.encode_snapshot(x, edge_index, batch)

        # Add redshift encoding if enabled
        if self.redshift_encoding:
            z_tensor = torch.full((h.size(0), 1), redshift, device=h.device)
            z_encoded = self.redshift_encoder(z_tensor)
            h_combined = torch.cat([h, z_encoded], dim=1)
            h = self.time_projection(h_combined)

        return h

    def forward(
        self,
        snapshot_sequence: List[Dict[str, torch.Tensor]],
        redshifts: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process temporal sequence with cosmological time encoding."""
        graph_embeddings = []

        for i, snapshot in enumerate(snapshot_sequence):
            redshift = redshifts[i] if redshifts else 0.0
            h = self.encode_snapshot_with_redshift(
                snapshot["x"], snapshot["edge_index"], redshift, snapshot.get("batch")
            )
            graph_embeddings.append(h)

        # Temporal processing
        graph_sequence = torch.stack(graph_embeddings, dim=1)
        rnn_out, _ = self.rnn(graph_sequence)
        final_output = rnn_out[:, -1, :]

        predictions = self.output_layer(final_output)
        result = {"predictions": predictions, "embeddings": final_output}

        # Add cosmological parameter predictions
        if self.cosmological_features:
            cosmo_params = self.cosmo_head(final_output)
            result["cosmological_parameters"] = cosmo_params

        return result


class GalaxyFormationGNN(TemporalGCN):
    """
    Temporal GNN for galaxy formation and evolution.
    Predicts stellar mass growth, star formation history, and morphological evolution.
    """

    def __init__(
        self,
        input_dim: int,
        num_galaxy_properties: int = 5,
        environment_dim: int = 32,
        **kwargs,
    ):
        """Initialize galaxy formation GNN."""
        super().__init__(
            input_dim=input_dim, output_dim=num_galaxy_properties, **kwargs
        )

        self.environment_dim = environment_dim
        self.num_galaxy_properties = num_galaxy_properties

        # Environment encoder
        self.env_encoder = nn.Sequential(
            LayerFactory.create_mlp(self.hidden_dim, environment_dim),
            nn.ReLU(),
            LayerFactory.create_mlp(environment_dim, environment_dim),
        )

        # Multi-task heads for galaxy properties
        self.property_heads = nn.ModuleDict(
            {
                "stellar_mass": LayerFactory.create_mlp(self.hidden_dim, 1),
                "sfr": LayerFactory.create_mlp(self.hidden_dim, 1),
                "metallicity": LayerFactory.create_mlp(self.hidden_dim, 1),
                "size": LayerFactory.create_mlp(self.hidden_dim, 1),
                "morphology": LayerFactory.create_mlp(self.hidden_dim, 3),  # disk, bulge, irregular
            }
        )

        # Star formation history prediction
        self.sfh_predictor = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            batch_first=True,
        )

    def predict_galaxy_properties(
        self, embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict galaxy properties from embeddings."""
        return {
            prop_name: head(embeddings)
            for prop_name, head in self.property_heads.items()
        }

    def forward(
        self,
        snapshot_sequence: List[Dict[str, torch.Tensor]],
        predict_properties: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for galaxy formation analysis."""
        base_output = super().forward(snapshot_sequence)

        result = {
            "embeddings": base_output["embeddings"],
            "temporal_evolution": base_output["predictions"],
        }

        if predict_properties:
            properties = self.predict_galaxy_properties(base_output["embeddings"])
            result.update(properties)

        return result


class HaloMergerGNN(TemporalGATCNN):
    """
    Temporal GNN for halo merger analysis using attention mechanisms.
    """

    def __init__(self, input_dim: int, merger_detection: bool = True, **kwargs):
        """Initialize halo merger GNN."""
        super().__init__(input_dim=input_dim, **kwargs)

        self.merger_detection = merger_detection

        if merger_detection:
            # Merger event detector
            self.merger_detector = nn.Sequential(
                LayerFactory.create_mlp(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                LayerFactory.create_mlp(self.hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            # Merger mass ratio predictor
            self.mass_ratio_predictor = nn.Sequential(
                LayerFactory.create_mlp(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                LayerFactory.create_mlp(self.hidden_dim // 2, 1),
            )

    def detect_merger_events(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect merger events from temporal embeddings."""
        merger_probs = self.merger_detector(embeddings)
        mass_ratios = self.mass_ratio_predictor(embeddings)
        return merger_probs, mass_ratios

    def forward(
        self, snapshot_sequence: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with merger detection."""
        base_output = super().forward(snapshot_sequence)

        result = {
            "predictions": base_output["predictions"],
            "embeddings": base_output["embeddings"],
        }

        if self.merger_detection:
            merger_probs, mass_ratios = self.detect_merger_events(
                base_output["embeddings"]
            )
            result.update(
                {"merger_probability": merger_probs, "merger_mass_ratio": mass_ratios}
            )

        return result


class EnvironmentalQuenchingGNN(TemporalGCN):
    """
    Temporal GNN for environmental quenching analysis.
    Studies how large-scale environment affects star formation.
    """

    def __init__(
        self,
        input_dim: int,
        environment_types: int = 4,  # field, group, cluster, void
        **kwargs,
    ):
        """Initialize environmental quenching GNN."""
        super().__init__(input_dim=input_dim, **kwargs)

        self.environment_types = environment_types

        # Environment classifier
        self.env_classifier = nn.Sequential(
            LayerFactory.create_mlp(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            LayerFactory.create_mlp(self.hidden_dim // 2, environment_types),
        )

        # Quenching predictor
        self.quenching_predictor = nn.Sequential(
            LayerFactory.create_mlp(self.hidden_dim + environment_types, self.hidden_dim // 2),
            nn.ReLU(),
            LayerFactory.create_mlp(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Environmental effect encoder
        self.env_effect_encoder = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, dropout=0.1
        )

    def encode_environmental_effects(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Encode environmental effects using attention."""
        env_embeddings, _ = self.env_effect_encoder(embeddings, embeddings, embeddings)
        return env_embeddings + embeddings  # Residual connection

    def forward(
        self, snapshot_sequence: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for environmental quenching analysis."""
        # Process snapshots with environmental encoding
        enhanced_embeddings = []

        for snapshot in snapshot_sequence:
            h = self.encode_snapshot(
                snapshot["x"], snapshot["edge_index"], snapshot.get("batch")
            )
            h_env = self.encode_environmental_effects(h)
            enhanced_embeddings.append(h_env)

        # Temporal processing
        graph_sequence = torch.stack(enhanced_embeddings, dim=1)
        rnn_out, _ = self.rnn(graph_sequence)
        final_embeddings = rnn_out[:, -1, :]

        # Environment classification and quenching prediction
        env_probs = F.softmax(self.env_classifier(final_embeddings), dim=-1)
        quenching_prob = self.quenching_predictor(
            torch.cat([final_embeddings, env_probs], dim=-1)
        )

        predictions = self.output_layer(final_embeddings)

        return {
            "predictions": predictions,
            "embeddings": final_embeddings,
            "environment_type": env_probs,
            "quenching_probability": quenching_prob,
        }


__all__ = [
    "CosmicEvolutionGNN",
    "GalaxyFormationGNN",
    "HaloMergerGNN",
    "EnvironmentalQuenchingGNN",
]
