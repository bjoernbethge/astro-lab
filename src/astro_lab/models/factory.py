"""Model factory aligned with ``examples/train_gaia_real_data.py``."""

from __future__ import annotations

from typing import Any, Optional

from astro_lab.models.astro_model import AstroModel


def create_model(
    model_type: str = "astro_model",
    in_channels: Optional[int] = None,
    hidden_channels: int = 128,
    out_channels: Optional[int] = None,
    num_layers: int = 3,
    conv_type: str = "gcn",
    dropout: float = 0.2,
    task: str = "node_classification",
    learning_rate: float = 1e-3,
    **kwargs: Any,
) -> AstroModel:
    """Instantiate a Lightning :class:`AstroModel` from high-level hyperparameters."""
    if model_type != "astro_model":
        raise ValueError(
            f"Unsupported model_type {model_type!r}; only 'astro_model' is supported."
        )
    if in_channels is None or out_channels is None:
        raise ValueError("in_channels and out_channels are required")

    return AstroModel(
        num_features=in_channels,
        hidden_dim=hidden_channels,
        num_classes=out_channels,
        num_layers=num_layers,
        conv_type=conv_type,
        dropout=dropout,
        task=task,
        learning_rate=learning_rate,
        **kwargs,
    )
