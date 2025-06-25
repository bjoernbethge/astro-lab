"""Simple model configuration with dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Simple model configuration."""

    name: str
    hidden_dim: int = 128
    num_layers: int = 3
    conv_type: str = "gcn"
    dropout: float = 0.1
    task: str = "classification"

    # Survey-specific settings
    use_photometry: bool = True
    use_astrometry: bool = True
    use_spectroscopy: bool = False

    # Additional settings
    output_dim: Optional[int] = None
    pooling: str = "mean"
    activation: str = "relu"

    # GAT/Transformer specific
    num_heads: int = 8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "conv_type": self.conv_type,
            "dropout": self.dropout,
            "task": self.task,
            "use_photometry": self.use_photometry,
            "use_astrometry": self.use_astrometry,
            "use_spectroscopy": self.use_spectroscopy,
            "output_dim": self.output_dim,
            "pooling": self.pooling,
            "activation": self.activation,
            "num_heads": self.num_heads,
        }


# Simple predefined configs
CONFIGS = {
    "gaia_classifier": ModelConfig(
        name="gaia_classifier",
        conv_type="gat",
        task="classification",
        use_spectroscopy=False,
    ),
    "sdss_galaxy": ModelConfig(
        name="sdss_galaxy",
        conv_type="transformer",
        hidden_dim=256,
        use_spectroscopy=True,
        task="regression",
    ),
    "lsst_transient": ModelConfig(
        name="lsst_transient",
        conv_type="sage",
        task="classification",
        pooling="max",
    ),
    "asteroid_period": ModelConfig(
        name="asteroid_period",
        conv_type="gcn",
        task="period_detection",
        use_photometry=False,
        use_astrometry=False,
        use_spectroscopy=False,
    ),
}


def get_predefined_config(name: str) -> ModelConfig:
    """Get predefined config by name."""
    if name not in CONFIGS:
        available = list(CONFIGS.keys())
        raise ValueError(f"Unknown config: {name}. Available: {available}")

    # Return a copy to avoid modifications
    config = CONFIGS[name]
    return ModelConfig(**config.to_dict())


def list_predefined_configs() -> list[str]:
    """List available predefined configs."""
    return list(CONFIGS.keys())


# Backward compatibility exports
def get_config(name: str) -> ModelConfig:
    """Alias for get_predefined_config."""
    return get_predefined_config(name)


# Keep some of the old exports for compatibility
EncoderConfig = Dict[str, Any]  # Type alias for compatibility
GraphConfig = Dict[str, Any]
OutputConfig = Dict[str, Any]
TrainingConfig = Dict[str, Any]
PREDEFINED_CONFIGS = CONFIGS  # Alias for compatibility
