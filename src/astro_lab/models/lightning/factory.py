"""
Lightning Model Factory
======================

Factory functions for creating Lightning-wrapped AstroLab models
with a unified interface.
"""

import logging
from typing import Any, Dict, Optional, Type

from .wrappers import (
    LightningALCDEFTemporalGNN,
    LightningAsteroidPeriodDetector,
    LightningAstroPhotGNN,
    LightningAstroSurveyGNN,
    LightningGaiaClassifier,
    LightningGalaxyModeler,
    LightningTemporalGCN,
    LightningTransientClassifier,
)

logger = logging.getLogger(__name__)

# Registry of available Lightning models
LIGHTNING_MODELS: Dict[str, Type] = {
    # Core models
    "survey_gnn": LightningAstroSurveyGNN,
    "photo_gnn": LightningAstroPhotGNN,
    "temporal_gcn": LightningTemporalGCN,
    "alcdef_temporal": LightningALCDEFTemporalGNN,
    # Specialized models for common tasks
    "gaia_classifier": LightningGaiaClassifier,
    "galaxy_modeler": LightningGalaxyModeler,
    "asteroid_period": LightningAsteroidPeriodDetector,
    "transient_classifier": LightningTransientClassifier,
    # Aliases for compatibility with existing factory
    "lightcurve_classifier": LightningALCDEFTemporalGNN,
    "stellar_classifier": LightningGaiaClassifier,
    "photometry_model": LightningAstroPhotGNN,
}


def create_lightning_model(model_name: str, **kwargs) -> Any:
    """
    Create a Lightning-wrapped AstroLab model.

    Args:
        model_name: Name of the model to create
        **kwargs: Model and Lightning parameters

    Returns:
        Lightning-wrapped model instance

    Raises:
        ValueError: If model_name is not supported

    Example:
        >>> model = create_lightning_model(
        ...     "gaia_classifier",
        ...     num_classes=3,
        ...     learning_rate=0.001,
        ...     optimizer="adamw"
        ... )
        >>> trainer.fit(model, datamodule)
    """
    if model_name not in LIGHTNING_MODELS:
        available = list(LIGHTNING_MODELS.keys())
        raise ValueError(
            f"Unknown Lightning model: {model_name}. Available models: {available}"
        )

    model_class = LIGHTNING_MODELS[model_name]

    try:
        return model_class(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create {model_name} with kwargs {kwargs}: {e}")
        raise


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a Lightning model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information
    """
    if model_name not in LIGHTNING_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = LIGHTNING_MODELS[model_name]

    # Extract docstring and signature info
    import inspect

    info = {
        "name": model_name,
        "class": model_class.__name__,
        "description": model_class.__doc__.split("\n")[0]
        if model_class.__doc__
        else "",
        "parameters": {},
    }

    # Get __init__ signature
    try:
        sig = inspect.signature(model_class.__init__)
        for param_name, param in sig.parameters.items():
            if param_name != "self":
                param_info = {
                    "type": param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else "Any",
                    "default": param.default
                    if param.default != inspect.Parameter.empty
                    else None,
                }
                info["parameters"][param_name] = param_info
    except Exception as e:
        logger.warning(f"Could not extract signature for {model_name}: {e}")

    return info


def list_lightning_models() -> Dict[str, str]:
    """
    List all available Lightning models with descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    models = {}

    for name, model_class in LIGHTNING_MODELS.items():
        description = "Lightning-wrapped AstroLab model"
        if model_class.__doc__:
            # Extract first line of docstring
            description = model_class.__doc__.split("\n")[0].strip()

        models[name] = description

    return models


# Convenience functions for common model types
def create_gaia_classifier(
    num_classes: int = 3, learning_rate: float = 0.001, **kwargs
) -> LightningGaiaClassifier:
    """Create a Lightning Gaia stellar classifier."""
    return create_lightning_model(
        "gaia_classifier",
        num_classes=num_classes,
        learning_rate=learning_rate,
        **kwargs,
    )


def create_galaxy_modeler(
    learning_rate: float = 0.0005, hidden_dim: int = 512, **kwargs
) -> LightningGalaxyModeler:
    """Create a Lightning galaxy modeling system."""
    return create_lightning_model(
        "galaxy_modeler", learning_rate=learning_rate, hidden_dim=hidden_dim, **kwargs
    )


def create_asteroid_detector(
    learning_rate: float = 0.001, **kwargs
) -> LightningAsteroidPeriodDetector:
    """Create a Lightning asteroid period detector."""
    return create_lightning_model(
        "asteroid_period", learning_rate=learning_rate, **kwargs
    )


def create_transient_classifier(
    num_classes: int = 5, learning_rate: float = 0.0008, **kwargs
) -> LightningTransientClassifier:
    """Create a Lightning transient classifier."""
    return create_lightning_model(
        "transient_classifier",
        num_classes=num_classes,
        learning_rate=learning_rate,
        **kwargs,
    )


# Model presets for common configurations
MODEL_PRESETS = {
    "gaia_fast": {
        "model_name": "gaia_classifier",
        "hidden_dim": 128,
        "num_gnn_layers": 2,
        "learning_rate": 0.002,
        "scheduler": "onecycle",
    },
    "gaia_accurate": {
        "model_name": "gaia_classifier",
        "hidden_dim": 512,
        "num_gnn_layers": 4,
        "learning_rate": 0.0005,
        "scheduler": "cosine",
        "warmup_epochs": 10,
    },
    "galaxy_small": {
        "model_name": "galaxy_modeler",
        "hidden_dim": 256,
        "num_gnn_layers": 3,
        "learning_rate": 0.001,
    },
    "galaxy_large": {
        "model_name": "galaxy_modeler",
        "hidden_dim": 1024,
        "num_gnn_layers": 6,
        "learning_rate": 0.0003,
        "scheduler": "cosine",
    },
    "asteroid_quick": {
        "model_name": "asteroid_period",
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 0.002,
    },
    "transient_detection": {
        "model_name": "transient_classifier",
        "hidden_dim": 512,
        "num_temporal_layers": 3,
        "num_gnn_layers": 3,
        "learning_rate": 0.001,
        "use_attention": True,
    },
}


def create_preset_model(preset_name: str, **override_kwargs) -> Any:
    """
    Create a model using a predefined preset configuration.

    Args:
        preset_name: Name of the preset configuration
        **override_kwargs: Parameters to override in the preset

    Returns:
        Lightning model instance

    Example:
        >>> model = create_preset_model("gaia_accurate", num_classes=5)
    """
    if preset_name not in MODEL_PRESETS:
        available = list(MODEL_PRESETS.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. Available presets: {available}"
        )

    config = MODEL_PRESETS[preset_name].copy()
    model_name = config.pop("model_name")

    # Override with provided kwargs
    config.update(override_kwargs)

    return create_lightning_model(model_name, **config)


def list_presets() -> Dict[str, Dict[str, Any]]:
    """List all available model presets."""
    return MODEL_PRESETS.copy()
