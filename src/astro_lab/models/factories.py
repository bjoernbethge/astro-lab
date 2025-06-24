"""Simple factory functions for common models."""

import inspect
from typing import Optional, Union

import torch

from .config import get_predefined_config
from .core import ALCDEFTemporalGNN, AstroPhotGNN, AstroSurveyGNN, TemporalGCN


def filter_kwargs(target_class, **kwargs):
    """Filter kwargs to only include parameters accepted by target_class.__init__."""
    sig = inspect.signature(target_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def create_gaia_classifier(
    num_classes: int = 7,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AstroSurveyGNN:
    """Create Gaia stellar classifier (robust to all kwargs)."""
    config = get_predefined_config("gaia_classifier").to_dict()
    config.update(kwargs)
    config.setdefault("output_dim", num_classes)
    config.setdefault("hidden_dim", hidden_dim)
    config.setdefault("device", device)
    config.setdefault("task", "classification")

    # Filter to only valid AstroSurveyGNN parameters
    filtered = filter_kwargs(AstroSurveyGNN, **config)
    return AstroSurveyGNN(**filtered)


def create_sdss_galaxy_model(
    output_dim: int = 5,
    hidden_dim: int = 256,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AstroSurveyGNN:
    """Create SDSS galaxy property predictor (robust to all kwargs)."""
    config = get_predefined_config("sdss_galaxy").to_dict()
    config.update(kwargs)
    config.setdefault("output_dim", output_dim)
    config.setdefault("hidden_dim", hidden_dim)
    config.setdefault("device", device)
    config.setdefault("task", "regression")

    # Filter to only valid AstroSurveyGNN parameters
    filtered = filter_kwargs(AstroSurveyGNN, **config)
    return AstroSurveyGNN(**filtered)


def create_lsst_transient_detector(
    output_dim: int = 2,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AstroSurveyGNN:
    """Create LSST transient detector (robust to all kwargs)."""
    config = get_predefined_config("lsst_transient").to_dict()
    config.update(kwargs)
    config.setdefault("output_dim", output_dim)
    config.setdefault("hidden_dim", hidden_dim)
    config.setdefault("device", device)
    config.setdefault("task", "classification")

    # Filter to only valid AstroSurveyGNN parameters
    filtered = filter_kwargs(AstroSurveyGNN, **config)
    return AstroSurveyGNN(**filtered)


def create_asteroid_period_detector(
    hidden_dim: int = 128, device: Optional[Union[str, torch.device]] = None, **kwargs
) -> ALCDEFTemporalGNN:
    """Create asteroid period detector (robust to all kwargs)."""
    config = get_predefined_config("asteroid_period").to_dict()
    config.update(kwargs)
    config.setdefault("hidden_dim", hidden_dim)
    config.setdefault("device", device)
    config.setdefault("task", "period_detection")

    # Filter to only valid ALCDEFTemporalGNN parameters
    filtered = filter_kwargs(ALCDEFTemporalGNN, **config)
    return ALCDEFTemporalGNN(**filtered)


def create_lightcurve_classifier(
    num_classes: int = 2,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> ALCDEFTemporalGNN:
    """Create lightcurve classifier (robust to all kwargs)."""
    config = get_predefined_config("asteroid_period").to_dict()
    config.update(kwargs)
    config.setdefault("output_dim", num_classes)
    config.setdefault("hidden_dim", hidden_dim)
    config.setdefault("device", device)
    config.setdefault("num_classes", num_classes)
    # Force classification task (override the period_detection from config)
    config["task"] = "classification"

    # Filter to only valid ALCDEFTemporalGNN parameters
    filtered = filter_kwargs(ALCDEFTemporalGNN, **config)
    return ALCDEFTemporalGNN(**filtered)


def create_galaxy_modeler(
    model_components: Optional[list] = None,
    hidden_dim: int = 128,
    output_dim: int = 12,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AstroPhotGNN:
    """Create galaxy modeler (robust to all kwargs)."""
    config = get_predefined_config("sdss_galaxy").to_dict()
    config.update(kwargs)
    config.setdefault("model_components", model_components or ["sersic", "disk"])
    config.setdefault("output_dim", output_dim)
    config.setdefault("hidden_dim", hidden_dim)
    config.setdefault("device", device)
    config.setdefault("task", "regression")

    # Filter to only valid AstroPhotGNN parameters
    filtered = filter_kwargs(AstroPhotGNN, **config)
    return AstroPhotGNN(**filtered)


def create_temporal_graph_model(
    input_dim: int = 10,
    output_dim: int = 5,
    hidden_dim: int = 128,
    num_graph_layers: int = 3,
    num_rnn_layers: int = 2,
    rnn_type: str = "lstm",
    conv_type: str = "gcn",
    task: str = "regression",
    dropout: float = 0.1,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> TemporalGCN:
    """Create temporal graph model (robust to all kwargs)."""
    config = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "num_graph_layers": num_graph_layers,
        "num_rnn_layers": num_rnn_layers,
        "rnn_type": rnn_type,
        "conv_type": conv_type,
        "task": task,
        "dropout": dropout,
        "device": device,
    }
    config.update(kwargs)

    # Filter to only valid TemporalGCN parameters
    filtered = filter_kwargs(TemporalGCN, **config)
    return TemporalGCN(**filtered)


# Centralized model registry for training
MODELS = {
    "gaia_classifier": create_gaia_classifier,
    "sdss_galaxy": create_sdss_galaxy_model,
    "lsst_transient": create_lsst_transient_detector,
    "asteroid_period": create_asteroid_period_detector,
    "lightcurve_classifier": create_lightcurve_classifier,
    "galaxy_modeler": create_galaxy_modeler,
    "temporal_graph": create_temporal_graph_model,
}


def create_model(model_name: str, **kwargs):
    """Create model by name with robust parameter filtering."""
    if model_name not in MODELS:
        available = list(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    factory_fn = MODELS[model_name]
    return factory_fn(**kwargs)
