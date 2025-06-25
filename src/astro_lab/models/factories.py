"""Simple factory functions for common models."""

import inspect
import logging
from typing import Optional, Union

import torch

from .config import get_predefined_config
from .core import ALCDEFTemporalGNN, AstroPhotGNN, AstroSurveyGNN, TemporalGCN
from .utils import filter_kwargs

logger = logging.getLogger(__name__)


def create_gaia_classifier(
    num_classes: Optional[int] = None,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AstroSurveyGNN:
    """
    Create Gaia stellar classifier (robust to all kwargs).

    Args:
        num_classes: Number of output classes (required for classification)
        hidden_dim: Hidden dimension size
        device: Device to place model on
        **kwargs: Additional model parameters

    Returns:
        Configured AstroSurveyGNN model

    Raises:
        ValueError: If num_classes is not provided or invalid
    """
    # Validate num_classes
    if num_classes is None:
        raise ValueError(
            "num_classes must be specified for classification models. "
            "It should be determined from your data (e.g., datamodule.num_classes)"
        )
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    config = get_predefined_config("gaia_classifier").to_dict()
    config.update(kwargs)
    config["output_dim"] = num_classes  # Force the correct number of classes
    config["hidden_dim"] = hidden_dim
    config["device"] = device
    config["task"] = "classification"

    logger.info(f"Creating Gaia classifier with {num_classes} classes")

    # Filter to only valid AstroSurveyGNN parameters
    filtered = filter_kwargs(AstroSurveyGNN, **config)
    return AstroSurveyGNN(**filtered)


def create_sdss_galaxy_model(
    output_dim: Optional[int] = None,
    hidden_dim: int = 256,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AstroSurveyGNN:
    """
    Create SDSS galaxy property predictor (robust to all kwargs).

    Args:
        output_dim: Output dimension (required)
        hidden_dim: Hidden dimension size
        device: Device to place model on
        **kwargs: Additional model parameters

    Returns:
        Configured AstroSurveyGNN model

    Raises:
        ValueError: If output_dim is not provided
    """
    if output_dim is None:
        raise ValueError(
            "output_dim must be specified for regression models. "
            "It depends on the number of properties you want to predict."
        )

    config = get_predefined_config("sdss_galaxy").to_dict()
    config.update(kwargs)
    config["output_dim"] = output_dim
    config["hidden_dim"] = hidden_dim
    config["device"] = device
    config["task"] = "regression"

    # Filter to only valid AstroSurveyGNN parameters
    filtered = filter_kwargs(AstroSurveyGNN, **config)
    return AstroSurveyGNN(**filtered)


def create_lsst_transient_detector(
    num_classes: Optional[int] = None,
    hidden_dim: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> AstroSurveyGNN:
    """
    Create LSST transient detector (robust to all kwargs).

    Args:
        num_classes: Number of output classes (required)
        hidden_dim: Hidden dimension size
        device: Device to place model on
        **kwargs: Additional model parameters

    Returns:
        Configured AstroSurveyGNN model

    Raises:
        ValueError: If num_classes is not provided
    """
    if num_classes is None:
        raise ValueError(
            "num_classes must be specified for classification models. "
            "It should be determined from your data."
        )

    config = get_predefined_config("lsst_transient").to_dict()
    config.update(kwargs)
    config["output_dim"] = num_classes
    config["hidden_dim"] = hidden_dim
    config["device"] = device
    config["task"] = "classification"

    # Filter to only valid AstroSurveyGNN parameters
    filtered = filter_kwargs(AstroSurveyGNN, **config)
    return AstroSurveyGNN(**filtered)


def create_asteroid_period_detector(
    hidden_dim: int = 128, device: Optional[Union[str, torch.device]] = None, **kwargs
) -> ALCDEFTemporalGNN:
    """
    Create asteroid period detector (robust to all kwargs).

    Args:
        hidden_dim: Hidden dimension size
        device: Device to place model on
        **kwargs: Additional model parameters

    Returns:
        Configured ALCDEFTemporalGNN model
    """
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
    """
    Create lightcurve classifier (robust to all kwargs).

    Args:
        num_classes: Number of output classes
        hidden_dim: Hidden dimension size
        device: Device to place model on
        **kwargs: Additional model parameters

    Returns:
        Configured ALCDEFTemporalGNN model
    """
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
    """
    Create galaxy modeler (robust to all kwargs).

    Args:
        model_components: List of model components
        hidden_dim: Hidden dimension size
        output_dim: Output dimension
        device: Device to place model on
        **kwargs: Additional model parameters

    Returns:
        Configured AstroPhotGNN model
    """
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
    """
    Create temporal graph model (robust to all kwargs).

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension size
        num_graph_layers: Number of graph convolution layers
        num_rnn_layers: Number of RNN layers
        rnn_type: Type of RNN ('lstm', 'gru')
        conv_type: Type of graph convolution ('gcn', 'gat', 'sage')
        task: Task type ('regression', 'classification')
        dropout: Dropout rate
        device: Device to place model on
        **kwargs: Additional model parameters

    Returns:
        Configured TemporalGCN model
    """
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
    """
    Create model by name with robust parameter filtering.

    Args:
        model_name: Name of the model to create
        **kwargs: Model parameters

    Returns:
        Configured model instance

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODELS:
        available = list(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    factory_fn = MODELS[model_name]
    return factory_fn(**kwargs)
