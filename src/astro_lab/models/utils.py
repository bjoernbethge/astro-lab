"""
Utility functions for GNN models - consolidated from multiple files.
"""

from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Activation function name

    Returns:
        PyTorch activation module
    """
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    return activations.get(name.lower(), nn.ReLU())


class AttentionPooling(nn.Module):
    """Attention-based global pooling layer for graphs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Pooled features [batch_size, hidden_dim]
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # [num_nodes, 1]

        if batch is None:
            # Single graph case
            attn_weights = torch.softmax(attn_weights, dim=0)
            pooled = torch.sum(attn_weights * x, dim=0, keepdim=True)
        else:
            # Batch case
            attn_weights = torch.softmax(attn_weights, dim=0)
            try:
                from torch_geometric.nn import global_add_pool

                pooled = global_add_pool(attn_weights * x, batch)
            except ImportError:
                # Fallback
                pooled = torch.sum(attn_weights * x, dim=0, keepdim=True)

        return pooled


def get_pooling(name: str, hidden_dim: Optional[int] = None) -> Callable:
    """
    Get pooling function by name.

    Args:
        name: Pooling function name
        hidden_dim: Hidden dimension (needed for attention pooling)

    Returns:
        Pooling function
    """
    try:
        from torch_geometric.nn import (
            global_add_pool,
            global_max_pool,
            global_mean_pool,
        )

        if name == "mean":
            return global_mean_pool
        elif name == "max":
            return global_max_pool
        elif name == "add":
            return global_add_pool
        elif name == "attention" and hidden_dim is not None:
            return _create_attention_pooling(hidden_dim)
        else:
            return global_mean_pool
    except ImportError:
        # Fallback if PyG not available
        return lambda x, batch: torch.mean(x, dim=0, keepdim=True)


def _create_attention_pooling(hidden_dim: int):
    """Create attention pooling function."""
    attention_layer = AttentionPooling(hidden_dim)

    def attention_pool(x: torch.Tensor, batch: Optional[torch.Tensor] = None):
        return attention_layer(x, batch)

    return attention_pool


def initialize_weights(module: nn.Module):
    """
    Initialize model weights using Xavier/Kaiming initialization.

    Args:
        module: PyTorch module to initialize
    """
    for name, param in module.named_parameters():
        if "weight" in name:
            if len(param.shape) >= 2:
                if "conv" in name.lower():
                    nn.init.kaiming_uniform_(param, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
        elif "bias" in name:
            nn.init.zeros_(param)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> dict:
    """
    Get model summary information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "num_layers": len([m for m in model.modules() if len(list(m.children())) == 0]),
    }


# Factory functions for astronomical models
def create_gaia_classifier(
    hidden_dim: int = 128, num_classes: int = 7, **kwargs
) -> Any:
    """Create Gaia stellar classifier."""
    from astro_lab.models.astro import AstroSurveyGNN

    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        use_astrometry=True,
        use_photometry=True,
        use_spectroscopy=False,
        conv_type="gat",  # Attention for high-precision astrometry
        task="node_classification",
        **kwargs,
    )


def create_sdss_galaxy_classifier(
    hidden_dim: int = 128, output_dim: int = 1, **kwargs
) -> Any:
    """Create SDSS galaxy property predictor."""
    from astro_lab.models.astro import AstroSurveyGNN

    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_astrometry=False,
        use_photometry=True,
        use_spectroscopy=True,
        conv_type="transformer",  # Multi-modal integration
        task="node_regression",
        **kwargs,
    )


def create_lsst_transient_detector(hidden_dim: int = 96, **kwargs) -> Any:
    """Create LSST transient detection model."""
    from astro_lab.models.astro import AstroSurveyGNN

    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=1,
        use_astrometry=True,
        use_photometry=True,
        use_spectroscopy=False,
        conv_type="sage",  # Good for time-domain analysis
        task="node_classification",
        **kwargs,
    )


def create_multi_survey_model(
    surveys: List[str], hidden_dim: int = 256, output_dim: int = 1, **kwargs
) -> Any:
    """Create model for multiple surveys."""
    from astro_lab.models.astro import AstroSurveyGNN

    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_astrometry=True,
        use_photometry=True,
        use_spectroscopy=True,  # Support all modalities
        conv_type="transformer",  # Most flexible
        task="node_classification",
        **kwargs,
    )


def compile_astro_model(
    model: Any,
    mode: str = "default",
    dynamic: bool = True,
) -> Any:
    """
    Compile astronomical model for PyTorch 2.x.

    Args:
        model: Model to compile
        mode: Compilation mode
        dynamic: Enable dynamic shapes

    Returns:
        Compiled model
    """
    try:
        return torch.compile(model, mode=mode, dynamic=dynamic)
    except Exception:
        # Fallback if compilation fails
        return model


def create_lightcurve_classifier(
    hidden_dim: int = 128, output_dim: int = 1, **kwargs
) -> Any:
    """Create lightcurve/ALCDEF classifier with LightcurveEncoder."""
    from astro_lab.models.astro import AstroSurveyGNN

    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_astrometry=False,
        use_photometry=False,
        use_spectroscopy=False,
        # Note: Would need to add use_lightcurve=True to AstroSurveyGNN
        conv_type="gat",  # Good for temporal relationships
        task="node_classification",
        **kwargs,
    )


def create_asteroid_period_detector(hidden_dim: int = 96, **kwargs) -> Any:
    """Create asteroid rotation period detector using lightcurve data."""
    from astro_lab.models.astro import AstroSurveyGNN

    return AstroSurveyGNN(
        hidden_dim=hidden_dim,
        output_dim=1,  # Period output
        use_astrometry=False,
        use_photometry=False,
        use_spectroscopy=False,
        conv_type="transformer",  # Good for temporal modeling
        task="node_regression",
        **kwargs,
    )


def create_astrophot_model(
    model_type: str = "sersic+disk",
    hidden_dim: int = 128,
    **kwargs,
) -> Any:
    """Create AstroPhot-integrated model for galaxy modeling."""
    from astro_lab.models.astrophot_models import AstroPhotGNN

    components = {
        "sersic": ["sersic"],
        "disk": ["disk"],
        "sersic+disk": ["sersic", "disk"],
        "bulge+disk": ["bulge", "disk"],
        "full": ["sersic", "disk", "bulge"],
    }

    return AstroPhotGNN(
        model_components=components.get(model_type, ["sersic"]),
        hidden_dim=hidden_dim,
        **kwargs,
    )


def create_nsa_galaxy_modeler(hidden_dim: int = 128, **kwargs) -> Any:
    """Create NSA galaxy modeler with full component set."""
    from astro_lab.models.astrophot_models import NSAGalaxyModeler

    return NSAGalaxyModeler(
        hidden_dim=hidden_dim,
        **kwargs,
    )
