"""
Lightning Wrappers for AstroLab Core Models
==========================================

Lightning wrapper classes that wrap existing AstroLab models without
modifying them, allowing both traditional and Lightning APIs to coexist.
"""

import logging
from typing import Any, Dict, Optional, Union

import torch

from .base import AstroLabLightningMixin

logger = logging.getLogger(__name__)


def _filter_model_kwargs(model_class, **kwargs):
    """Filter kwargs to only include parameters accepted by the model class."""
    import inspect

    # Get the model's __init__ signature
    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    # Filter kwargs to only valid parameters
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    # Log filtered out parameters for debugging
    filtered_out = set(kwargs.keys()) - set(filtered.keys())
    if filtered_out:
        logger.debug(f"Filtered out kwargs for {model_class.__name__}: {filtered_out}")

    return filtered


class LightningAstroSurveyGNN(AstroLabLightningMixin):
    """Lightning wrapper for AstroSurveyGNN model."""

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        use_photometry: bool = True,
        use_astrometry: bool = True,
        use_spectroscopy: bool = False,
        pooling_type: str = "mean",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **lightning_kwargs,
    ):
        """
        Initialize Lightning-wrapped AstroSurveyGNN.

        Args:
            output_dim: Output dimension for the model
            hidden_dim: Hidden dimension size
            num_gnn_layers: Number of GNN layers
            use_photometry: Whether to use photometric data
            use_astrometry: Whether to use astrometric data
            use_spectroscopy: Whether to use spectroscopic data
            pooling_type: Type of pooling ("mean", "max", "sum")
            dropout: Dropout rate
            device: Device to place model on
            **lightning_kwargs: Lightning-specific parameters (learning_rate, optimizer, etc.)
        """
        # Initialize Lightning mixin first
        super().__init__(**lightning_kwargs)

        # Import here to avoid circular imports
        from ..core import AstroSurveyGNN

        # Create the wrapped model with filtered parameters
        model_kwargs = _filter_model_kwargs(
            AstroSurveyGNN,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            use_photometry=use_photometry,
            use_astrometry=use_astrometry,
            use_spectroscopy=use_spectroscopy,
            pooling_type=pooling_type,
            dropout=dropout,
            device=device,
        )

        self.model = AstroSurveyGNN(**model_kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        # Handle both SurveyTensorDict and PyG Data objects
        if len(args) > 0:
            input_data = args[0]
        else:
            input_data = kwargs.get("x", None)

        if input_data is not None:
            # Check if it's a PyG Data object
            if hasattr(input_data, "x") and hasattr(input_data, "edge_index"):
                # Convert PyG Data to SurveyTensorDict
                from astro_lab.tensors import (
                    PhotometricTensorDict,
                    SpatialTensorDict,
                    SurveyTensorDict,
                )

                device = input_data.x.device
                # Create spatial tensor from coordinates (x[:, :3] if available, otherwise use x)
                if input_data.x.dim() >= 2 and input_data.x.size(1) >= 3:
                    coords = input_data.x[:, :3]
                elif input_data.x.dim() >= 2 and input_data.x.size(1) == 2:
                    coords = torch.cat(
                        [
                            input_data.x,
                            torch.zeros(input_data.x.size(0), 1, device=device),
                        ],
                        dim=1,
                    )
                else:
                    if input_data.x.dim() == 1:
                        coords = torch.cat(
                            [
                                input_data.x.unsqueeze(1),
                                torch.zeros(input_data.x.size(0), 2, device=device),
                            ],
                            dim=1,
                        )
                    else:
                        coords = torch.cat(
                            [
                                input_data.x,
                                torch.zeros(
                                    input_data.x.size(0),
                                    3 - input_data.x.size(1),
                                    device=device,
                                ),
                            ],
                            dim=1,
                        )

                spatial = SpatialTensorDict(
                    coordinates=coords,
                    coordinate_system="icrs",
                    unit="parsec",
                    epoch=2000.0,
                )

                # Create photometric tensor if we have more than 3 features
                photometric = None
                if input_data.x.dim() >= 2 and input_data.x.size(1) > 3:
                    mags = input_data.x[:, 3:]
                    photometric = PhotometricTensorDict(
                        magnitudes=mags,
                        bands=[f"band_{i}" for i in range(mags.size(1))],
                        filter_system="generic",
                        is_magnitude=True,
                    )
                else:
                    dummy_mags = torch.zeros(coords.size(0), 1, device=device)
                    photometric = PhotometricTensorDict(
                        magnitudes=dummy_mags,
                        bands=["dummy"],
                        filter_system="generic",
                        is_magnitude=True,
                    )

                # Create SurveyTensorDict
                survey_tensor = SurveyTensorDict(
                    spatial=spatial,
                    photometric=photometric,
                    survey_name="converted_from_pyg",
                )

                # Replace the input
                if len(args) > 0:
                    args = (survey_tensor,) + args[1:]
                else:
                    kwargs["x"] = survey_tensor

        return self.model(*args, **kwargs)


class LightningAstroPhotGNN(AstroLabLightningMixin):
    """Lightning wrapper for AstroPhotGNN model."""

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        use_color_features: bool = True,
        use_magnitude_errors: bool = True,
        pooling_type: str = "mean",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **lightning_kwargs,
    ):
        """
        Initialize Lightning-wrapped AstroPhotGNN.

        Args:
            output_dim: Output dimension for the model
            hidden_dim: Hidden dimension size
            num_gnn_layers: Number of GNN layers
            use_color_features: Whether to use color features
            use_magnitude_errors: Whether to use magnitude errors
            pooling_type: Type of pooling ("mean", "max", "sum")
            dropout: Dropout rate
            device: Device to place model on
            **lightning_kwargs: Lightning-specific parameters
        """
        super().__init__(**lightning_kwargs)

        from ..core import AstroPhotGNN

        model_kwargs = _filter_model_kwargs(
            AstroPhotGNN,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            use_color_features=use_color_features,
            use_magnitude_errors=use_magnitude_errors,
            pooling_type=pooling_type,
            dropout=dropout,
            device=device,
        )

        self.model = AstroPhotGNN(**model_kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)


class LightningTemporalGCN(AstroLabLightningMixin):
    """Lightning wrapper for TemporalGCN model."""

    def __init__(
        self,
        output_dim: int = 128,
        hidden_dim: int = 256,
        num_temporal_layers: int = 2,
        num_gnn_layers: int = 2,
        temporal_model: str = "lstm",
        dropout: float = 0.1,
        use_attention: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **lightning_kwargs,
    ):
        """
        Initialize Lightning-wrapped TemporalGCN.

        Args:
            output_dim: Output dimension for the model
            hidden_dim: Hidden dimension size
            num_temporal_layers: Number of temporal layers
            num_gnn_layers: Number of GNN layers
            temporal_model: Type of temporal model ("lstm", "gru")
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            device: Device to place model on
            **lightning_kwargs: Lightning-specific parameters
        """
        super().__init__(**lightning_kwargs)

        from ..core import TemporalGCN

        model_kwargs = _filter_model_kwargs(
            TemporalGCN,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_temporal_layers=num_temporal_layers,
            num_gnn_layers=num_gnn_layers,
            temporal_model=temporal_model,
            dropout=dropout,
            use_attention=use_attention,
            device=device,
        )

        self.model = TemporalGCN(**model_kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)


class LightningALCDEFTemporalGNN(AstroLabLightningMixin):
    """Lightning wrapper for ALCDEFTemporalGNN model."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        task: str = "period_detection",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **lightning_kwargs,
    ):
        """
        Initialize Lightning-wrapped ALCDEFTemporalGNN.

        Args:
            input_dim: Input dimension (typically 1 for magnitude values)
            hidden_dim: Hidden dimension size
            output_dim: Output dimension
            num_layers: Number of GNN layers
            task: Task type ("period_detection", "shape_modeling", "classification")
            dropout: Dropout rate
            device: Device to place model on
            **lightning_kwargs: Lightning-specific parameters
        """
        super().__init__(**lightning_kwargs)

        from ..core import ALCDEFTemporalGNN

        model_kwargs = _filter_model_kwargs(
            ALCDEFTemporalGNN,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            task=task,
            dropout=dropout,
            device=device,
        )

        self.model = ALCDEFTemporalGNN(**model_kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)


# Additional specialized wrappers for common use cases
class LightningGaiaClassifier(LightningAstroSurveyGNN):
    """Specialized Lightning wrapper for Gaia stellar classification."""

    def __init__(
        self,
        num_classes: int = 3,  # Main sequence, giant, white dwarf
        **kwargs,
    ):
        """Initialize Gaia classifier with optimal defaults."""
        # Set defaults that can be overridden
        defaults = {
            "output_dim": num_classes,
            "task": "classification",
            "num_classes": num_classes,
            "use_photometry": True,
            "use_astrometry": True,
            "use_spectroscopy": False,
            "hidden_dim": 256,
            "num_gnn_layers": 3,
        }

        # Merge with user kwargs (user kwargs take precedence)
        final_kwargs = {**defaults, **kwargs}
        super().__init__(**final_kwargs)


class LightningGalaxyModeler(LightningAstroPhotGNN):
    """Specialized Lightning wrapper for galaxy modeling."""

    def __init__(self, **kwargs):
        """Initialize galaxy modeler with optimal defaults."""
        defaults = {
            "task": "shape_modeling",
            "use_color_features": True,
            "use_magnitude_errors": True,
            "hidden_dim": 512,
            "num_gnn_layers": 4,
        }

        final_kwargs = {**defaults, **kwargs}
        super().__init__(**final_kwargs)


class LightningAsteroidPeriodDetector(LightningALCDEFTemporalGNN):
    """Specialized Lightning wrapper for asteroid period detection."""

    def __init__(self, **kwargs):
        """Initialize asteroid period detector with optimal defaults."""
        defaults = {
            "task": "period_detection",
            "input_dim": 1,  # Magnitude values
            "output_dim": 1,  # Period estimate
            "hidden_dim": 256,
            "num_layers": 4,
        }

        final_kwargs = {**defaults, **kwargs}
        super().__init__(**final_kwargs)


class LightningTransientClassifier(LightningTemporalGCN):
    """Specialized Lightning wrapper for transient classification."""

    def __init__(
        self,
        num_classes: int = 5,  # SN Ia, SN II, SN Ib/c, AGN, etc.
        **kwargs,
    ):
        """Initialize transient classifier with optimal defaults."""
        defaults = {
            "output_dim": num_classes,
            "task": "classification",
            "num_classes": num_classes,
            "temporal_model": "lstm",
            "use_attention": True,
            "hidden_dim": 512,
        }

        final_kwargs = {**defaults, **kwargs}
        super().__init__(**final_kwargs)
