"""
AstroLab Widgets Module
=======================

This module provides visualization and interactive widgets for astronomical data.
"""

import logging
from typing import Any, Dict, Optional

from astro_lab.tensors import (
    AnalysisTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
)

logger = logging.getLogger(__name__)


def get_optimal_backend(tensordict: Any) -> str:
    """
    Determine optimal backend for given TensorDict.

    Args:
        tensordict: AstroLab TensorDict

    Returns:
        Optimal backend name
    """
    n_objects = getattr(tensordict, "n_objects", 0)

    # Size-based backend selection
    if n_objects > 100_000:
        return "cosmograph"  # Large datasets - GPU-accelerated
    elif n_objects > 10_000:
        return "pyvista"  # Medium datasets - interactive 3D

    # Type-based selection for smaller datasets
    if isinstance(tensordict, PhotometricTensorDict):
        return "plotly"  # Statistical plots for photometry
    elif isinstance(tensordict, AnalysisTensorDict):
        return "cosmograph"  # Interactive for analysis results

    # Default
    return "pyvista"


def visualize(tensordict: Any, backend: str = "auto", **kwargs) -> Any:
    """
    Unified visualization function for AstroLab TensorDicts.

    Args:
        tensordict: Any AstroLab TensorDict (Spatial, Photometric, Analysis, etc.)
        backend: Visualization backend ('auto', 'pyvista', 'open3d', 'blender', 'cosmograph', 'plotly')
        **kwargs: Backend-specific parameters

    Returns:
        Backend-specific visualization object

    Examples:
        >>> from astro_lab.widgets import visualize
        >>> viz = visualize(spatial_tensor)  # Auto-select backend
        >>> viz = visualize(spatial_tensor, backend="pyvista")  # Explicit backend
        >>> viz = visualize(analysis_tensor, backend="cosmograph", interactive=True)
    """
    # Auto-select backend
    if backend == "auto":
        backend = get_optimal_backend(tensordict)
        logger.debug(f"Auto-selected backend: {backend}")

    # Route to appropriate backend
    if backend == "pyvista":
        from .alpv import create_pyvista_visualization

        return create_pyvista_visualization(tensordict, **kwargs)

    elif backend == "open3d":
        from .alo3d import create_open3d_visualization

        return create_open3d_visualization(tensordict, **kwargs)

    elif backend == "blender":
        from .albpy import create_blender_visualization

        return create_blender_visualization(tensordict, **kwargs)

    elif backend == "cosmograph":
        from .alcg import create_cosmograph_from_tensordict

        return create_cosmograph_from_tensordict(tensordict, **kwargs)

    elif backend == "plotly":
        from .plotly import create_plotly_visualization

        return create_plotly_visualization(tensordict, **kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}")


def visualize_cosmic_web(
    spatial_tensor: SpatialTensorDict,
    analysis_tensor: Optional[AnalysisTensorDict] = None,
    backend: str = "auto",
    **kwargs,
) -> Any:
    """
    Specialized cosmic web visualization.

    Args:
        spatial_tensor: SpatialTensorDict with coordinates
        analysis_tensor: Optional AnalysisTensorDict with clustering results
        backend: Visualization backend
        **kwargs: Backend-specific parameters

    Returns:
        Cosmic web visualization
    """
    if backend == "auto":
        n_objects = spatial_tensor.n_objects
        backend = "cosmograph" if n_objects > 50_000 else "pyvista"

    # Add cosmic web specific parameters
    kwargs.setdefault("show_connections", True)
    kwargs.setdefault("cluster_colors", True)

    if analysis_tensor is not None:
        kwargs["analysis_results"] = analysis_tensor

    return visualize(spatial_tensor, backend=backend, **kwargs)


def visualize_hr_diagram(
    photometric_tensor: PhotometricTensorDict, backend: str = "plotly", **kwargs
) -> Any:
    """
    Specialized Hertzsprung-Russell diagram visualization.

    Args:
        photometric_tensor: PhotometricTensorDict with magnitude data
        backend: Visualization backend (plotly recommended)
        **kwargs: Backend-specific parameters

    Returns:
        HR diagram visualization
    """
    kwargs.setdefault("plot_type", "hr_diagram")
    return visualize(photometric_tensor, backend=backend, **kwargs)


def visualize_survey_comparison(
    tensordicts: Dict[str, Any], backend: str = "auto", **kwargs
) -> Any:
    """
    Compare multiple survey datasets.

    Args:
        tensordicts: Dictionary of {survey_name: tensordict}
        backend: Visualization backend
        **kwargs: Backend-specific parameters

    Returns:
        Multi-survey comparison visualization
    """
    if backend == "auto":
        max_objects = max(getattr(td, "n_objects", 0) for td in tensordicts.values())
        backend = "cosmograph" if max_objects > 50_000 else "pyvista"

    # Route to backend-specific multi-survey function
    if backend == "cosmograph":
        from .alcg import create_multimodal_cosmograph

        return create_multimodal_cosmograph(tensordicts, **kwargs)

    elif backend == "pyvista":
        from .alpv import create_multi_survey_visualization

        return create_multi_survey_visualization(tensordicts, **kwargs)

    elif backend == "plotly":
        from .plotly import create_survey_comparison_plot

        return create_survey_comparison_plot(tensordicts, **kwargs)

    else:
        # Fallback: visualize largest dataset
        largest_key = max(
            tensordicts.keys(), key=lambda k: getattr(tensordicts[k], "n_objects", 0)
        )
        return visualize(tensordicts[largest_key], backend=backend, **kwargs)


# Re-export important backend components
from .cosmograph_bridge import CosmographBridge

# Marimo widgets
from .marimo_widgets import (
    AstroLabDashboard,
    AstroLabWidget,
    CosmicWebWidget,
    HRDiagramWidget,
    SurveyComparisonWidget,
    TimeSeriesWidget,
    create_astrolab_dashboard,
    create_cosmic_web_widget,
    create_hr_diagram_widget,
    create_survey_comparison_widget,
)
from .tensor_bridge import (
    AstronomicalTensorBridge,
    to_cosmograph,
    to_plotly,
    to_pyvista,
)

# Export main functions
__all__ = [
    # Main API
    "visualize",
    "visualize_cosmic_web",
    "visualize_hr_diagram",
    "visualize_survey_comparison",
    # Utility functions
    "get_optimal_backend",
    # Important components
    "CosmographBridge",
    "AstronomicalTensorBridge",
    # Backend converters
    "to_pyvista",
    "to_plotly",
    "to_cosmograph",
    # Widget classes
    "AstroLabWidget",
    "CosmicWebWidget",
    "HRDiagramWidget",
    "SurveyComparisonWidget",
    "TimeSeriesWidget",
    "AstroLabDashboard",
    # Factory functions
    "create_cosmic_web_widget",
    "create_hr_diagram_widget",
    "create_survey_comparison_widget",
    "create_astrolab_dashboard",
]

logger.debug("AstroLab Widgets initialized with clean API")
