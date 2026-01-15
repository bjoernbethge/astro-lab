"""
Plotly Widgets for AstroLab - Statistical and Interactive Astronomical Plots
============================================================================

Complete Plotly backend for astronomical data visualization.
"""

from .bridge import (
    AstronomicalPlotlyBridge,
    create_plotly_visualization,
    create_survey_comparison_plot,
)


def create_3d_scatter_plot(data, **kwargs):
    """Create 3D scatter plot using Plotly."""
    return create_plotly_visualization(data, plot_type="scatter_3d", **kwargs)


def create_3d_analysis_plot(data, **kwargs):
    """Create 3D analysis plot using Plotly."""
    return create_plotly_visualization(data, plot_type="analysis_3d", **kwargs)


from .cosmic_web_plots import (
    plot_cosmic_web_3d,
    plot_density_heatmap,
    plot_multi_scale_clustering,
)
from .stellar_plots import (
    plot_exoplanet_system,
    plot_galaxy_cluster,
    plot_stellar_evolution,
)

__all__ = [
    # Main bridge
    "AstronomicalPlotlyBridge",
    "create_plotly_visualization",
    "create_survey_comparison_plot",
    "create_3d_scatter_plot",
    "create_3d_analysis_plot",
    # Stellar plots
    "plot_stellar_evolution",
    "plot_galaxy_cluster",
    "plot_exoplanet_system",
    # Cosmic web plots
    "plot_cosmic_web_3d",
    "plot_density_heatmap",
    "plot_multi_scale_clustering",
]
