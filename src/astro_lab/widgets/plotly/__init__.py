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
    # Stellar plots
    "plot_stellar_evolution",
    "plot_galaxy_cluster",
    "plot_exoplanet_system",
    # Cosmic web plots
    "plot_cosmic_web_3d",
    "plot_density_heatmap",
    "plot_multi_scale_clustering",
]
