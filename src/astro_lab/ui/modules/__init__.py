"""
AstroLab UI Modules
==================

Modular UI components that directly integrate with AstroLab backend.
Each module uses the actual AstroLab classes and methods.
"""

from .analysis import (
    analysis_panel,
    clustering_tool,
    statistics_viewer,
    subgraph_sampler,
)
from .cosmic_web import comparison_tool, connectivity_analyzer, cosmic_web_panel
from .data import catalog_manager, data_loader
from .monitoring import (
    gpu_monitor,
    mlflow_dashboard,
    system_monitor,
    training_monitor,
)
from .training import experiment_tracker, model_selector, training_dashboard
from .visualization import (
    clustering_visualizer,
    cosmograph_viewer,
    graph_creator,
    graph_visualizer,
    plot_creator,
    results_viewer,
)

__all__ = [
    # Data modules
    "data_loader",
    "catalog_manager",
    # Training modules
    "training_dashboard",
    "model_selector",
    "experiment_tracker",
    # Visualization modules
    "plot_creator",
    "results_viewer",
    "graph_visualizer",
    "graph_creator",
    "clustering_visualizer",
    "cosmograph_viewer",
    # Analysis modules
    "analysis_panel",
    "clustering_tool",
    "statistics_viewer",
    "subgraph_sampler",
    # Cosmic web modules
    "cosmic_web_panel",
    "connectivity_analyzer",
    "comparison_tool",
    # Monitoring modules
    "system_monitor",
    "gpu_monitor",
    "mlflow_dashboard",
    "training_monitor",
]
