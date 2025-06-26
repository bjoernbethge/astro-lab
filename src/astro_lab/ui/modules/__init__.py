"""
AstroLab UI Modules
==================

Modular UI components that directly integrate with AstroLab backend.
Each module uses the actual AstroLab classes and methods.
"""

from .data import data_explorer, data_loader, catalog_manager
from .training import training_dashboard, model_selector, experiment_tracker
from .visualization import plot_creator, results_viewer, graph_visualizer
from .analysis import analysis_panel, clustering_tool, statistics_viewer
from .monitoring import system_monitor, gpu_monitor, mlflow_dashboard

__all__ = [
    # Data modules
    "data_explorer",
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
    # Analysis modules
    "analysis_panel",
    "clustering_tool",
    "statistics_viewer",
    # Monitoring modules
    "system_monitor",
    "gpu_monitor",
    "mlflow_dashboard",
]
