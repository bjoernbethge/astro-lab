"""AstroLab Widgets - Interactive astronomical visualization and analysis."""

from .astro_lab import AstroLabWidget
from .analysis import AnalysisModule
from .graph import GraphModule
from .visualization import VisualizationModule

__all__ = [
    "AstroLabWidget",
    "AnalysisModule", 
    "GraphModule",
    "VisualizationModule"
]

# Main widget provides unified API:
# - widget.plot() - Visualization with multiple backends
# - widget.find_neighbors() - GPU-accelerated neighbor finding
# - widget.cluster_data() - Clustering analysis
# - widget.analyze_density() - Density analysis
# - widget.create_graph() - PyTorch Geometric integration
# - widget.prepare_for_model() - Model preparation
# - widget.al, widget.ops, widget.data, widget.context, widget.scene - Blender API
