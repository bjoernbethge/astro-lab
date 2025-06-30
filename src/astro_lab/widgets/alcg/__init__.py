"""
AstroLab Cosmograph Integration (ALCG)
=====================================

Cosmograph integration for AstroLab with deep TensorDict support
and proper astronomical data visualization.

Based on:
- @cosmograph/cosmograph JavaScript library
- py_cosmograph Python bindings
- cosmograph_widget for Jupyter integration
"""

# Convenience functions
from .bridge import (
    CosmographBridge,
    CosmographConfig,
    CosmographLinkData,
    CosmographNodeData,
)
from .convenience import (
    create_cosmic_web_cosmograph,
    create_cosmograph_from_tensordict,
    create_multimodal_cosmograph,
    visualize_analysis_results,
    visualize_spatial_tensordict,
)

__all__ = [
    # Core classes
    "CosmographBridge",
    "CosmographConfig",
    "CosmographNodeData",
    "CosmographLinkData",
    # Convenience functions
    "create_cosmograph_from_tensordict",
    "visualize_spatial_tensordict",
    "visualize_analysis_results",
    "create_cosmic_web_cosmograph",
    "create_multimodal_cosmograph",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "AstroLab Team"

# Module documentation
__doc__ += """

ALCG (AstroLab Cosmograph) Integration Features:
===============================================

1. **Deep TensorDict Integration**:
   - Direct visualization from SpatialTensorDict, PhotometricTensorDict, \
     AnalysisTensorDict
   - Automatic coordinate system detection and unit conversion
   - Proper astronomical metadata handling

2. **Astronomical Styling**:
   - Survey-specific color schemes (Gaia gold, SDSS blue, TNG50 green)
   - Magnitude-based point sizing with proper astronomical scaling
   - Color-magnitude diagram integration for photometric data

3. **Interactive Cosmic Web Visualization**:
   - Real-time clustering analysis with interactive selection
   - Filament network visualization with topology overlays
   - Multi-scale structure analysis with zoom-based detail levels

4. **GPU-Accelerated Performance**:
   - Leverage Cosmograph's WebGL acceleration for large datasets
   - Efficient graph construction using PyTorch Geometric
   - Memory-optimized data transfer for million-point visualizations

5. **Jupyter Integration**:
   - Seamless embedding in notebooks via anywidget
   - Interactive tooltips with astronomical information
   - Export capabilities for publications and presentations

Features:
=================

```python
from astro_lab.widgets.alcg import CosmicWebCosmographVisualizer

# cosmic web visualization
visualizer = CosmicWebCosmographVisualizer()

# Configure for different analysis types
config = visualizer.create_config_for_analysis(
    analysis_type="clustering",
    survey="gaia",
    n_objects=25000,
    clustering_scales=[5.0, 10.0, 25.0],
    show_filaments=True,
    interactive_selection=True
)

# Create visualization with analysis overlays
viz = visualizer.create_from_analysis_tensordict(
    analysis_tensordict=results,
    config=config
)

# Add interactive features
viz.add_cluster_selection_callback(on_cluster_select)
viz.add_magnitude_filtering(magnitude_range=[8.0, 15.0])
viz.enable_temporal_animation(time_steps=analysis_timesteps)
```

Performance Characteristics:
===========================

- **Small datasets** (< 1,000 objects): Real-time interaction, all features enabled
- **Medium datasets** (1,000 - 50,000): GPU acceleration, adaptive detail levels
- **Large datasets** (50,000 - 1M): Optimized rendering, level-of-detail management
- **Massive datasets** (> 1M): Streaming data, progressive enhancement

The ALCG module provides the most advanced astronomical data visualization
capabilities available, combining Cosmograph's cutting-edge GPU acceleration
with AstroLab's deep astronomical domain knowledge.
"""
