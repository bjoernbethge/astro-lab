"""
AstroLab Cosmograph Integration (ALCG)
=====================================

Cosmograph integration for AstroLab with support for both TensorDict
and non-TensorDict data sources.

Based on:
- @cosmograph/cosmograph JavaScript library
- py_cosmograph Python bindings
- cosmograph_widget for Jupyter integration
"""

# Core bridge classes
from .bridge import (
    CosmographBridge,
    CosmographConfig,
    CosmographLinkData,
    CosmographNodeData,
)

# Convenience functions
from .convenience import (
    # TensorDict-based functions
    create_cosmic_web_cosmograph,
    # Non-TensorDict functions
    create_cosmograph_from_coordinates,
    create_cosmograph_from_dataframe,
    create_cosmograph_from_tensordict,
    create_cosmograph_visualization,
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
    # TensorDict convenience functions
    "create_cosmograph_from_tensordict",
    "visualize_spatial_tensordict",
    "visualize_analysis_results",
    "create_cosmic_web_cosmograph",
    "create_multimodal_cosmograph",
    # Non-TensorDict convenience functions
    "create_cosmograph_from_coordinates",
    "create_cosmograph_from_dataframe",
    "create_cosmograph_visualization",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "AstroLab Team"

# Module documentation
__doc__ = (
    (__doc__ or "")
    + """

ALCG (AstroLab Cosmograph) Integration Features:
===============================================

1. **Flexible Data Support**:
   - Direct visualization from SpatialTensorDict, PhotometricTensorDict, AnalysisTensorDict
   - Support for numpy arrays, Polars/Pandas DataFrames
   - Automatic data source detection and routing
   - Zero-copy operations where possible

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

Usage Examples:
===============

```python
from astro_lab.widgets.alcg import create_cosmograph_visualization

# From numpy coordinates
coords = np.random.randn(1000, 3) * 100
viz = create_cosmograph_visualization(coords, survey="gaia")

# From Polars DataFrame
df = pl.DataFrame({
    "x": np.random.randn(1000),
    "y": np.random.randn(1000),
    "z": np.random.randn(1000),
    "magnitude": np.random.rand(1000) * 10
})
viz = create_cosmograph_visualization(df)

# From TensorDict (if available)
spatial_tensor = SpatialTensorDict(coordinates=coords_tensor)
viz = create_cosmograph_visualization(spatial_tensor)

# From dictionary
data = {"coordinates": coords}
viz = create_cosmograph_visualization(data)
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
)
