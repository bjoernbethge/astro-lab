"""
AstroLab Visualization Utilities
===============================

High-performance visualization tools for astronomical data with zero-copy data bridges.
"""

from .tensor_bridge import (
    PyVistaZeroCopyBridge,
    BlenderZeroCopyBridge,
    NumpyZeroCopyBridge,
    ZeroCopyBridge,
    transfer_to_framework,
    optimize_tensor_layout,
    get_tensor_memory_info,
    zero_copy_context,
    pinned_memory_context,
    TensorProtocol,
)

from .bidirectional_bridge import (
    BidirectionalPyVistaBlenderBridge,
    MaterialBridge,
    SyncConfig,
    create_bidirectional_bridge,
    quick_convert_pyvista_to_blender,
    quick_convert_blender_to_pyvista,
)

from .tng50 import (
    TNG50Visualizer,
    load_tng50_gas,
    load_tng50_stars,
    quick_pyvista_plot,
    quick_blender_import,
    list_available_data,
)

from .graph import (
    create_spatial_graph,
    calculate_graph_metrics,
    spatial_distance_matrix,
    TORCH_GEOMETRIC_AVAILABLE,
)

from .cosmograph_bridge import CosmographBridge, create_cosmograph_visualization

__all__ = [
    # Tensor bridges
    "PyVistaZeroCopyBridge",
    "BlenderZeroCopyBridge", 
    "NumpyZeroCopyBridge",
    "ZeroCopyBridge",
    "transfer_to_framework",
    "optimize_tensor_layout",
    "get_tensor_memory_info",
    "zero_copy_context",
    "pinned_memory_context",
    "TensorProtocol",
    
    # Bidirectional bridge
    "BidirectionalPyVistaBlenderBridge",
    "MaterialBridge",
    "SyncConfig",
    "create_bidirectional_bridge",
    "quick_convert_pyvista_to_blender",
    "quick_convert_blender_to_pyvista",
    
    # TNG50 visualization
    "TNG50Visualizer",
    "load_tng50_gas",
    "load_tng50_stars",
    "quick_pyvista_plot",
    "quick_blender_import",
    "list_available_data",
    
    # Graph utilities
    "create_spatial_graph",
    "calculate_graph_metrics", 
    "spatial_distance_matrix",
    "TORCH_GEOMETRIC_AVAILABLE",
    
    # Cosmograph
    "CosmographBridge",
    "create_cosmograph_visualization"
] 