"""
AstroLab Visualization Utilities
===============================

High-performance visualization tools for astronomical data with zero-copy data bridges.
"""

# SAFE imports - no Blender dependencies
from .cosmograph_bridge import CosmographBridge, create_cosmograph_visualization
from .graph import (
    TORCH_GEOMETRIC_AVAILABLE,
    calculate_graph_metrics,
    create_spatial_graph,
    spatial_distance_matrix,
)


# LAZY imports - only when explicitly needed to avoid Blender loading
def get_tensor_bridge():
    """Lazy import of tensor bridge (loads Blender)."""
    from .tensor_bridge import (
        BlenderZeroCopyBridge,
        NumpyZeroCopyBridge,
        PyVistaZeroCopyBridge,
        TensorProtocol,
        ZeroCopyBridge,
        get_tensor_memory_info,
        optimize_tensor_layout,
        pinned_memory_context,
        transfer_to_framework,
        zero_copy_context,
    )

    return {
        "PyVistaZeroCopyBridge": PyVistaZeroCopyBridge,
        "BlenderZeroCopyBridge": BlenderZeroCopyBridge,
        "NumpyZeroCopyBridge": NumpyZeroCopyBridge,
        "ZeroCopyBridge": ZeroCopyBridge,
        "transfer_to_framework": transfer_to_framework,
        "optimize_tensor_layout": optimize_tensor_layout,
        "get_tensor_memory_info": get_tensor_memory_info,
        "zero_copy_context": zero_copy_context,
        "pinned_memory_context": pinned_memory_context,
        "TensorProtocol": TensorProtocol,
    }


def get_bidirectional_bridge():
    """Lazy import of bidirectional bridge (consolidated in tensor_bridge)."""
    from .tensor_bridge import (
        BidirectionalTensorBridge,
        SyncConfig,
        create_bidirectional_bridge,
        quick_convert_pyvista_to_blender,
        quick_convert_tensor_to_blender,
        quick_convert_tensor_to_pyvista,
    )

    return {
        "BidirectionalTensorBridge": BidirectionalTensorBridge,
        "SyncConfig": SyncConfig,
        "create_bidirectional_bridge": create_bidirectional_bridge,
        "quick_convert_pyvista_to_blender": quick_convert_pyvista_to_blender,
        "quick_convert_tensor_to_blender": quick_convert_tensor_to_blender,
        "quick_convert_tensor_to_pyvista": quick_convert_tensor_to_pyvista,
    }


def get_tng50():
    """Lazy import of TNG50 visualizer (loads Blender)."""
    from .tng50 import (
        TNG50Visualizer,
        list_available_data,
        load_tng50_gas,
        load_tng50_stars,
        quick_blender_import,
        quick_pyvista_plot,
    )

    return {
        "TNG50Visualizer": TNG50Visualizer,
        "load_tng50_gas": load_tng50_gas,
        "load_tng50_stars": load_tng50_stars,
        "quick_pyvista_plot": quick_pyvista_plot,
        "quick_blender_import": quick_blender_import,
        "list_available_data": list_available_data,
    }


# SAFE exports only
__all__ = [
    # Graph utilities (safe)
    "create_spatial_graph",
    "calculate_graph_metrics",
    "spatial_distance_matrix",
    "TORCH_GEOMETRIC_AVAILABLE",
    # Cosmograph (safe)
    "CosmographBridge",
    "create_cosmograph_visualization",
    # Lazy loaders
    "get_tensor_bridge",
    "get_bidirectional_bridge",
    "get_tng50",
]
