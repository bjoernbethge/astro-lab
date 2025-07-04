"""
Enhanced Widgets Module
======================

Advanced widget functionality that orchestrates multiple specialized packages
for astronomical data visualization and processing.
"""

# Image processing functionality
# Backend-specific converters
from .backend_converters import (
    astronomical_tensor_zero_copy_context,
    bridge,
    to_blender,
    to_cosmograph,
    to_open3d,
    to_plotly,
    to_pyvista,
    transfer_astronomical_tensor,
)
from .image_processing import (
    ImageProcessor,
    create_pyvista_visualization,
    orchestrate_pipeline,
    use_astrophot_models,
    use_numpy_operations,
    use_photutils_analysis,
)

# Post-processing functionality
from .post_processing import (
    PostProcessor,
    apply_pyvista_filters,
    apply_pyvista_transforms,
    create_pyvista_animation,
    create_pyvista_screenshot,
    export_pyvista_mesh,
    orchestrate_post_processing,
    use_blender_compositing_nodes,
)

# Tensor bridge functionality
from .tensor_bridge import (
    AstronomicalTensorBridge,
    AstronomicalTensorZeroCopyBridge,
    tensor_bridge_context,
)

# Tensor converters
from .tensor_converters import (
    ZeroCopyTensorConverter,
    converter,
)

# Texture generation functionality
from .texture_generation import (
    TextureGenerator,
    create_photometric_texture,
    create_spatial_texture,
    orchestrate_texture_pipeline,
    use_blender_texture_generation,
    use_numpy_texture_processing,
    use_open3d_texture_generation,
    use_pyvista_texture_generation,
    use_pyvista_texture_mapping,
)

__all__ = [
    # Image processing
    "ImageProcessor",
    "create_pyvista_visualization",
    "use_astrophot_models",
    "use_photutils_analysis",
    "use_numpy_operations",
    "orchestrate_pipeline",
    # Post-processing
    "PostProcessor",
    "apply_pyvista_filters",
    "apply_pyvista_transforms",
    "create_pyvista_animation",
    "create_pyvista_screenshot",
    "export_pyvista_mesh",
    "use_blender_compositing_nodes",
    "orchestrate_post_processing",
    # Texture generation
    "TextureGenerator",
    "use_pyvista_texture_mapping",
    "use_pyvista_texture_generation",
    "use_open3d_texture_generation",
    "use_blender_texture_generation",
    "use_numpy_texture_processing",
    "create_photometric_texture",
    "create_spatial_texture",
    "orchestrate_texture_pipeline",
    # Tensor bridge
    "AstronomicalTensorBridge",
    "tensor_bridge_context",
    "AstronomicalTensorZeroCopyBridge",
    # Tensor converters
    "ZeroCopyTensorConverter",
    "converter",
    # Backend converters
    "to_pyvista",
    "to_open3d",
    "to_blender",
    "to_plotly",
    "to_cosmograph",
    "bridge",
    "transfer_astronomical_tensor",
    "astronomical_tensor_zero_copy_context",
]
