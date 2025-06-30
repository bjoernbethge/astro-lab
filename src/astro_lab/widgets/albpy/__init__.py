"""
Blender Widgets for AstroLab - 3D Visualization and Animation
============================================================

High-quality 3D visualization and animation using Blender's Python API.
"""

import logging
from typing import Any, Dict, Optional

import bpy

# 2D/3D annotation features
from .grease_pencil_2d import GreasePencil2DPlotter

# Core components
from .materials import (
    create_astronomical_material,
    create_galaxy_material,
    create_nebula_material,
    get_stellar_material,
)
from .objects import (
    create_astronomical_object,
    create_galaxy_object,
    create_point_cloud,
    create_stellar_object,
)
from .operators import AstronomicalOperators
from .scene import (
    create_astronomical_scene,
    setup_astronomical_scene,
    setup_camera,
    setup_lighting,
)
from .tensor_bridge import (
    AstronomicalBlenderZeroCopyBridge,
    create_blender_from_tensordict,
)

logger = logging.getLogger(__name__)


def create_blender_visualization(tensordict: Any, **kwargs):
    """
    Create Blender visualization from TensorDict.

    Args:
        tensordict: AstroLab TensorDict
        **kwargs: Blender-specific parameters

    Returns:
        Blender scene or object
    """
    bridge = AstronomicalBlenderZeroCopyBridge()

    # Route based on TensorDict type
    from astro_lab.tensors import (
        AnalysisTensorDict,
        PhotometricTensorDict,
        SpatialTensorDict,
    )

    if isinstance(tensordict, SpatialTensorDict):
        return bridge.spatial_to_blender(tensordict, **kwargs)
    elif isinstance(tensordict, PhotometricTensorDict):
        return bridge.photometric_to_blender(tensordict, **kwargs)
    elif isinstance(tensordict, AnalysisTensorDict):
        return bridge.analysis_to_blender(tensordict, **kwargs)
    else:
        # Generic coordinate conversion
        coords = (
            tensordict["coordinates"] if "coordinates" in tensordict else tensordict
        )
        return bridge.coordinates_to_blender(coords, **kwargs)


def create_stellar_field(
    spatial_tensor: Any,
    temperature_data: Optional[Any] = None,
    magnitude_data: Optional[Any] = None,
    **kwargs,
):
    """
    Create realistic stellar field in Blender.

    Args:
        spatial_tensor: SpatialTensorDict with stellar coordinates
        temperature_data: Optional temperature data for realistic colors
        magnitude_data: Optional magnitude data for sizing
        **kwargs: Rendering parameters

    Returns:
        Blender objects
    """
    # Setup scene
    setup_astronomical_scene("stellar_field")

    if hasattr(spatial_tensor, "coordinates"):
        coords = spatial_tensor["coordinates"].cpu().numpy()
    else:
        coords = spatial_tensor.cpu().numpy()

    # Create point cloud with instancing for performance
    star_instance = create_stellar_object(
        temperature=5778,  # Template star
        radius=kwargs.get("base_star_radius", 0.1),
    )

    # Create instances
    objects = create_point_cloud(
        coords,
        instance_object=star_instance,
        color_data=temperature_data,
        size_data=magnitude_data,
        **kwargs,
    )

    return objects


def create_galaxy_cluster_visualization(
    galaxy_coords: Any, galaxy_properties: Optional[Dict] = None, **kwargs
):
    """
    Create galaxy cluster visualization with realistic galaxy types.

    Args:
        galaxy_coords: Galaxy coordinates
        galaxy_properties: Optional galaxy properties (mass, type, redshift, etc.)
        **kwargs: Visualization parameters

    Returns:
        Blender galaxy objects
    """
    setup_astronomical_scene("galaxy_cluster", scale="Mpc")

    if hasattr(galaxy_coords, "coordinates"):
        coords = galaxy_coords["coordinates"].cpu().numpy()
    else:
        coords = galaxy_coords.cpu().numpy()

    galaxies = []

    for i, coord in enumerate(coords):
        # Determine galaxy properties
        props = {}
        if galaxy_properties:
            props = {
                "galaxy_type": galaxy_properties.get("type", ["spiral"])[i],
                "mass": galaxy_properties.get("mass", [1e12])[i],
                "redshift": galaxy_properties.get("redshift", [0.0])[i],
                "size": galaxy_properties.get("size", [50.0])[i],
            }

        # Create galaxy with appropriate visualization
        galaxy = create_galaxy_visualization(position=coord, **props, **kwargs)
        galaxies.append(galaxy)

    return galaxies


def render_astronomical_scene(
    output_path: str,
    render_engine: str = "CYCLES",
    resolution: tuple = (1920, 1080),
    samples: int = 128,
    **kwargs,
):
    """
    Render astronomical scene with professional settings.

    Args:
        output_path: Output file path
        render_engine: Render engine ('CYCLES', 'EEVEE', 'WORKBENCH')
        resolution: Output resolution (width, height)
        samples: Number of samples for quality
        **kwargs: Additional render parameters
    """
    # Configure render settings
    scene = bpy.context.scene
    scene.render.engine = render_engine
    scene.render.filepath = output_path
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]

    if render_engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = kwargs.get("denoise", True)
        scene.cycles.use_adaptive_sampling = kwargs.get("adaptive", True)
    elif render_engine == "EEVEE":
        scene.eevee.taa_render_samples = samples
        scene.eevee.use_bloom = kwargs.get("bloom", True)
        scene.eevee.use_ssr = kwargs.get("reflections", True)

    # Apply post-processing if requested
    if kwargs.get("post_processing", False):
        setup_post_processing(**kwargs)

    # Render
    bpy.ops.render.render(write_still=True)


def setup_cosmic_web_scene(
    spatial_tensor: Any, analysis_results: Optional[Dict] = None, **kwargs
):
    """
    Setup complete cosmic web scene with filaments and clusters.

    Args:
        spatial_tensor: SpatialTensorDict with coordinates
        analysis_results: Optional analysis results (clusters, filaments, etc.)
        **kwargs: Scene parameters

    Returns:
        Complete scene setup
    """
    # Setup base scene
    scene = setup_astronomical_scene("cosmic_web", scale="Mpc")

    # Create main point cloud
    points = create_blender_visualization(spatial_tensor, **kwargs)

    # Add clusters if available
    if analysis_results and "cluster_labels" in analysis_results:
        # Color points by cluster
        # Implementation depends on specific cluster visualization
        pass

    # Add filament connections
    if kwargs.get("show_filaments", False) or (
        analysis_results and "filaments" in analysis_results
    ):
        create_filament_network(
            spatial_tensor, filament_data=analysis_results.get("filaments"), **kwargs
        )

    # Add volumetric effects for density
    if kwargs.get("volumetric", False):
        setup_volumetric_rendering(
            density_field=analysis_results.get("density"), **kwargs
        )

    # Setup appropriate lighting
    setup_lighting(scene_type="cosmic_web", **kwargs)

    # Setup camera
    setup_camera(
        target_object=points[0] if points else None,
        distance=kwargs.get("camera_distance", 100.0),
        **kwargs,
    )

    return scene


def create_animated_timeline(
    temporal_data: Dict[str, Any],
    frame_rate: int = 24,
    duration: float = 10.0,
    **kwargs,
):
    """
    Create animated timeline visualization.

    Args:
        temporal_data: Dictionary with time-indexed data
        frame_rate: Frames per second
        duration: Total animation duration in seconds
        **kwargs: Animation parameters

    Returns:
        Animated scene
    """
    scene = bpy.context.scene
    scene.render.fps = frame_rate
    total_frames = int(duration * frame_rate)
    scene.frame_end = total_frames

    # Setup keyframes for temporal data
    for frame, (time, data) in enumerate(temporal_data.items()):
        if frame >= total_frames:
            break

        scene.frame_set(frame)

        # Update visualization based on data
        # Implementation depends on specific data type

    return scene


# Convenience functions
def quick_render(tensordict: Any, output_path: str = "astro_render.png", **kwargs):
    """Quick render of astronomical data."""
    create_blender_visualization(tensordict, **kwargs)
    render_astronomical_scene(output_path, **kwargs)


def create_publication_figure(
    tensordict: Any, output_path: str, style: str = "scientific", **kwargs
):
    """Create publication-ready figure."""
    # Apply style presets
    if style == "scientific":
        kwargs.setdefault("background_color", (1, 1, 1))
        kwargs.setdefault("render_engine", "CYCLES")
        kwargs.setdefault("samples", 256)
        kwargs.setdefault("resolution", (2400, 1800))
    elif style == "presentation":
        kwargs.setdefault("background_color", (0, 0, 0))
        kwargs.setdefault("emission_strength", 2.0)
        kwargs.setdefault("bloom", True)

    create_blender_visualization(tensordict, **kwargs)
    render_astronomical_scene(output_path, **kwargs)


__all__ = [
    # Main functions
    "create_blender_visualization",
    "create_stellar_field",
    "create_galaxy_cluster_visualization",
    "render_astronomical_scene",
    "setup_cosmic_web_scene",
    "create_animated_timeline",
    # Scene setup
    "setup_astronomical_scene",
    "create_astronomical_scene",
    "setup_lighting",
    "setup_camera",
    # Materials
    "create_astronomical_material",
    "get_stellar_material",
    "create_galaxy_material",
    "create_nebula_material",
    "create_emission_material",
    "create_glass_material",
    "create_subsurface_material",
    # Objects
    "create_astronomical_object",
    "create_stellar_object",
    "create_galaxy_object",
    "create_point_cloud",
    # 2D plotting
    "GreasePencil2DPlotter",
    # Advanced features
    "create_galaxy_visualization",
    "create_nebula_visualization",
    "create_filament_network",
    "create_gravitational_lens",
    "setup_volumetric_rendering",
    "setup_post_processing",
    "create_particle_system",
    "create_custom_shader",
    "setup_gravity_simulation",
    # Core components
    "AstronomicalBlenderZeroCopyBridge",
    "AstronomicalOperators",
    # Convenience
    "quick_render",
    "create_publication_figure",
]
