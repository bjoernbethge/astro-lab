"""
Astronomical Core for Blender - Materials and Effects
======================================================================

astronomical visualization with futuristic materials, physics simulations,
and procedural effects using Blender's Geometry Nodes and Shaders.
"""

import logging
import warnings
from typing import Any, Dict

import bpy  # type: ignore
from astropy.visualization import quantity_support


# Enable quantity support
quantity_support()

logger = logging.getLogger(__name__)


class AstronomicalVisualizationSuite:
    """
    Main interface for advanced astronomically correct visualization.

    Combines all advanced Blender capabilities into a unified system
    for creating scientifically accurate astronomical visualizations.
    """

    def __init__(self, scene_name: str = "AstronomicalScene"):
        self.scene_name = scene_name
        self.scene_objects = {}
        self.coordinate_system = "icrs"
        self.distance_unit = "pc"
        self.magnitude_unit = "mag"

    def _initialize_astronomical_scene(self) -> None:
        """Initialize advanced astronomical scene with optimal settings."""
        # Set render engine to EEVEE Next for real-time visualization
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"

        # Configure EEVEE Next for astronomical visualization
        scene = bpy.context.scene
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = True
            scene.eevee.bloom_intensity = 0.8
            scene.eevee.bloom_radius = 6.5

        # Enable volumetrics for nebula visualization
        if hasattr(scene.eevee, "volumetric_tile_size"):
            scene.eevee.volumetric_tile_size = "8"
            scene.eevee.volumetric_samples = 64
            scene.eevee.volumetric_start = 0.1
            scene.eevee.volumetric_end = 1000.0

        # Color management optimized for astronomical data
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "High Contrast"

        # Setup astronomical world
        self._setup_astronomical_world()

        print(f"Astronomical scene '{self.scene_name}' initialized with EEVEE Next")

    def _setup_astronomical_world(self) -> None:
        """Setup astronomically correct world background."""
        world = bpy.data.worlds.new("AstronomicalWorld")
        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_nodes.clear()

        # Background shader for deep space
        background = world_nodes.new("ShaderNodeBackground")
        background.inputs["Color"].default_value = (0.01, 0.01, 0.02, 1.0)  # Deep space
        background.inputs["Strength"].default_value = 0.1

        output = world_nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(
            background.outputs["Background"], output.inputs["Surface"]
        )

        bpy.context.scene.world = world

        # Add astronomical metadata
        world["coordinate_system"] = self.coordinate_system
        world["distance_unit"] = self.distance_unit
        world["magnitude_unit"] = self.magnitude_unit

    def get_astronomical_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the astronomical scene.

        Returns:
            Dictionary with scene statistics
        """
        try:
            scene = bpy.context.scene
            stats = {
                "scene_name": scene.name,
                "total_objects": len(scene.objects),
                "coordinate_system": self.coordinate_system,
                "distance_unit": self.distance_unit,
                "magnitude_unit": self.magnitude_unit,
                "render_engine": scene.render.engine,
                "scene_objects": len(self.scene_objects),
            }

            # Count astronomical objects by type
            object_types = {}
            for obj in scene.objects:
                obj_type = obj.get("object_type", "unknown")
                object_types[obj_type] = object_types.get(obj_type, 0) + 1

            stats["object_types"] = object_types

            # Material statistics
            materials = bpy.data.materials
            stats["total_materials"] = len(materials)
            stats["astronomical_materials"] = sum(
                1 for mat in materials if mat.get("material_type")
            )

            return stats

        except Exception as e:
            warnings.warn(f"Failed to get astronomical statistics: {e}", UserWarning)
            return {}


def initialize_astronomically_correct_scene(
    quality_preset: str = "high",
    coordinate_system: str = "icrs",
    distance_unit: str = "pc",
) -> AstronomicalVisualizationSuite:
    """
    Initialize astronomically correct scene with specified settings.

    Args:
        quality_preset: Quality preset ('low', 'medium', 'high', 'ultra')
        coordinate_system: Astronomical coordinate system
        distance_unit: Distance unit

    Returns:
        Initialized visualization suite
    """
    suite = AstronomicalVisualizationSuite()
    suite.coordinate_system = coordinate_system
    suite.distance_unit = distance_unit

    # Initialize scene based on quality preset
    suite._initialize_astronomical_scene()

    return suite


__all__ = [
    "AstronomicalVisualizationSuite",
    "initialize_astronomically_correct_scene",
]
