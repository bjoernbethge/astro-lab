"""
Geometry Node Groups for AlbPy
=============================

Modern Blender 4.4 geometry node groups using factory functions.
"""

import logging
from typing import Any, Dict

import bpy

# Import individual geometry modules
from . import (
    elliptical_galaxy,
    filament,
    hr_diagram,
    pointcloud,
    spiral_galaxy,
)

logger = logging.getLogger(__name__)


def register():
    """Register all geometry node groups using factory functions."""
    logger.info("🏗️ Registering AlbPy Geometry Node Groups...")

    try:
        # Register core astronomical geometry nodes
        spiral_galaxy.register()
        elliptical_galaxy.register()
        filament.register()
        hr_diagram.register()
        pointcloud.register()

        # Additional geometry nodes (if implemented)
        modules_to_register = [
            # galactic_dust,
            # galaxy_comparison,
            # irregular_galaxy,
            # lens,
            # n_body_system,
            # orbital_trail,
            # particle,
            # planetary_nebula,
            # star_forming_region,
            # stellar_instances,
            # stellar_wind,
            # supernova_remnant,
        ]

        for module in modules_to_register:
            try:
                if hasattr(module, "register"):
                    module.register()
            except Exception as e:
                logger.warning(f"Failed to register {module.__name__}: {e}")

        logger.info("✅ Geometry Node Groups registered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to register Geometry Node Groups: {e}")
        raise


def unregister():
    """Unregister all geometry node groups."""
    logger.info("🧹 Unregistering AlbPy Geometry Node Groups...")

    try:
        # Unregister in reverse order
        pointcloud.unregister()
        hr_diagram.unregister()
        filament.unregister()
        elliptical_galaxy.unregister()
        spiral_galaxy.unregister()

        # Additional modules
        modules_to_unregister = [
            # supernova_remnant,
            # stellar_wind,
            # stellar_instances,
            # star_forming_region,
            # planetary_nebula,
            # particle,
            # orbital_trail,
            # n_body_system,
            # lens,
            # irregular_galaxy,
            # galaxy_comparison,
            # galactic_dust,
        ]

        for module in modules_to_unregister:
            try:
                if hasattr(module, "unregister"):
                    module.unregister()
            except Exception as e:
                logger.warning(f"Failed to unregister {module.__name__}: {e}")

        logger.info("✅ Geometry Node Groups unregistered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to unregister Geometry Node Groups: {e}")


def get_available_geometry_nodes() -> Dict[str, Any]:
    """
    Get list of available geometry node groups.

    Returns:
        Dict mapping node group names to their metadata
    """
    import bpy

    available_nodes = {}

    # Check which ALBPY geometry node groups exist
    albpy_geometry_nodes = [
        "ALBPY_SpiralGalaxy",
        "ALBPY_EllipticalGalaxy",
        "ALBPY_CosmicFilament",
        "ALBPY_HRDiagram",
        "ALBPY_PointCloudVisualization",
    ]

    for node_name in albpy_geometry_nodes:
        if node_name in bpy.data.node_groups:
            node_group = bpy.data.node_groups[node_name]
            available_nodes[node_name] = {
                "name": node_group.name,
                "type": node_group.type,
                "inputs": [
                    inp.name
                    for inp in node_group.interface.items_tree
                    if inp.in_out == "INPUT"
                ],
                "outputs": [
                    out.name
                    for out in node_group.interface.items_tree
                    if out.in_out == "OUTPUT"
                ],
                "description": getattr(
                    node_group, "description", "No description available"
                ),
            }

    return available_nodes


def create_geometry_modifier(
    obj: bpy.types.Object, node_group_name: str
) -> bpy.types.Modifier:
    """
    Add geometry nodes modifier with specified node group.

    Args:
        obj: Blender object to add modifier to
        node_group_name: Name of the geometry node group

    Returns:
        The created modifier
    """
    import bpy

    # Add geometry nodes modifier
    modifier = obj.modifiers.new(name=f"GeomNodes_{node_group_name}", type="NODES")

    # Assign node group if it exists
    if node_group_name in bpy.data.node_groups:
        modifier.node_group = bpy.data.node_groups[node_group_name]
    else:
        logger.warning(f"Node group '{node_group_name}' not found")

    return modifier


__all__ = [
    "register",
    "unregister",
    "get_available_geometry_nodes",
    "create_geometry_modifier",
]
