#!/usr/bin/env python3
"""
Blender Widget Example
======================

Demonstrates integration with Blender for 3D astronomical visualization.
Shows how to use the AstroLab widget with direct Blender API access.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from astro_lab.widgets.astro_lab import AstroLabWidget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_basic_blender_operations():
    """Demonstrate basic Blender operations using widget.ops."""
    logger.info("🔧 Basic Blender Operations")
    logger.info("=" * 35)
    
    widget = AstroLabWidget(num_galaxies=100)
    
    if not widget.blender_available():
        logger.warning("❌ Blender not available")
        return
    
    # Create a simple scene
    widget.create_blender_scene()
    
    # Use widget.ops directly
    widget.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    widget.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))
    widget.ops.mesh.primitive_cylinder_add(location=(-2, 0, 0))
    
    logger.info("✅ Created cube, sphere, and cylinder using widget.ops")


def demonstrate_object_manipulation():
    """Demonstrate object manipulation using widget.data and widget.context."""
    logger.info("📦 Object Manipulation")
    logger.info("=" * 30)
    
    widget = AstroLabWidget(num_galaxies=50)
    
    if not widget.blender_available():
        logger.warning("❌ Blender not available")
        return
    
    # Access objects through widget.data
    for obj in widget.data.objects:
        logger.info(f"   Object: {obj.name} at {obj.location}")
    
    # Modify objects through widget.context
    if widget.data.objects:
        active_obj = widget.context.active_object
        if active_obj:
            active_obj.scale = (2, 2, 2)
            logger.info(f"   Scaled {active_obj.name} to 2x size")


def demonstrate_camera_and_lighting():
    """Demonstrate camera and lighting setup using widget.utils."""
    logger.info("📷 Camera and Lighting Setup")
    logger.info("=" * 35)
    
    widget = AstroLabWidget(num_galaxies=200)
    
    if not widget.blender_available():
        logger.warning("❌ Blender not available")
        return
    
    # Create camera using utilities
    camera = widget.utils.create_camera(
        location=(5, -5, 3), 
        target=(0, 0, 0)
    )
    
    # Create different types of lights
    sun_light = widget.utils.create_light(
        light_type='SUN', 
        location=(10, 10, 10)
    )
    
    point_light = widget.utils.create_light(
        light_type='POINT', 
        location=(-5, -5, 5)
    )
    
    logger.info("✅ Created camera and lights using widget.utils")


def demonstrate_astronomical_visualization():
    """Demonstrate complete astronomical visualization workflow."""
    logger.info("🌌 Astronomical Visualization")
    logger.info("=" * 35)
    
    widget = AstroLabWidget(num_galaxies=500)
    
    if not widget.blender_available():
        logger.warning("❌ Blender not available")
        return
    
    # Setup scene
    widget.create_blender_scene()
    
    # Add astronomical data
    widget.add_astronomical_data_to_blender(
        point_size=0.02, 
        use_colors=True
    )
    
    # Setup camera and lighting
    camera = widget.utils.create_camera(
        location=(0, 0, 10), 
        target=(0, 0, 0)
    )
    
    widget.utils.create_light(
        light_type='SUN', 
        location=(5, 5, 5)
    )
    
    # Export the scene
    widget.export_blender_scene("astronomical_visualization.blend")
    
    logger.info("✅ Complete astronomical visualization created")


def main():
    """Main Blender widget demonstration."""
    logger.info("🌌 AstroLab Blender Widget Example")
    logger.info("=" * 45)
    
    try:
        # Run demonstrations
        demonstrate_basic_blender_operations()
        demonstrate_object_manipulation()
        demonstrate_camera_and_lighting()
        demonstrate_astronomical_visualization()
        
        logger.info("🎉 All Blender demonstrations completed!")
        
    except Exception as e:
        logger.error(f"❌ Blender demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 