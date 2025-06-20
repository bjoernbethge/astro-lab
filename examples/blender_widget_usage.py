#!/usr/bin/env python3
"""
🌌 Example: Using direct Blender API on AstroLab Widget

This example shows how to use the new direct Blender API access:
- widget.ops: Direct access to bpy.ops
- widget.data: Direct access to bpy.data  
- widget.context: Direct access to bpy.context
- widget.scene: Current scene
- widget.utils: Blender utilities
"""

from src.astro_lab.widgets.astro_lab import AstroLabWidget


def basic_blender_operations():
    """Basic Blender operations using widget.ops."""
    print("🔧 Basic Blender Operations")
    print("=" * 30)
    
    widget = AstroLabWidget(num_galaxies=100)
    
    if not widget.blender_available():
        print("❌ Blender not available")
        return
    
    # Create a simple scene
    widget.create_blender_scene()
    
    # Use widget.ops directly
    widget.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    widget.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))
    widget.ops.mesh.primitive_cylinder_add(location=(-2, 0, 0))
    
    print("✅ Created cube, sphere, and cylinder using widget.ops")


def working_with_objects():
    """Working with Blender objects using widget.data and widget.context."""
    print("\n📦 Working with Objects")
    print("=" * 25)
    
    widget = AstroLabWidget(num_galaxies=50)
    
    if not widget.blender_available():
        print("❌ Blender not available")
        return
    
    # Access objects through widget.data
    for obj in widget.data.objects:
        print(f"   Object: {obj.name} at {obj.location}")
    
    # Modify objects through widget.context
    if widget.data.objects:
        active_obj = widget.context.active_object
        if active_obj:
            active_obj.scale = (2, 2, 2)
            print(f"   Scaled {active_obj.name} to 2x size")


def camera_and_lighting():
    """Setting up camera and lighting using widget.utils."""
    print("\n📷 Camera and Lighting")
    print("=" * 22)
    
    widget = AstroLabWidget(num_galaxies=200)
    
    if not widget.blender_available():
        print("❌ Blender not available")
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
    
    print("✅ Created camera and lights using widget.utils")


def astronomical_visualization():
    """Complete astronomical visualization workflow."""
    print("\n🌌 Astronomical Visualization")
    print("=" * 30)
    
    widget = AstroLabWidget(num_galaxies=500)
    
    if not widget.blender_available():
        print("❌ Blender not available")
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
    
    print("✅ Complete astronomical visualization created")


def advanced_blender_operations():
    """Advanced Blender operations using the direct API."""
    print("\n🚀 Advanced Blender Operations")
    print("=" * 32)
    
    widget = AstroLabWidget(num_galaxies=100)
    
    if not widget.blender_available():
        print("❌ Blender not available")
        return
    
    # Create a complex scene
    widget.create_blender_scene()
    
    # Create a collection for our objects
    collection = widget.data.collections.new("AstronomicalObjects")
    widget.scene.collection.children.link(collection)
    
    # Create multiple objects and add to collection
    for i in range(5):
        widget.ops.mesh.primitive_uv_sphere_add(
            location=(i * 2, 0, 0),
            radius=0.5
        )
        sphere = widget.context.active_object
        sphere.name = f"galaxy_{i}"
        
        # Move object to our collection
        widget.scene.collection.objects.unlink(sphere)
        collection.objects.link(sphere)
    
    # Create material
    material = widget.data.materials.new(name="GalaxyMaterial")
    material.use_nodes = True
    
    # Set material properties
    principled_bsdf = material.node_tree.nodes["Principled BSDF"]
    principled_bsdf.inputs["Base Color"].default_value = (0.8, 0.2, 0.2, 1.0)
    principled_bsdf.inputs["Emission"].default_value = (0.2, 0.2, 0.8, 1.0)
    principled_bsdf.inputs["Emission Strength"].default_value = 2.0
    
    # Apply material to all objects in collection
    for obj in collection.objects:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    
    print("✅ Created complex scene with collections and materials")


if __name__ == "__main__":
    print("🌌 AstroLab Widget - Direct Blender API Examples")
    print("=" * 50)
    
    # Run all examples
    basic_blender_operations()
    working_with_objects()
    camera_and_lighting()
    astronomical_visualization()
    advanced_blender_operations()
    
    print("\n🎉 All examples completed!")
    print("\n💡 Key benefits of direct API access:")
    print("   - No need to import bpy separately")
    print("   - All Blender operations available via widget.ops")
    print("   - Direct access to scene, objects, materials via widget.data")
    print("   - Context management via widget.context")
    print("   - Utility functions via widget.utils") 