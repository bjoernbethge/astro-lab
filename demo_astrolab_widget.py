#!/usr/bin/env python3
"""
🌌 Demo: AstroLab Widget with direct Blender API

Shows the new direct Blender API integration:
- widget.ops: Blender Operations
- widget.data: Blender Data
- widget.context: Blender Context
- widget.scene: Current Scene
- widget.utils: Blender Utilities
"""

from src.astro_lab.widgets.astro_lab import AstroLabWidget


def demo_blender_api():
    """Demonstrates the direct Blender API on the widget."""
    print("🚀 AstroLab Widget with direct Blender API Demo")
    print("=" * 50)
    
    # Create widget
    widget = AstroLabWidget(num_galaxies=1000)
    
    # Check if Blender is available
    if widget.blender_available():
        print("✅ Blender API available!")
        print(f"   - widget.ops: {type(widget.ops)}")
        print(f"   - widget.data: {type(widget.data)}")
        print(f"   - widget.context: {type(widget.context)}")
        print(f"   - widget.scene: {type(widget.scene)}")
        print(f"   - widget.al: {type(widget.al)}")
        
        # Create Blender scene
        print("\n🎬 Creating Blender scene...")
        widget.create_blender_scene()
        
        # Add astronomical data
        print("📊 Adding astronomical data...")
        widget.add_astronomical_data_to_blender(point_size=0.05, use_colors=True)
        
        # Add camera and light
        print("📷 Creating camera and light...")
        camera = widget.al.core['create_camera'](location=(0, 0, 5))
        light = widget.al.core['create_light'](light_type='SUN', location=(5, 5, 5))
        
        # Export scene
        print("💾 Exporting scene...")
        widget.export_blender_scene("demo_astronomical_scene.blend")
        
        print("\n🎉 Demo completed!")
        print("Open 'demo_astronomical_scene.blend' in Blender")
        
    else:
        print("⚠️  Blender API not available")
        print("   Mock APIs created for development")
        
        # Show what happens when using the APIs
        try:
            widget.ops.mesh.primitive_cube_add()
        except Exception as e:
            print(f"   Mock API working: {e}")


def demo_pyvista_integration():
    """Demonstrates the PyVista integration."""
    print("\n🎨 PyVista Integration Demo")
    print("=" * 30)
    
    widget = AstroLabWidget(num_galaxies=500)
    
    # Create PyVista visualization
    plotter = widget.create_visualization()
    
    print("✅ PyVista visualization created")
    print("   Use plotter.show() to display")
    
    return plotter


if __name__ == "__main__":
    # Blender API demo
    demo_blender_api()
    
    # PyVista demo
    plotter = demo_pyvista_integration()
    
    print("\n🎯 Usage:")
    print("   widget.ops.mesh.primitive_cube_add()  # Create Blender cube")
    print("   widget.data.objects.new()             # New object")
    print("   widget.context.scene.objects.link()   # Link object to scene")
    print("   widget.utils.create_camera()          # Create camera")
    print("   plotter.show()                        # Show PyVista")
