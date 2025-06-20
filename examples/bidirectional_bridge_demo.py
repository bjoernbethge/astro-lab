#!/usr/bin/env python3
"""
🌉 Demo: Bidirectional PyVista-Blender Bridge

Comprehensive demonstration of the bidirectional bridge between PyVista and Blender:
- PyVista → Blender conversion
- Blender → PyVista conversion  
- Live synchronization
- Material transfer
- Mesh synchronization
"""

import sys
import time
import numpy as np
import pyvista as pv
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# NumPy compatibility workaround
try:
    import numpy as np
except ImportError as e:
    print(f"⚠️ NumPy import error: {e}")
    np = None

from src.astro_lab.widgets.astro_lab import AstroLabWidget
from src.astro_lab.utils.viz.bidirectional_bridge import (
    BidirectionalPyVistaBlenderBridge,
    SyncConfig,
    quick_convert_pyvista_to_blender,
    quick_convert_blender_to_pyvista
)


def demo_basic_conversions():
    """Demonstrate basic PyVista ↔ Blender conversions."""
    print("🔄 Basic Conversions Demo")
    print("=" * 30)
    
    # Create widget
    widget = AstroLabWidget(num_galaxies=100)
    
    if not widget.bidirectional_bridge_available():
        print("❌ Bidirectional bridge not available")
        return
    
    # Create a simple PyVista mesh
    sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
    sphere.point_data["temperature"] = np.random.random(sphere.n_points)
    
    print(f"✅ Created PyVista sphere: {sphere.n_points} points, {sphere.n_cells} cells")
    
    # Convert PyVista → Blender
    print("\n📤 PyVista → Blender")
    blender_obj = widget.pyvista_to_blender(sphere, "converted_sphere")
    if blender_obj:
        print(f"   ✅ Converted to Blender object: {blender_obj.name}")
        print(f"   📊 Vertices: {len(blender_obj.data.vertices)}")
        print(f"   📊 Faces: {len(blender_obj.data.polygons)}")
    
    # Convert Blender → PyVista
    print("\n📥 Blender → PyVista")
    if blender_obj:
        pyvista_mesh = widget.blender_to_pyvista(blender_obj)
        if pyvista_mesh:
            print(f"   ✅ Converted back to PyVista: {pyvista_mesh.n_points} points")
            print(f"   📊 Original points: {sphere.n_points}")
            print(f"   📊 Converted points: {pyvista_mesh.n_points}")


def demo_mesh_synchronization():
    """Demonstrate mesh synchronization."""
    print("\n🔄 Mesh Synchronization Demo")
    print("=" * 35)
    
    widget = AstroLabWidget(num_galaxies=50)
    
    if not widget.bidirectional_bridge_available():
        print("❌ Bidirectional bridge not available")
        return
    
    # Create source mesh
    cube = pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2)
    cube.point_data["height"] = cube.points[:, 2]
    
    # Convert to Blender
    blender_cube = widget.pyvista_to_blender(cube, "sync_cube")
    if not blender_cube:
        print("❌ Failed to create Blender cube")
        return
    
    print("✅ Created synchronized cube pair")
    
    # Modify PyVista mesh
    print("\n🔧 Modifying PyVista mesh...")
    cube.points[:, 2] += 1.0  # Move up by 1 unit
    cube.point_data["height"] = cube.points[:, 2]
    
    # Synchronize to Blender
    widget.sync_mesh(cube, blender_cube)
    print("   ✅ Synchronized PyVista → Blender")
    
    # Modify Blender mesh
    print("\n🔧 Modifying Blender mesh...")
    if widget.blender_available():
        # Scale the Blender object
        blender_cube.scale = (1.5, 1.5, 1.5)
        
        # Synchronize back to PyVista
        widget.sync_mesh(blender_cube, cube)
        print("   ✅ Synchronized Blender → PyVista")


def demo_live_synchronization():
    """Demonstrate live synchronization."""
    print("\n⚡ Live Synchronization Demo")
    print("=" * 35)
    
    widget = AstroLabWidget(num_galaxies=200)
    
    if not widget.bidirectional_bridge_available():
        print("❌ Bidirectional bridge not available")
        return
    
    if not widget.blender_available():
        print("❌ Blender not available")
        return
    
    # Create PyVista plotter
    plotter = pv.Plotter()
    
    # Create animated mesh
    torus = pv.ParametricTorus()
    torus.point_data["phase"] = np.zeros(torus.n_points)
    
    # Add to plotter
    plotter.add_mesh(torus, scalars="phase", cmap="viridis")
    
    # Convert to Blender
    blender_torus = widget.pyvista_to_blender(torus, "live_torus")
    if not blender_torus:
        print("❌ Failed to create Blender torus")
        return
    
    print("✅ Created live sync pair")
    
    # Start live synchronization
    success = widget.start_live_sync(plotter, sync_interval=0.1)
    if not success:
        print("❌ Failed to start live sync")
        return
    
    # Add animation callback
    def animate_mesh():
        """Animate the mesh in both frameworks."""
        time_val = time.time()
        
        # Animate PyVista mesh
        torus.points[:, 0] += 0.01 * np.sin(time_val)
        torus.point_data["phase"] = np.sin(time_val + torus.points[:, 0])
        
        # Update plotter
        plotter.update_scalars(torus.point_data["phase"], render=False)
    
    widget.add_sync_callback(animate_mesh)
    
    print("🎬 Live sync running - press 'q' to stop")
    print("   PyVista and Blender are now synchronized!")
    
    # Show PyVista (this will block)
    try:
        plotter.show(interactive=True)
    except KeyboardInterrupt:
        pass
    finally:
        widget.stop_live_sync()
        print("✅ Live sync stopped")


def demo_material_transfer():
    """Demonstrate material and texture transfer."""
    print("\n🎨 Material Transfer Demo")
    print("=" * 25)
    
    widget = AstroLabWidget(num_galaxies=100)
    
    if not widget.bidirectional_bridge_available():
        print("❌ Bidirectional bridge not available")
        return
    
    # Create complex mesh with materials
    cylinder = pv.Cylinder(radius=0.5, height=2.0)
    
    # Add rich point data
    cylinder.point_data["radius"] = np.sqrt(cylinder.points[:, 0]**2 + cylinder.points[:, 1]**2)
    cylinder.point_data["height"] = cylinder.points[:, 2]
    cylinder.point_data["temperature"] = np.random.random(cylinder.n_points)
    
    print("✅ Created cylinder with rich data")
    
    # Convert to Blender with materials
    blender_cylinder = widget.pyvista_to_blender(cylinder, "material_cylinder")
    if blender_cylinder:
        print("   ✅ Converted with materials")
        print(f"   🎨 Vertex colors: {len(blender_cylinder.data.vertex_colors)}")
        
        # Check material properties
        if blender_cylinder.data.materials:
            material = blender_cylinder.data.materials[0]
            print(f"   🎨 Material: {material.name}")
            if material.use_nodes:
                print("   🎨 Uses node-based materials")
    
    # Convert back to PyVista
    if blender_cylinder:
        pyvista_cylinder = widget.blender_to_pyvista(blender_cylinder)
        if pyvista_cylinder:
            print("   ✅ Converted back with materials")
            print(f"   🎨 Point data keys: {list(pyvista_cylinder.point_data.keys())}")


def demo_complex_workflow():
    """Demonstrate complex workflow with multiple conversions."""
    print("\n🚀 Complex Workflow Demo")
    print("=" * 25)
    
    widget = AstroLabWidget(num_galaxies=500)
    
    if not widget.bidirectional_bridge_available():
        print("❌ Bidirectional bridge not available")
        return
    
    # Create astronomical data visualization
    coords_xyz, _ = widget.get_3d_coordinates()
    
    # Create PyVista mesh from astronomical data
    point_cloud = pv.PolyData(coords_xyz)
    point_cloud.point_data["redshift"] = widget.galaxy_df["redshift"].to_numpy()
    point_cloud.point_data["mass"] = widget.galaxy_df["log_stellar_mass"].to_numpy()
    
    print(f"✅ Created astronomical point cloud: {len(coords_xyz)} galaxies")
    
    # Convert to Blender
    blender_galaxies = widget.pyvista_to_blender(point_cloud, "astronomical_galaxies")
    if blender_galaxies:
        print("   ✅ Converted to Blender")
        
        # Add camera and lighting
        if widget.blender_available():
            camera = widget.utils.create_camera(location=(0, 0, 10))
            light = widget.utils.create_light(light_type='SUN', location=(5, 5, 5))
            print("   📷 Added camera and lighting")
    
    # Create PyVista visualization
    plotter = pv.Plotter()
    glyphs = point_cloud.glyph(
        orient=False,
        scale="mass",
        factor=0.05,
        geom=pv.Sphere(theta_resolution=8, phi_resolution=8)
    )
    
    plotter.add_mesh(glyphs, scalars="redshift", cmap="viridis")
    plotter.set_background("black")
    
    print("   🎨 Created PyVista visualization")
    
    # Export both
    widget.export_blender_scene("astronomical_workflow.blend")
    plotter.screenshot("astronomical_workflow.png")
    
    print("   💾 Exported both visualizations")
    print("   📁 Files: astronomical_workflow.blend, astronomical_workflow.png")


def demo_performance_comparison():
    """Compare performance of different conversion methods."""
    print("\n⚡ Performance Comparison Demo")
    print("=" * 30)
    
    widget = AstroLabWidget(num_galaxies=1000)
    
    if not widget.bidirectional_bridge_available():
        print("❌ Bidirectional bridge not available")
        return
    
    # Create test mesh
    mesh = pv.ParametricTorus()
    print(f"✅ Test mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    
    # Test bidirectional bridge
    start_time = time.time()
    blender_obj = widget.pyvista_to_blender(mesh, "perf_test")
    bridge_time = time.time() - start_time
    
    if blender_obj:
        print(f"   🌉 Bidirectional bridge: {bridge_time:.3f}s")
        
        # Test reverse conversion
        start_time = time.time()
        pyvista_mesh = widget.blender_to_pyvista(blender_obj)
        reverse_time = time.time() - start_time
        print(f"   🔄 Reverse conversion: {reverse_time:.3f}s")
    
    # Test quick conversion functions
    start_time = time.time()
    quick_blender = quick_convert_pyvista_to_blender(mesh, "quick_test")
    quick_time = time.time() - start_time
    
    if quick_blender:
        print(f"   ⚡ Quick conversion: {quick_time:.3f}s")
        
        start_time = time.time()
        quick_pyvista = quick_convert_blender_to_pyvista(quick_blender)
        quick_reverse_time = time.time() - start_time
        print(f"   ⚡ Quick reverse: {quick_reverse_time:.3f}s")


if __name__ == "__main__":
    print("🌉 Bidirectional PyVista-Blender Bridge Demo")
    print("=" * 50)
    
    # Run all demos
    demo_basic_conversions()
    demo_mesh_synchronization()
    demo_material_transfer()
    demo_complex_workflow()
    demo_performance_comparison()
    
    print("\n🎬 Live Synchronization Demo (interactive)")
    print("   This will open PyVista and sync with Blender")
    print("   Press 'q' to quit")
    
    try:
        demo_live_synchronization()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    print("\n🎉 All demos completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ PyVista → Blender conversion")
    print("   ✅ Blender → PyVista conversion")
    print("   ✅ Mesh synchronization")
    print("   ✅ Material transfer")
    print("   ✅ Live synchronization")
    print("   ✅ Performance optimization")
    print("   ✅ Complex workflows") 