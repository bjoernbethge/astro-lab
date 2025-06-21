"""
Futuristic Astronomical Visualization Demo
=========================================

Blender script demonstrating all new advanced visualization features.
Run this in Blender's Python console or as a Blender script.

Features demonstrated:
- Post-Processing & Compositor effects
- Futuristic Materials (Iridescent, Glass, Metallic, Holographic)
- Visual Style Presets (luxury_teal, autumn, iridescent, etc.)
- High-Level API for easy scene creation
- Cinematic rendering with artistic effects

Author: Astro-Graph Agent
Version: 1.0.0
"""

import bpy
import numpy as np
from mathutils import Vector

# Import AstroLab modules
from astro_lab.utils.blender.advanced import (
    create_futuristic_scene,
    apply_material_preset,
    PostProcessingSuite,
    FuturisticMaterials,
    MaterialPresets,
    AdvancedVisualizationSuite,
)
from astro_lab.utils.blender.core import create_astro_object
from astro_lab.utils.blender.advanced.volumetrics import VolumetricAstronomy


def demo_luxury_teal_scene():
    """Create a luxury teal futuristic scene."""
    print("🎨 Creating Luxury Teal Scene...")
    
    # Create complete scene with one function call
    suite = create_futuristic_scene(
        scene_name="LuxuryTealScene",
        style="luxury_teal",
        use_post_processing=True
    )
    
    # Add some astronomical objects
    star = create_astro_object("star", position=[0, 0, 0], scale=2.0)
    if star:
        apply_material_preset("Star_Star", "golden_metallic")
    
    # Add a nebula
    nebula = VolumetricAstronomy.create_emission_nebula(
        center=Vector((5, 5, 0)),
        size=8.0,
        nebula_type="h_alpha"
    )
    
    # Add some planets with futuristic materials
    planet1 = create_astro_object("galaxy", position=[-8, 0, 0], scale=1.5)
    if planet1:
        apply_material_preset("Galaxy_Galaxy", "crystal_glass")
    
    planet2 = create_astro_object("galaxy", position=[8, 0, 0], scale=1.2)
    if planet2:
        apply_material_preset("Galaxy_Galaxy", "holographic_blue")
    
    print("✅ Luxury Teal Scene created with Post-Processing!")


def demo_autumn_scene():
    """Create a warm autumn scene."""
    print("🍂 Creating Autumn Scene...")
    
    suite = create_futuristic_scene(
        scene_name="AutumnScene",
        style="autumn",
        use_post_processing=True
    )
    
    # Add warm-colored objects
    star = create_astro_object("star", position=[0, 0, 0], scale=3.0)
    if star:
        # Create custom warm material
        warm_material = FuturisticMaterials.create_metallic_material(
            color=(1.0, 0.6, 0.2),  # Orange-gold
            metallic=0.8,
            roughness=0.2
        )
        star.data.materials.append(warm_material)
    
    # Add nebula with warm colors
    nebula = VolumetricAstronomy.create_emission_nebula(
        center=Vector((0, 0, 5)),
        size=10.0,
        nebula_type="supernova"  # Orange-yellow
    )
    
    print("✅ Autumn Scene created with warm colors!")


def demo_iridescent_scene():
    """Create an iridescent scene with holographic effects."""
    print("✨ Creating Iridescent Scene...")
    
    suite = create_futuristic_scene(
        scene_name="IridescentScene",
        style="iridescent",
        use_post_processing=True
    )
    
    # Add iridescent objects
    for i in range(5):
        angle = i * 2 * np.pi / 5
        x = 5 * np.cos(angle)
        y = 5 * np.sin(angle)
        
        obj = create_astro_object("galaxy", position=[x, y, 0], scale=0.8)
        if obj:
            # Create iridescent material with different shifts
            iridescent_mat = FuturisticMaterials.create_iridescent_material(
                base_color=(0.8, 0.2, 0.8),  # Purple base
                iridescence_strength=1.0,
                iridescence_shift=i * 0.2
            )
            obj.data.materials.append(iridescent_mat)
    
    # Add central holographic object
    center_obj = create_astro_object("star", position=[0, 0, 0], scale=2.0)
    if center_obj:
        apply_material_preset("Star_Star", "holographic_blue")
    
    print("✅ Iridescent Scene created with holographic effects!")


def demo_cinematic_closeup():
    """Create a dramatic cinematic closeup scene."""
    print("🎬 Creating Cinematic Closeup Scene...")
    
    suite = create_futuristic_scene(
        scene_name="CinematicScene",
        style="cinematic_closeup",
        use_post_processing=True
    )
    
    # Create a dramatic central object
    main_obj = create_astro_object("star", position=[0, 0, 0], scale=4.0)
    if main_obj:
        # Create energy field material
        energy_mat = FuturisticMaterials.create_energy_field_material(
            color=(0.6, 0.2, 1.0),  # Purple energy
            energy_strength=3.0,
            pulse_speed=2.0
        )
        main_obj.data.materials.append(energy_mat)
    
    # Add dramatic lighting elements
    for i in range(3):
        angle = i * 2 * np.pi / 3
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        
        light_obj = create_astro_object("galaxy", position=[x, y, 0], scale=0.5)
        if light_obj:
            # Create force field material
            force_mat = FuturisticMaterials.create_force_field_material(
                color=(0.8, 0.2, 1.0),
                field_strength=2.0,
                ripple_speed=3.0
            )
            light_obj.data.materials.append(force_mat)
    
    print("✅ Cinematic Closeup Scene created with dramatic effects!")


def demo_dreamy_pastel():
    """Create a dreamy pastel scene."""
    print("💫 Creating Dreamy Pastel Scene...")
    
    suite = create_futuristic_scene(
        scene_name="DreamyScene",
        style="dreamy_pastel",
        use_post_processing=True
    )
    
    # Add soft, pastel objects
    for i in range(8):
        angle = i * 2 * np.pi / 8
        radius = 4 + i * 0.5
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = i * 0.3
        
        obj = create_astro_object("galaxy", position=[x, y, z], scale=0.6)
        if obj:
            # Create glass material with pastel colors
            pastel_colors = [
                (0.95, 0.8, 0.9),  # Soft pink
                (0.8, 0.9, 0.95),  # Soft blue
                (0.9, 0.95, 0.8),  # Soft green
                (0.95, 0.9, 0.8),  # Soft yellow
            ]
            color = pastel_colors[i % len(pastel_colors)]
            
            glass_mat = FuturisticMaterials.create_glass_material(
                color=color,
                transmission=0.9,
                ior=1.4,
                roughness=0.1
            )
            obj.data.materials.append(glass_mat)
    
    print("✅ Dreamy Pastel Scene created with soft effects!")


def demo_custom_post_processing():
    """Demonstrate custom post-processing effects."""
    print("🎛️ Creating Custom Post-Processing Scene...")
    
    # Create basic scene
    suite = AdvancedVisualizationSuite("CustomPostScene")
    suite._initialize_scene()
    
    # Add some objects
    star = create_astro_object("star", position=[0, 0, 0], scale=2.0)
    
    # Apply custom post-processing
    post_processing = PostProcessingSuite("CustomPostScene")
    
    # Setup compositor
    post_processing.setup_compositor()
    
    # Add multiple effects
    post_processing.add_lens_flare("stellar", 1.2)
    post_processing.add_vignette(0.5, 0.6)
    post_processing.add_color_grading("dramatic")
    post_processing.add_star_glow(1.8, 15)
    post_processing.add_depth_of_field(8.0, 1.8)
    
    # Add artistic filters
    from astro_lab.utils.blender.advanced.post_processing import ArtisticFilters
    ArtisticFilters.add_film_grain(0.15)
    ArtisticFilters.add_chromatic_aberration(0.03)
    
    print("✅ Custom Post-Processing Scene created with multiple effects!")


def demo_material_showcase():
    """Showcase all futuristic materials."""
    print("💎 Creating Material Showcase...")
    
    suite = AdvancedVisualizationSuite("MaterialShowcase")
    suite._initialize_scene()
    
    # Create grid of objects with different materials
    materials = [
        ("luxury_teal", MaterialPresets.luxury_teal_material()),
        ("golden_metallic", MaterialPresets.golden_metallic_material()),
        ("crystal_glass", MaterialPresets.crystal_glass_material()),
        ("holographic_blue", MaterialPresets.holographic_blue_material()),
        ("energy_purple", MaterialPresets.energy_purple_material()),
    ]
    
    for i, (name, material) in enumerate(materials):
        x = (i - 2) * 4  # Spread objects horizontally
        
        obj = create_astro_object("galaxy", position=[x, 0, 0], scale=1.0)
        if obj:
            obj.data.materials.append(material)
            print(f"   Added {name} material")
    
    print("✅ Material Showcase created with all futuristic materials!")


def setup_eevee_next():
    """Setup EEVEE Next for optimal rendering."""
    print("⚡ Setting up EEVEE Next...")
    
    # Set render engine to EEVEE Next
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    
    # Configure EEVEE Next settings
    scene = bpy.context.scene
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = True
        scene.eevee.bloom_intensity = 1.0
        scene.eevee.bloom_radius = 6.0
    
    # Enable volumetrics
    if hasattr(scene.eevee, "volumetric_tile_size"):
        scene.eevee.volumetric_tile_size = "8"
        scene.eevee.volumetric_samples = 64
    
    # Color management
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "High Contrast"
    
    print("✅ EEVEE Next configured for optimal rendering!")


def main():
    """Run all demonstration scenes."""
    print("🚀 AstroLab Futuristic Visualization Demo")
    print("=" * 50)
    
    # Setup EEVEE Next
    setup_eevee_next()
    
    try:
        # Demo 1: Luxury Teal Scene
        demo_luxury_teal_scene()
        
        # Demo 2: Autumn Scene  
        demo_autumn_scene()
        
        # Demo 3: Iridescent Scene
        demo_iridescent_scene()
        
        # Demo 4: Cinematic Closeup
        demo_cinematic_closeup()
        
        # Demo 5: Dreamy Pastel
        demo_dreamy_pastel()
        
        # Demo 6: Custom Post-Processing
        demo_custom_post_processing()
        
        # Demo 7: Material Showcase
        demo_material_showcase()
        
        print("\n🎉 All demonstration scenes created successfully!")
        print("\n💡 Tips:")
        print("   - Use Blender's viewport to explore the scenes")
        print("   - Render with EEVEE Next for best results")
        print("   - Adjust post-processing settings in the Compositor")
        print("   - Experiment with different material presets")
        print("   - Switch between scenes in the Outliner")
        
    except Exception as e:
        print(f"❌ Error creating demo scenes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 