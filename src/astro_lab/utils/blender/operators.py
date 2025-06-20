"""
Astro-Lab (AL) Operator Registry & API
=======================================

Central registry for all Astro-Lab Blender operators and the primary
programmatic API for the AstroLabWidget.

This module achieves DRY by:
1. Defining a single Python API (`AstroLabApi`) that contains all functionality.
2. Creating thin Blender operator wrappers (`AL_OT_...`) that call the Python API.
3. Providing a single registration point for all Blender operators.

Access via:
- Widget: `widget.al.some_module.some_function()`
- Blender UI/Python: `bpy.ops.al.some_operator()`
"""

from typing import Set

try:
    import bpy
    from . import advanced as b3d_adv
    from .core import (
        create_camera, create_light, create_material, normalize_scene, reset_scene,
        create_cosmic_grid, create_text_legend
    )
    from .grease_pencil_2d import GreasePencil2DPlotter
    from .grease_pencil_3d import GreasePencil3DPlotter
    from .live_tensor_bridge import live_bridge

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None


# =============================================================================
# Programmatic Python API (for Widget and advanced scripting)
# =============================================================================

class AstroLabApi:
    """
    The main programmatic API for Astro-Lab's Blender functionalities.
    This class provides direct access to all visualization and simulation tools.
    """
    def __init__(self):
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender is not available. AstroLabApi cannot be initialized.")

        # Live Tensor-Socket Bridge
        self.live_bridge = live_bridge

        # Advanced Visualization Suite
        self.advanced = b3d_adv.AdvancedVisualizationSuite()

        # Core functionalities
        self.core = {
            "create_camera": create_camera,
            "create_light": create_light,
            "create_material": create_material,
            "reset_scene": reset_scene,
            "normalize_scene": normalize_scene,
            "create_cosmic_grid": create_cosmic_grid,
            "create_text_legend": create_text_legend,
        }

        # 2D and 3D Plotters
        self.plot_2d = GreasePencil2DPlotter()
        self.plot_3d = GreasePencil3DPlotter()

    def __repr__(self):
        return "AstroLabApi(live_bridge, advanced, core, plot_2d, plot_3d)"


# =============================================================================
# Blender Operator Wrappers (for Blender UI and simple scripting)
# =============================================================================

class AL_OT_CreateProceduralGalaxy(bpy.types.Operator):
    """Create a procedural galaxy using Astro-Lab's advanced tools."""
    bl_idname = "al.create_procedural_galaxy"
    bl_label = "AL: Create Procedural Galaxy"
    bl_description = "Generates a procedural galaxy with Geometry Nodes"
    bl_options = {'REGISTER', 'UNDO'}

    galaxy_type: bpy.props.EnumProperty(
        name="Galaxy Type",
        items=[
            ('spiral', "Spiral", "A spiral galaxy with arms"),
            ('elliptical', "Elliptical", "An elliptical galaxy"),
            ('irregular', "Irregular", "An irregular galaxy"),
        ],
        default='spiral'
    )
    num_stars: bpy.props.IntProperty(name="Number of Stars", default=50000, min=1000, max=500000)
    radius: bpy.props.FloatProperty(name="Radius", default=20.0, min=1.0, max=100.0)

    @classmethod
    def poll(cls, context):
        return BLENDER_AVAILABLE and b3d_adv.ADVANCED_AVAILABLE

    def execute(self, context):
        api = AstroLabApi()
        api.advanced.create_procedural_galaxy(
            galaxy_type=self.galaxy_type,
            num_stars=self.num_stars,
            radius=self.radius
        )
        self.report({'INFO'}, f"Created {self.galaxy_type} galaxy.")
        return {'FINISHED'}


class AL_OT_CreateEmissionNebula(bpy.types.Operator):
    """Create a volumetric emission nebula."""
    bl_idname = "al.create_emission_nebula"
    bl_label = "AL: Create Emission Nebula"
    bl_description = "Generates a volumetric emission nebula"
    bl_options = {'REGISTER', 'UNDO'}

    nebula_type: bpy.props.EnumProperty(
        name="Nebula Type",
        items=[
            ('h_alpha', "H-Alpha", "Hydrogen-alpha emission"),
            ('oxygen', "Oxygen-III", "Oxygen-III emission"),
            ('planetary', "Planetary", "A planetary nebula"),
        ],
        default='h_alpha'
    )
    size: bpy.props.FloatProperty(name="Size", default=15.0, min=1.0, max=100.0)
    density: bpy.props.FloatProperty(name="Density", default=0.2, min=0.01, max=1.0)

    @classmethod
    def poll(cls, context):
        return BLENDER_AVAILABLE and b3d_adv.ADVANCED_AVAILABLE

    def execute(self, context):
        api = AstroLabApi()
        api.advanced.create_emission_nebula_complex(
            nebula_type=self.nebula_type,
            size=self.size,
        )
        self.report({'INFO'}, f"Created {self.nebula_type} nebula.")
        return {'FINISHED'}


# Add more operators for other functionalities here...


# =============================================================================
# Registration
# =============================================================================

# List of all operator classes to register
operator_classes = [
    AL_OT_CreateProceduralGalaxy,
    AL_OT_CreateEmissionNebula,
]

def register():
    """Register all Astro-Lab operators and create the API."""
    if not BLENDER_AVAILABLE:
        return
    for cls in operator_classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister all Astro-Lab operators."""
    if not BLENDER_AVAILABLE:
        return
    for cls in reversed(operator_classes):
        bpy.utils.unregister_class(cls) 