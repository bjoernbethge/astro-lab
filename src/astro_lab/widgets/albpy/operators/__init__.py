"""
AlbPy Operators
===============

Modern Blender 4.4 operators for astronomical object creation and manipulation.
"""

import logging
from typing import Any, Dict, List, Type

import bpy

# Import operator modules
from . import (
    filament,
    galaxy,
    hr_diagram,
    planet,
    pointcloud,
    star,
)

logger = logging.getLogger(__name__)


def register():
    """Register all AlbPy operators."""
    logger.info("🎮 Registering AlbPy Operators...")

    try:
        # Register core operators
        star.register()
        galaxy.register()
        planet.register()
        pointcloud.register()
        hr_diagram.register()
        filament.register()

        logger.info("✅ AlbPy Operators registered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to register AlbPy Operators: {e}")
        raise


def unregister():
    """Unregister all AlbPy operators."""
    logger.info("🧹 Unregistering AlbPy Operators...")

    try:
        # Unregister modules in reverse order
        filament.unregister()
        hr_diagram.unregister()
        pointcloud.unregister()
        planet.unregister()
        galaxy.unregister()
        star.unregister()

        logger.info("✅ AlbPy Operators unregistered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to unregister AlbPy Operators: {e}")


def get_operator_classes() -> List[Type[bpy.types.Operator]]:
    """Get list of all AlbPy operator classes."""
    operator_classes = [
        # Star operators
        star.ALBPY_OT_CreateStar,
        star.ALBPY_OT_CreateStarField,
        # Galaxy operators
        galaxy.ALBPY_OT_CreateGalaxy,
        galaxy.ALBPY_OT_CreateGalaxyCluster,
        # Planet operators
        planet.ALBPY_OT_CreatePlanet,
        planet.ALBPY_OT_CreatePlanetarySystem,
        # Point cloud operators
        pointcloud.ALBPY_OT_CreatePointCloud,
        pointcloud.ALBPY_OT_ImportAstronomicalData,
        # HR Diagram operators
        hr_diagram.ALBPY_OT_CreateHRDiagram,
        hr_diagram.ALBPY_OT_AnimateHRDiagram,
        # Filament operators
        filament.ALBPY_OT_CreateFilament,
        filament.ALBPY_OT_CreateCosmicWeb,
    ]

    return operator_classes


def get_available_operators() -> List[str]:
    """Get list of available operator identifiers."""
    return [
        # Star creation
        "albpy.create_star",
        "albpy.create_star_field",
        # Galaxy creation
        "albpy.create_galaxy",
        "albpy.create_galaxy_cluster",
        # Planet creation
        "albpy.create_planet",
        "albpy.create_planetary_system",
        # Point cloud visualization
        "albpy.create_pointcloud",
        "albpy.import_astronomical_data",
        # HR Diagram
        "albpy.create_hr_diagram",
        "albpy.animate_hr_diagram",
        # Cosmic structure
        "albpy.create_filament",
        "albpy.create_cosmic_web",
    ]


def create_astronomical_scene(
    scene_type: str = "stellar_field", **kwargs
) -> Dict[str, Any]:
    """
    Create complete astronomical scene using operators.

    Args:
        scene_type: Type of scene to create
        **kwargs: Additional parameters for scene creation

    Returns:
        Dict containing created objects and metadata
    """
    import bpy

    # Clear existing scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    created_objects = []
    metadata = {"scene_type": scene_type}

    if scene_type == "stellar_field":
        # Create star field
        bpy.ops.albpy.create_star_field(
            count=kwargs.get("star_count", 500),
            distribution=kwargs.get("distribution", "DISK"),
            radius=kwargs.get("radius", 50.0),
            mixed_classes=kwargs.get("mixed_classes", True),
        )
        created_objects.extend(
            [obj for obj in bpy.context.scene.objects if obj.name.startswith("Star")]
        )

    elif scene_type == "galaxy_cluster":
        # Create galaxy cluster
        bpy.ops.albpy.create_galaxy_cluster(
            galaxy_count=kwargs.get("galaxy_count", 20),
            cluster_radius=kwargs.get("cluster_radius", 100.0),
            mixed_types=kwargs.get("mixed_types", True),
        )
        created_objects.extend(
            [obj for obj in bpy.context.scene.objects if obj.name.startswith("Galaxy")]
        )

    elif scene_type == "planetary_system":
        # Create planetary system
        bpy.ops.albpy.create_planetary_system(
            planet_count=kwargs.get("planet_count", 8),
            system_radius=kwargs.get("system_radius", 30.0),
            include_moons=kwargs.get("include_moons", True),
        )
        created_objects.extend(
            [obj for obj in bpy.context.scene.objects if "Planet" in obj.name]
        )

    elif scene_type == "cosmic_web":
        # Create cosmic web structure
        bpy.ops.albpy.create_cosmic_web(
            grid_size=kwargs.get("grid_size", 100),
            filament_density=kwargs.get("filament_density", 0.3),
            node_count=kwargs.get("node_count", 50),
        )
        created_objects.extend(
            [obj for obj in bpy.context.scene.objects if "Filament" in obj.name]
        )

    elif scene_type == "hr_diagram":
        # Create 3D HR diagram
        bpy.ops.albpy.create_hr_diagram(
            star_count=kwargs.get("star_count", 2000),
            include_evolution=kwargs.get("include_evolution", True),
            animate=kwargs.get("animate", False),
        )
        created_objects.extend(
            [obj for obj in bpy.context.scene.objects if "HR" in obj.name]
        )

    else:
        logger.warning(f"Unknown scene type: {scene_type}")
        return {"error": f"Unknown scene type: {scene_type}"}

    # Setup lighting and camera for astronomical scenes
    setup_astronomical_lighting()
    setup_astronomical_camera(scene_type)

    metadata.update(
        {
            "objects_created": len(created_objects),
            "object_names": [obj.name for obj in created_objects],
            "scene_parameters": kwargs,
        }
    )

    logger.info(f"✅ Created {scene_type} scene with {len(created_objects)} objects")

    return {"objects": created_objects, "metadata": metadata, "scene_type": scene_type}


def setup_astronomical_lighting() -> None:
    """Setup lighting appropriate for astronomical scenes."""
    import bpy

    # Remove default light if it exists
    if "Light" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)

    # Create astronomical lighting setup

    # Weak ambient light (starlight)
    ambient_light = bpy.data.lights.new("Ambient_Starlight", "SUN")
    ambient_light.energy = 0.01
    ambient_light.color = (0.9, 0.95, 1.0)  # Slightly blue

    ambient_obj = bpy.data.objects.new("Ambient_Starlight", ambient_light)
    bpy.context.collection.objects.link(ambient_obj)
    ambient_obj.location = (0, 0, 100)

    # Optional: Galactic center glow (very weak)
    if len([obj for obj in bpy.context.scene.objects if "Galaxy" in obj.name]) > 0:
        galactic_light = bpy.data.lights.new("Galactic_Glow", "POINT")
        galactic_light.energy = 100
        galactic_light.color = (1.0, 0.9, 0.7)  # Warm glow

        galactic_obj = bpy.data.objects.new("Galactic_Glow", galactic_light)
        bpy.context.collection.objects.link(galactic_obj)
        galactic_obj.location = (0, 0, 0)


def setup_astronomical_camera(scene_type: str) -> None:
    """Setup camera for astronomical scene."""
    import bpy

    # Get or create camera
    if "Camera" in bpy.data.objects:
        camera_obj = bpy.data.objects["Camera"]
    else:
        camera_data = bpy.data.cameras.new("AstroCamera")
        camera_obj = bpy.data.objects.new("AstroCamera", camera_data)
        bpy.context.collection.objects.link(camera_obj)

    camera = camera_obj.data

    # Camera settings for astronomy
    camera.lens = 50  # Standard lens
    camera.sensor_width = 36  # Full frame sensor
    camera.clip_start = 0.1
    camera.clip_end = 10000  # Large distance for cosmic scales

    # Position based on scene type
    if scene_type == "stellar_field":
        camera_obj.location = (0, -80, 20)
        camera_obj.rotation_euler = (1.3, 0, 0)  # Looking up at stars

    elif scene_type == "galaxy_cluster":
        camera_obj.location = (150, -150, 100)
        camera_obj.rotation_euler = (0.8, 0, 0.785)  # Angled view

    elif scene_type == "planetary_system":
        camera_obj.location = (0, -50, 15)
        camera_obj.rotation_euler = (1.4, 0, 0)  # Slightly above orbital plane

    elif scene_type == "cosmic_web":
        camera_obj.location = (200, -200, 150)
        camera_obj.rotation_euler = (0.6, 0, 0.785)  # Wide view

    elif scene_type == "hr_diagram":
        camera_obj.location = (30, -30, 20)
        camera_obj.rotation_euler = (1.1, 0, 0.785)  # 3D view of diagram

    # Set as active camera
    bpy.context.scene.camera = camera_obj


def create_operator_menu() -> bpy.types.Menu:
    """Create menu for AlbPy operators."""

    class ALBPY_MT_OperatorMenu(bpy.types.Menu):
        bl_label = "AlbPy Astronomical Objects"
        bl_idname = "ALBPY_MT_operator_menu"

        def draw(self, context):
            layout = self.layout

            # Star creation
            layout.label(text="Stars & Stellar Objects")
            layout.operator("albpy.create_star", text="Single Star")
            layout.operator("albpy.create_star_field", text="Star Field")

            layout.separator()

            # Galaxy creation
            layout.label(text="Galaxies & Large Scale")
            layout.operator("albpy.create_galaxy", text="Single Galaxy")
            layout.operator("albpy.create_galaxy_cluster", text="Galaxy Cluster")
            layout.operator("albpy.create_cosmic_web", text="Cosmic Web")

            layout.separator()

            # Planet creation
            layout.label(text="Planets & Solar Systems")
            layout.operator("albpy.create_planet", text="Single Planet")
            layout.operator("albpy.create_planetary_system", text="Planetary System")

            layout.separator()

            # Visualization tools
            layout.label(text="Data Visualization")
            layout.operator("albpy.create_pointcloud", text="Point Cloud")
            layout.operator("albpy.create_hr_diagram", text="HR Diagram")
            layout.operator("albpy.import_astronomical_data", text="Import Data")

            layout.separator()

            # Complete scenes
            layout.label(text="Complete Scenes")
            scene_op = layout.operator(
                "albpy.create_complete_scene", text="Stellar Field Scene"
            )
            scene_op.scene_type = "stellar_field"

            scene_op = layout.operator(
                "albpy.create_complete_scene", text="Galaxy Cluster Scene"
            )
            scene_op.scene_type = "galaxy_cluster"

    return ALBPY_MT_OperatorMenu


class ALBPY_OT_CreateCompleteScene(bpy.types.Operator):
    """Create complete astronomical scene with lighting and camera."""

    bl_idname = "albpy.create_complete_scene"
    bl_label = "Create Complete Astronomical Scene"
    bl_description = (
        "Create a complete astronomical scene with objects, lighting, and camera"
    )
    bl_options = {"REGISTER", "UNDO"}

    scene_type: bpy.props.EnumProperty(
        name="Scene Type",
        description="Type of astronomical scene to create",
        items=[
            (
                "stellar_field",
                "Stellar Field",
                "Field of stars with realistic distribution",
            ),
            ("galaxy_cluster", "Galaxy Cluster", "Cluster of galaxies"),
            ("planetary_system", "Planetary System", "Solar system with planets"),
            ("cosmic_web", "Cosmic Web", "Large-scale structure filaments"),
            ("hr_diagram", "HR Diagram", "3D Hertzsprung-Russell diagram"),
        ],
        default="stellar_field",
    )

    # Common parameters
    object_count: bpy.props.IntProperty(
        name="Object Count",
        description="Number of objects to create",
        default=100,
        min=10,
        max=10000,
    )

    scale: bpy.props.FloatProperty(
        name="Scene Scale",
        description="Overall scale of the scene",
        default=50.0,
        min=1.0,
        max=1000.0,
    )

    def execute(self, context):
        """Execute scene creation."""
        try:
            # Create scene based on type
            result = create_astronomical_scene(
                scene_type=self.scene_type,
                star_count=self.object_count if "star" in self.scene_type else None,
                galaxy_count=self.object_count if "galaxy" in self.scene_type else None,
                planet_count=min(self.object_count, 20)
                if "planet" in self.scene_type
                else None,
                radius=self.scale,
                cluster_radius=self.scale if "cluster" in self.scene_type else None,
                system_radius=self.scale if "system" in self.scene_type else None,
                grid_size=int(self.scale) if "web" in self.scene_type else None,
            )

            if "error" in result:
                self.report({"ERROR"}, result["error"])
                return {"CANCELLED"}

            objects_created = len(result["objects"])
            self.report(
                {"INFO"},
                f"Created {self.scene_type} scene with {objects_created} objects",
            )

            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to create scene: {str(e)}")
            return {"CANCELLED"}


def register_menu():
    """Register AlbPy menu in Blender."""
    menu_class = create_operator_menu()
    bpy.utils.register_class(menu_class)
    bpy.utils.register_class(ALBPY_OT_CreateCompleteScene)

    # Add to Add menu
    def draw_menu(self, context):
        self.layout.menu("ALBPY_MT_operator_menu")

    bpy.types.VIEW3D_MT_add.append(draw_menu)


def unregister_menu():
    """Unregister AlbPy menu from Blender."""
    menu_class = create_operator_menu()

    # Remove from Add menu
    def draw_menu(self, context):
        self.layout.menu("ALBPY_MT_operator_menu")

    bpy.types.VIEW3D_MT_add.remove(draw_menu)

    bpy.utils.unregister_class(ALBPY_OT_CreateCompleteScene)
    bpy.utils.unregister_class(menu_class)


# Enhanced registration with menu
def register():
    """Register all AlbPy operators and menu."""
    logger.info("🎮 Registering AlbPy Operators...")

    try:
        # Register core operators
        star.register()
        galaxy.register()
        planet.register()
        pointcloud.register()
        hr_diagram.register()
        filament.register()

        # Register menu
        register_menu()

        logger.info("✅ AlbPy Operators and menu registered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to register AlbPy Operators: {e}")
        raise


def unregister():
    """Unregister all AlbPy operators and menu."""
    logger.info("🧹 Unregistering AlbPy Operators...")

    try:
        # Unregister menu first
        unregister_menu()

        # Unregister modules in reverse order
        filament.unregister()
        hr_diagram.unregister()
        pointcloud.unregister()
        planet.unregister()
        galaxy.unregister()
        star.unregister()

        logger.info("✅ AlbPy Operators unregistered successfully")

    except Exception as e:
        logger.error(f"❌ Failed to unregister AlbPy Operators: {e}")


__all__ = [
    "register",
    "unregister",
    "get_operator_classes",
    "get_available_operators",
    "create_astronomical_scene",
    "setup_astronomical_lighting",
    "setup_astronomical_camera",
    "create_operator_menu",
    "ALBPY_OT_CreateCompleteScene",
]
