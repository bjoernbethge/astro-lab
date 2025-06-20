"""
Blender 4.4 Advanced Astronomical Visualization Suite
====================================================

Professional astronomical visualization system leveraging Blender's unique capabilities:
- Procedural galaxy generation with Geometry Nodes
- Volumetric nebula rendering with emission spectra
- N-body gravitational simulations with orbital mechanics
- Scientifically accurate stellar and planetary shaders
- Real-time physics and atmospheric effects

Designed for scientific accuracy, performance, and visual excellence.

Author: Astro-Graph Agent
Version: 1.0.0
Blender: 4.4+
Dependencies: EEVEE Next, Geometry Nodes 4.4+
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

try:
    # Import all advanced modules
    from .geometry_nodes import AstronomicalMaterials, ProceduralAstronomy
    from .physics import GravitationalSimulation, OrbitalMechanics, PhysicsShaders
    from .shaders import AstronomicalShaders
    from .volumetrics import VolumetricAstronomy, VolumetricShaders

    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Advanced modules not available: {e}")
    ADVANCED_AVAILABLE = False


class AdvancedVisualizationSuite:
    """
    Main interface for advanced astronomical visualization.

    Combines all advanced Blender capabilities into a unified system
    for creating scientific astronomical visualizations.
    """

    def __init__(self, scene_name: str = "AstroAdvanced"):
        self.scene_name = scene_name
        self.scene_objects = {}

        if ADVANCED_AVAILABLE:
            self._initialize_scene()

    def _initialize_scene(self) -> None:
        """Initialize advanced scene with optimal settings."""
        # Set render engine to EEVEE Next
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"

        # Configure EEVEE Next for astronomy
        scene = bpy.context.scene
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = True
            scene.eevee.bloom_intensity = 0.8
            scene.eevee.bloom_radius = 6.5

        # Enable volumetrics
        if hasattr(scene.eevee, "volumetric_tile_size"):
            scene.eevee.volumetric_tile_size = "8"
            scene.eevee.volumetric_samples = 64
            scene.eevee.volumetric_start = 0.1
            scene.eevee.volumetric_end = 1000.0

        # Color management for space scenes
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "High Contrast"

        # World settings for deep space
        world = bpy.data.worlds.new("AstroWorld")
        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_nodes.clear()

        # Background shader for space
        background = world_nodes.new("ShaderNodeBackground")
        background.inputs["Color"].default_value = (0.01, 0.01, 0.02, 1.0)  # Deep space
        background.inputs["Strength"].default_value = 0.1

        output = world_nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(
            background.outputs["Background"], output.inputs["Surface"]
        )

        bpy.context.scene.world = world

        print(f"Advanced scene '{self.scene_name}' initialized with EEVEE Next")

    def create_procedural_galaxy(
        self,
        galaxy_type: str = "spiral",
        position: Vector = Vector((0, 0, 0)),
        num_stars: int = 50000,
        radius: float = 20.0,
    ) -> Optional[Any]:
        """
        Create procedural galaxy using Geometry Nodes.

        Args:
            galaxy_type: Type of galaxy ('spiral', 'elliptical', 'irregular')
            position: Galaxy center position
            num_stars: Number of stars to generate
            radius: Galaxy radius

        Returns:
            Created galaxy object
        """
        if not ADVANCED_AVAILABLE:
            print("Advanced modules not available")
            return None

        galaxy = ProceduralAstronomy.create_galaxy_structure(
            center=position, galaxy_type=galaxy_type, num_stars=num_stars, radius=radius
        )

        self.scene_objects[f"galaxy_{galaxy_type}"] = galaxy

        # Add appropriate stellar material
        if galaxy_type == "spiral":
            material = AstronomicalMaterials.create_stellar_classification_material("B")
        elif galaxy_type == "elliptical":
            material = AstronomicalMaterials.create_stellar_classification_material("K")
        else:  # irregular
            material = AstronomicalMaterials.create_stellar_classification_material("M")

        galaxy.data.materials.append(material)

        print(f"Created {galaxy_type} galaxy with {num_stars} stars")
        return galaxy

    def create_emission_nebula_complex(
        self,
        center: Vector = Vector((0, 0, 0)),
        nebula_type: str = "h_alpha",
        size: float = 15.0,
    ) -> Optional[Any]:
        """
        Create complex emission nebula with realistic volumetrics.

        Args:
            center: Nebula center position
            nebula_type: Type of emission ('h_alpha', 'oxygen', 'planetary')
            size: Nebula size

        Returns:
            Created nebula object
        """
        if not ADVANCED_AVAILABLE:
            print("Advanced modules not available")
            return None

        nebula = VolumetricAstronomy.create_emission_nebula(
            center=center, size=size, nebula_type=nebula_type, density=0.2
        )

        self.scene_objects[f"nebula_{nebula_type}"] = nebula

        print(f"Created {nebula_type} emission nebula")
        return nebula

    def create_orbital_system(
        self, star_mass: float = 1.0, planet_data: List[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Create realistic orbital system with physics.

        Args:
            star_mass: Central star mass in solar masses
            planet_data: List of planet parameters

        Returns:
            List of created objects [star, ...planets]
        """
        if not ADVANCED_AVAILABLE:
            print("Advanced modules not available")
            return []

        # Create central star
        bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
        star = bpy.context.active_object
        star.name = "CentralStar"
        star.scale = Vector(
            [
                math.pow(star_mass, 1 / 3) * 2,
                math.pow(star_mass, 1 / 3) * 2,
                math.pow(star_mass, 1 / 3) * 2,
            ]
        )

        # Add stellar material
        stellar_material = AstronomicalShaders.create_stellar_blackbody_shader(
            temperature=5778 * math.pow(star_mass, 0.5),
            luminosity=star_mass,
            stellar_class=None,
        )
        star.data.materials.append(stellar_material)

        # Default planet data if none provided
        if planet_data is None:
            planet_data = [
                {"radius": 5, "period": 100, "size": 0.3, "type": "terrestrial"},
                {"radius": 8, "period": 200, "size": 0.5, "type": "terrestrial"},
                {"radius": 15, "period": 500, "size": 2.0, "type": "gas_giant"},
                {"radius": 25, "period": 1000, "size": 1.5, "type": "ice_giant"},
            ]

        # Create orbital system
        orbits = [
            {"radius": p["radius"], "period": p["period"], "size": p["size"]}
            for p in planet_data
        ]

        planets = OrbitalMechanics.create_orbital_system(star, orbits)

        # Add planetary materials
        for i, (planet, data) in enumerate(zip(planets, planet_data)):
            planet_material = AstronomicalShaders.create_planetary_surface_shader(
                planet_type=data["type"]
            )
            planet.data.materials.append(planet_material)

            # Add atmosphere for suitable planets
            if data["type"] in ["terrestrial", "gas_giant", "ice_giant"]:
                atmosphere_type = (
                    "earth" if data["type"] == "terrestrial" else data["type"]
                )
                VolumetricAstronomy.create_planetary_atmosphere(
                    planet_obj=planet,
                    atmosphere_type=atmosphere_type,
                    thickness=0.3 if data["type"] == "terrestrial" else 0.1,
                )

        system_objects = [star] + planets
        self.scene_objects["orbital_system"] = system_objects

        print(f"Created orbital system with {len(planets)} planets")
        return system_objects

    def create_binary_star_system(
        self,
        primary_mass: float = 2.0,
        secondary_mass: float = 0.8,
        separation: float = 10.0,
    ) -> Tuple[Any, Any]:
        """
        Create gravitationally bound binary star system.

        Args:
            primary_mass: Primary star mass in solar masses
            secondary_mass: Secondary star mass in solar masses
            separation: Orbital separation in AU

        Returns:
            Tuple of (primary_star, secondary_star)
        """
        if not ADVANCED_AVAILABLE:
            print("Advanced modules not available")
            return None, None

        primary, secondary = GravitationalSimulation.create_binary_system(
            primary_mass=primary_mass,
            secondary_mass=secondary_mass,
            separation=separation,
        )

        # Add stellar winds
        for star in [primary, secondary]:
            VolumetricAstronomy.create_stellar_wind(
                star_obj=star, wind_speed=500.0, mass_loss_rate=1e-6, wind_radius=8.0
            )

        self.scene_objects["binary_system"] = [primary, secondary]

        print(f"Created binary system: {primary_mass}M☉ + {secondary_mass}M☉")
        return primary, secondary

    def create_hr_diagram_3d(
        self, stellar_data: List[Dict[str, float]] = None, scale_factor: float = 2.0
    ) -> Optional[Any]:
        """
        Create 3D Hertzsprung-Russell diagram.

        Args:
            stellar_data: List of stellar parameters
            scale_factor: Scale factor for the diagram

        Returns:
            Created HR diagram object
        """
        if not ADVANCED_AVAILABLE:
            print("Advanced modules not available")
            return None

        # Generate sample data if none provided
        if stellar_data is None:
            stellar_data = []
            import random

            for i in range(200):
                temp = random.uniform(3000, 30000)
                luminosity = math.pow(temp / 5778, 3.5) + random.uniform(-0.5, 0.5)
                mass = math.pow(temp / 5778, 0.7)

                stellar_data.append(
                    {"temperature": temp, "luminosity": luminosity, "mass": mass}
                )

        hr_diagram = ProceduralAstronomy.create_hr_diagram_3d(
            stellar_data=stellar_data, scale_factor=scale_factor
        )

        self.scene_objects["hr_diagram"] = hr_diagram

        print(f"Created 3D HR diagram with {len(stellar_data)} stars")
        return hr_diagram

    def setup_cinematic_camera(
        self,
        target: Vector = Vector((0, 0, 0)),
        distance: float = 30.0,
        height: float = 10.0,
    ) -> Any:
        """
        Setup cinematic camera for astronomical visualization.

        Args:
            target: Camera target position
            distance: Distance from target
            height: Height above target plane

        Returns:
            Created camera object
        """
        # Calculate camera position
        angle = math.radians(45)  # 45-degree angle
        camera_pos = Vector(
            (
                target.x + distance * math.cos(angle),
                target.y - distance * math.sin(angle),
                target.z + height,
            )
        )

        # Create camera
        bpy.ops.object.camera_add(location=camera_pos)
        camera = bpy.context.active_object
        camera.name = "AstroCinematicCamera"

        # Point at target
        direction = target - camera_pos
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        # Set camera properties
        camera.data.lens_unit = "FOV"
        camera.data.angle = math.radians(35)  # 35mm equivalent
        camera.data.clip_start = 0.1
        camera.data.clip_end = 10000.0
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = distance

        # Set as active camera
        bpy.context.scene.camera = camera

        self.scene_objects["camera"] = camera

        print(f"Setup cinematic camera at distance {distance}")
        return camera

    def setup_advanced_lighting(self, preset: str = "deep_space") -> List[Any]:
        """
        Setup advanced lighting for astronomical scenes.

        Args:
            preset: Lighting preset ('deep_space', 'nebula', 'planetary')

        Returns:
            List of created light objects
        """
        lights = []

        if preset == "deep_space":
            # Ambient starlight
            bpy.ops.object.light_add(type="SUN", location=(50, 50, 50))
            ambient = bpy.context.active_object
            ambient.name = "StarfieldAmbient"
            ambient.data.energy = 0.1
            ambient.data.color = (0.8, 0.9, 1.0)
            lights.append(ambient)

            # Key celestial light
            bpy.ops.object.light_add(type="SUN", location=(-30, -30, 40))
            key_light = bpy.context.active_object
            key_light.name = "CelestialKey"
            key_light.data.energy = 2.0
            key_light.data.color = (1.0, 0.9, 0.8)
            lights.append(key_light)

        elif preset == "nebula":
            # Warm nebula illumination
            bpy.ops.object.light_add(type="AREA", location=(20, 0, 20))
            nebula_light = bpy.context.active_object
            nebula_light.name = "NebulaIllumination"
            nebula_light.data.energy = 100.0
            nebula_light.data.color = (1.0, 0.6, 0.4)
            nebula_light.data.size = 10.0
            lights.append(nebula_light)

        else:  # planetary
            # Solar illumination
            bpy.ops.object.light_add(type="SUN", location=(100, 0, 50))
            solar = bpy.context.active_object
            solar.name = "SolarIllumination"
            solar.data.energy = 5.0
            solar.data.color = (1.0, 0.95, 0.9)
            lights.append(solar)

        self.scene_objects["lights"] = lights

        print(f"Setup {preset} lighting with {len(lights)} lights")
        return lights

    def render_scene(
        self,
        output_path: str,
        resolution: Tuple[int, int] = (1920, 1080),
        samples: int = 128,
    ) -> bool:
        """
        Render the advanced astronomical scene.

        Args:
            output_path: Output file path
            resolution: Render resolution (width, height)
            samples: Number of samples for quality

        Returns:
            True if successful
        """
        import os
        from pathlib import Path

        # Convert to absolute path relative to current working directory
        if not os.path.isabs(output_path):
            output_path = str(Path.cwd() / output_path)

        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set render settings
        scene = bpy.context.scene
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.filepath = output_path

        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = samples

        # Render
        try:
            bpy.ops.render.render(write_still=True)
            print(f"Rendered scene to {output_path}")
            return True
        except Exception as e:
            print(f"Render failed: {e}")
            return False


def initialize_advanced_scene(
    quality_preset: str = "high",
) -> AdvancedVisualizationSuite:
    """
    Initialize advanced astronomical visualization scene.

    Args:
        quality_preset: Quality preset ('low', 'medium', 'high', 'cinematic')

    Returns:
        Configured AdvancedVisualizationSuite instance
    """
    if not ADVANCED_AVAILABLE:
        print("Advanced modules not available - using stub")
        return AdvancedVisualizationSuite()

    # Clear existing scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Create suite
    suite = AdvancedVisualizationSuite("AstroAdvanced")

    # Configure quality settings
    scene = bpy.context.scene
    if quality_preset == "low":
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 32
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 720
    elif quality_preset == "medium":
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 64
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
    elif quality_preset == "high":
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 128
        scene.render.resolution_x = 2560
        scene.render.resolution_y = 1440
    else:  # cinematic
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 256
        scene.render.resolution_x = 3840
        scene.render.resolution_y = 2160

    print(f"Advanced scene initialized with {quality_preset} quality")
    return suite


# Convenience functions for quick scene creation
def create_galaxy_showcase() -> AdvancedVisualizationSuite:
    """Create showcase of different galaxy types."""
    suite = initialize_advanced_scene("high")

    # Create different galaxy types
    suite.create_procedural_galaxy("spiral", Vector((-20, 0, 0)), 30000, 15.0)
    suite.create_procedural_galaxy("elliptical", Vector((0, 0, 0)), 25000, 12.0)
    suite.create_procedural_galaxy("irregular", Vector((20, 0, 0)), 15000, 8.0)

    # Setup camera and lighting
    suite.setup_cinematic_camera(Vector((0, 0, 0)), 50.0, 15.0)
    suite.setup_advanced_lighting("deep_space")

    return suite


def create_nebula_showcase() -> AdvancedVisualizationSuite:
    """Create showcase of different nebula types."""
    suite = initialize_advanced_scene("high")

    # Create different nebula types
    suite.create_emission_nebula_complex(Vector((-15, 0, 0)), "h_alpha", 12.0)
    suite.create_emission_nebula_complex(Vector((0, 0, 0)), "oxygen", 10.0)
    suite.create_emission_nebula_complex(Vector((15, 0, 0)), "planetary", 8.0)

    # Setup camera and lighting
    suite.setup_cinematic_camera(Vector((0, 0, 0)), 30.0, 8.0)
    suite.setup_advanced_lighting("nebula")

    return suite


def create_stellar_system_showcase() -> AdvancedVisualizationSuite:
    """Create showcase stellar system with planets."""
    suite = initialize_advanced_scene("high")

    # Create orbital system
    suite.create_orbital_system(star_mass=1.2)

    # Add binary companion
    suite.create_binary_star_system(2.0, 0.8, 15.0)

    # Setup camera and lighting
    suite.setup_cinematic_camera(Vector((0, 0, 0)), 40.0, 12.0)
    suite.setup_advanced_lighting("planetary")

    return suite


if __name__ == "__main__":
    # Create demonstration scenes
    print("Creating advanced astronomical visualizations...")

    galaxy_suite = create_galaxy_showcase()
    print("Galaxy showcase created")

    # Can uncomment for additional showcases
    # nebula_suite = create_nebula_showcase()
    # stellar_suite = create_stellar_system_showcase()
