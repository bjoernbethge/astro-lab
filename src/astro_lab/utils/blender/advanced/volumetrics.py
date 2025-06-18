"""
Blender 4.4 Volumetric Astronomical Rendering Module
===================================================

Create realistic volumetric effects for astronomy:
- Nebula emission and absorption
- Stellar wind dynamics
- Galactic dust lanes
- Planetary atmospheres with scattering
- Interstellar medium visualization

Optimized for EEVEE Next volumetric rendering.

Author: Astro-Graph Agent
Version: 1.0.0
Blender: 4.4+
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector


class VolumetricAstronomy:
    """Create volumetric astronomical phenomena"""

    @staticmethod
    def create_emission_nebula(
        center: Vector = Vector((0, 0, 0)),
        size: float = 10.0,
        nebula_type: str = "h_alpha",
        density: float = 0.1,
    ) -> bpy.types.Object:
        """
        Create an emission nebula with realistic structure.

        Args:
            center: Nebula center position
            size: Nebula size
            nebula_type: Type ('h_alpha', 'oxygen', 'planetary', 'supernova')
            density: Base density value

        Returns:
            Created nebula object
        """
        # Create volume container
        bpy.ops.mesh.primitive_cube_add(location=center, size=size)
        nebula_obj = bpy.context.active_object
        nebula_obj.name = f"{nebula_type.title()}Nebula"

        # Apply volumetric material
        nebula_mat = VolumetricShaders.create_emission_nebula_material(nebula_type)
        nebula_obj.data.materials.append(nebula_mat)

        # Add geometry nodes for structure
        VolumetricAstronomy._add_nebula_structure_nodes(nebula_obj, density)

        return nebula_obj

    @staticmethod
    def create_stellar_wind(
        star_obj: bpy.types.Object,
        wind_speed: float = 500.0,
        mass_loss_rate: float = 1e-6,
        wind_radius: float = 5.0,
    ) -> bpy.types.Object:
        """
        Create stellar wind visualization around a star.

        Args:
            star_obj: Star object to create wind around
            wind_speed: Wind velocity in km/s
            mass_loss_rate: Mass loss rate in solar masses per year
            wind_radius: Maximum wind radius

        Returns:
            Created stellar wind object
        """
        # Create wind sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=star_obj.location, radius=wind_radius
        )
        wind_obj = bpy.context.active_object
        wind_obj.name = f"{star_obj.name}_StellarWind"

        # Apply stellar wind material
        wind_mat = VolumetricShaders.create_stellar_wind_material(wind_speed)
        wind_obj.data.materials.append(wind_mat)

        # Add particle system for wind dynamics
        VolumetricAstronomy._add_wind_particles(wind_obj, wind_speed, mass_loss_rate)

        return wind_obj

    @staticmethod
    def create_planetary_atmosphere(
        planet_obj: bpy.types.Object,
        atmosphere_type: str = "earth_like",
        thickness: float = 0.5,
    ) -> bpy.types.Object:
        """
        Create layered planetary atmosphere with scattering.

        Args:
            planet_obj: Planet object
            atmosphere_type: Type ('earth_like', 'mars_like', 'venus_like', 'gas_giant')
            thickness: Atmosphere thickness relative to planet radius

        Returns:
            Created atmosphere object
        """
        # Get planet scale
        planet_radius = max(planet_obj.scale)
        atmo_radius = planet_radius * (1.0 + thickness)

        # Create atmosphere sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=planet_obj.location, radius=atmo_radius
        )
        atmo_obj = bpy.context.active_object
        atmo_obj.name = f"{planet_obj.name}_Atmosphere"

        # Apply atmospheric material
        atmo_mat = VolumetricShaders.create_atmospheric_material(atmosphere_type)
        atmo_obj.data.materials.append(atmo_mat)

        return atmo_obj

    @staticmethod
    def create_galactic_dust_lane(
        start_pos: Vector,
        end_pos: Vector,
        width: float = 2.0,
        dust_density: float = 0.05,
    ) -> bpy.types.Object:
        """
        Create galactic dust lane with absorption and scattering.

        Args:
            start_pos: Start position of dust lane
            end_pos: End position of dust lane
            width: Width of the dust lane
            dust_density: Dust density

        Returns:
            Created dust lane object
        """
        # Calculate dust lane parameters
        direction = end_pos - start_pos
        length = direction.length
        center = (start_pos + end_pos) / 2

        # Create dust lane geometry
        bpy.ops.mesh.primitive_cube_add(location=center)
        dust_obj = bpy.context.active_object
        dust_obj.name = "GalacticDustLane"

        # Scale and orient
        dust_obj.scale = Vector((length, width, width * 0.5))
        dust_obj.rotation_euler = direction.to_track_quat("X", "Z").to_euler()

        # Apply dust material
        dust_mat = VolumetricShaders.create_dust_lane_material(dust_density)
        dust_obj.data.materials.append(dust_mat)

        return dust_obj

    @staticmethod
    def _add_nebula_structure_nodes(obj: bpy.types.Object, density: float) -> None:
        """Add geometry nodes for nebula structure."""
        # Add geometry nodes modifier
        modifier = obj.modifiers.new(name="NebulaStructure", type="NODES")
        tree = bpy.data.node_groups.new(name="NebulaStructure", type="GeometryNodeTree")
        modifier.node_group = tree

        nodes = tree.nodes
        links = tree.links

        # Clear and setup
        nodes.clear()
        tree.interface.clear()

        # Create interface
        tree.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        tree.interface.new_socket(
            name="Density", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        tree.interface.new_socket(
            name="Turbulence", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set defaults
        modifier["Input_2"] = density
        modifier["Input_3"] = 2.0  # Turbulence

        # Create nodes (simplified for space)
        input_node = nodes.new("NodeGroupInput")
        output_node = nodes.new("NodeGroupOutput")

        # Position nodes
        input_node.location = (-200, 0)
        output_node.location = (200, 0)

        # Connect
        links.new(input_node.outputs["Geometry"], output_node.inputs["Geometry"])

    @staticmethod
    def _add_wind_particles(
        obj: bpy.types.Object, wind_speed: float, mass_loss_rate: float
    ) -> None:
        """Add particle system for stellar wind."""
        # Add particle system
        particle_settings = bpy.data.particles.new("StellarWindParticles")
        particle_modifier = obj.modifiers.new("StellarWind", "PARTICLE_SYSTEM")
        particle_modifier.particle_system.settings = particle_settings

        # Configure particles
        particle_settings.type = "EMITTER"
        particle_settings.count = int(mass_loss_rate * 1e6)  # Scale appropriately
        particle_settings.emit_from = "VOLUME"
        particle_settings.normal_factor = wind_speed / 100.0
        particle_settings.lifetime = 100
        particle_settings.render_type = "NONE"  # Volume only


class VolumetricShaders:
    """Volumetric shaders for astronomical phenomena"""

    @staticmethod
    def create_emission_nebula_material(nebula_type: str) -> bpy.types.Material:
        """
        Create emission nebula material based on spectral lines.

        Args:
            nebula_type: Type of emission nebula

        Returns:
            Created material
        """
        mat = bpy.data.materials.new(name=f"Nebula_{nebula_type}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeVolumeEmission")

        # Spectral line colors
        spectral_colors = {
            "h_alpha": (0.8, 0.2, 0.2, 1.0),  # Red - Hydrogen alpha
            "oxygen": (0.2, 0.8, 0.3, 1.0),  # Green - Oxygen III
            "planetary": (0.3, 0.6, 1.0, 1.0),  # Blue-green mix
            "supernova": (1.0, 0.6, 0.2, 1.0),  # Orange-yellow
        }

        color = spectral_colors.get(nebula_type, spectral_colors["h_alpha"])
        emission.inputs["Color"].default_value = color

        # Add noise for structure
        noise1 = nodes.new("ShaderNodeTexNoise")
        noise1.inputs["Scale"].default_value = 2.0
        noise1.inputs["Detail"].default_value = 8.0

        noise2 = nodes.new("ShaderNodeTexNoise")
        noise2.inputs["Scale"].default_value = 0.5
        noise2.inputs["Detail"].default_value = 4.0

        # Combine noises
        multiply = nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"

        # Color ramp for density variation
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.2
        ramp.color_ramp.elements[1].position = 0.8

        # Position nodes
        noise1.location = (-600, 0)
        noise2.location = (-600, -200)
        multiply.location = (-400, -100)
        ramp.location = (-200, 0)
        emission.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        links.new(noise1.outputs["Fac"], multiply.inputs[0])
        links.new(noise2.outputs["Fac"], multiply.inputs[1])
        links.new(multiply.outputs["Value"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], emission.inputs["Density"])
        links.new(emission.outputs["Volume"], output.inputs["Volume"])

        # Set emission strength
        emission.inputs["Strength"].default_value = 2.0

        return mat

    @staticmethod
    def create_stellar_wind_material(wind_speed: float) -> bpy.types.Material:
        """
        Create stellar wind material with velocity-based opacity.

        Args:
            wind_speed: Wind velocity in km/s

        Returns:
            Created material
        """
        mat = bpy.data.materials.new(name="StellarWind")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        scatter = nodes.new("ShaderNodeVolumeScatter")

        # Wind color based on temperature
        if wind_speed > 1000:  # Fast wind - hot
            color = (0.8, 0.6, 1.0, 1.0)  # Blue-white
        elif wind_speed > 300:  # Medium wind
            color = (1.0, 0.8, 0.6, 1.0)  # Yellow-white
        else:  # Slow wind - cool
            color = (1.0, 0.6, 0.4, 1.0)  # Orange

        scatter.inputs["Color"].default_value = color

        # Radial falloff from center
        geometry = nodes.new("GeometryNodeInputPosition")
        vector_length = nodes.new("ShaderNodeVectorMath")
        vector_length.operation = "LENGTH"

        # Density falloff
        divide = nodes.new("ShaderNodeMath")
        divide.operation = "DIVIDE"
        divide.inputs[1].default_value = 5.0  # Falloff rate

        power = nodes.new("ShaderNodeMath")
        power.operation = "POWER"
        power.inputs[1].default_value = 2.0  # Square falloff

        subtract = nodes.new("ShaderNodeMath")
        subtract.operation = "SUBTRACT"
        subtract.inputs[0].default_value = 1.0

        # Position nodes
        geometry.location = (-400, -200)
        vector_length.location = (-200, -200)
        divide.location = (0, -200)
        power.location = (200, -200)
        subtract.location = (400, -200)
        scatter.location = (600, 0)
        output.location = (800, 0)

        # Connect nodes
        links.new(geometry.outputs["Position"], vector_length.inputs[0])
        links.new(vector_length.outputs["Value"], divide.inputs[0])
        links.new(divide.outputs["Value"], power.inputs[0])
        links.new(power.outputs["Value"], subtract.inputs[1])
        links.new(subtract.outputs["Value"], scatter.inputs["Density"])
        links.new(scatter.outputs["Volume"], output.inputs["Volume"])

        return mat

    @staticmethod
    def create_atmospheric_material(atmosphere_type: str) -> bpy.types.Material:
        """
        Create planetary atmosphere material with Rayleigh scattering.

        Args:
            atmosphere_type: Type of atmosphere

        Returns:
            Created material
        """
        mat = bpy.data.materials.new(name=f"Atmosphere_{atmosphere_type}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        scatter = nodes.new("ShaderNodeVolumeScatter")

        # Atmospheric compositions
        atmosphere_colors = {
            "earth_like": (0.3, 0.6, 1.0, 1.0),  # Blue - Rayleigh scattering
            "mars_like": (0.8, 0.5, 0.3, 1.0),  # Dusty orange
            "venus_like": (0.9, 0.8, 0.6, 1.0),  # Thick yellow
            "gas_giant": (0.7, 0.4, 0.2, 1.0),  # Banded colors
        }

        color = atmosphere_colors.get(atmosphere_type, atmosphere_colors["earth_like"])
        scatter.inputs["Color"].default_value = color

        # Density based on altitude (distance from center)
        geometry = nodes.new("GeometryNodeInputPosition")
        vector_length = nodes.new("ShaderNodeVectorMath")
        vector_length.operation = "LENGTH"

        # Exponential atmospheric falloff
        subtract = nodes.new("ShaderNodeMath")
        subtract.operation = "SUBTRACT"
        subtract.inputs[0].default_value = 2.0  # Atmosphere radius

        multiply = nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        multiply.inputs[1].default_value = -2.0  # Scale height

        power = nodes.new("ShaderNodeMath")
        power.operation = "POWER"
        power.inputs[0].default_value = math.e

        # Position nodes
        geometry.location = (-400, -200)
        vector_length.location = (-200, -200)
        subtract.location = (0, -200)
        multiply.location = (200, -200)
        power.location = (400, -200)
        scatter.location = (600, 0)
        output.location = (800, 0)

        # Connect nodes
        links.new(geometry.outputs["Position"], vector_length.inputs[0])
        links.new(vector_length.outputs["Value"], subtract.inputs[1])
        links.new(subtract.outputs["Value"], multiply.inputs[0])
        links.new(multiply.outputs["Value"], power.inputs[1])
        links.new(power.outputs["Value"], scatter.inputs["Density"])
        links.new(scatter.outputs["Volume"], output.inputs["Volume"])

        return mat

    @staticmethod
    def create_dust_lane_material(dust_density: float) -> bpy.types.Material:
        """
        Create galactic dust lane material with absorption.

        Args:
            dust_density: Dust density

        Returns:
            Created material
        """
        mat = bpy.data.materials.new(name="GalacticDust")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        absorption = nodes.new("ShaderNodeVolumeAbsorption")

        # Dust is brownish and absorbs blue light more
        absorption.inputs["Color"].default_value = (0.3, 0.2, 0.1, 1.0)
        absorption.inputs["Density"].default_value = dust_density

        # Add some scattering for realism
        scatter = nodes.new("ShaderNodeVolumeScatter")
        scatter.inputs["Color"].default_value = (0.4, 0.3, 0.2, 1.0)
        scatter.inputs["Density"].default_value = dust_density * 0.1

        # Mix absorption and scattering
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 0.9  # Mostly absorption

        # Position nodes
        absorption.location = (-200, 100)
        scatter.location = (-200, -100)
        mix.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        links.new(absorption.outputs["Volume"], mix.inputs[1])
        links.new(scatter.outputs["Volume"], mix.inputs[2])
        links.new(mix.outputs["Shader"], output.inputs["Volume"])

        return mat


# Example usage functions
def create_nebula_complex():
    """Create a complex nebula with multiple emission regions."""
    # Central emission nebula
    central_nebula = VolumetricAstronomy.create_emission_nebula(
        center=Vector((0, 0, 0)), size=15.0, nebula_type="h_alpha", density=0.2
    )

    # Oxygen emission region
    oxygen_region = VolumetricAstronomy.create_emission_nebula(
        center=Vector((3, 2, 1)), size=8.0, nebula_type="oxygen", density=0.15
    )

    # Surrounding dust
    dust_lane = VolumetricAstronomy.create_galactic_dust_lane(
        start_pos=Vector((-20, -5, -2)),
        end_pos=Vector((20, 5, 2)),
        width=3.0,
        dust_density=0.05,
    )

    print("Complex nebula scene created!")


def create_stellar_system_with_winds():
    """Create stellar system with stellar winds."""
    # Create central star
    bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
    star = bpy.context.active_object
    star.name = "CentralStar"
    star.scale = Vector([2, 2, 2])

    # Add stellar wind
    stellar_wind = VolumetricAstronomy.create_stellar_wind(
        star_obj=star, wind_speed=800.0, mass_loss_rate=1e-6, wind_radius=10.0
    )

    # Create planets with atmospheres
    for i, distance in enumerate([5, 8, 12]):
        bpy.ops.mesh.primitive_uv_sphere_add(location=(distance, 0, 0))
        planet = bpy.context.active_object
        planet.name = f"Planet_{i + 1}"
        planet.scale = Vector([0.5, 0.5, 0.5])

        # Add atmosphere
        if i == 1:  # Earth-like
            VolumetricAstronomy.create_planetary_atmosphere(
                planet_obj=planet, atmosphere_type="earth_like", thickness=0.3
            )

    print("Stellar system with winds created!")


if __name__ == "__main__":
    create_nebula_complex()
    create_stellar_system_with_winds()
