"""
Blender 4.4 Astronomical Shader System Module
============================================

Create physically accurate shaders for astronomy:
- Stellar classification and blackbody radiation
- Planetary surface materials with composition
- Nebula emission line spectra
- Atmospheric scattering (Rayleigh/Mie)
- Cosmic dust and interstellar medium

Optimized for scientific accuracy and visual appeal.

Author: Astro-Graph Agent
Version: 1.0.0
Blender: 4.4+
"""

import os
import warnings
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from .. import numpy_compat  # noqa: F401
import bpy

# Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

try:
    import bmesh
    import numpy as np
    from mathutils import Euler, Matrix, Vector
    BPY_AVAILABLE = bpy is not None
except ImportError as e:
    print(f"Blender modules not available: {e}")
    BPY_AVAILABLE = False
    bpy = None
    bmesh = None
    Vector = None


class AstronomicalShaders:
    """Create scientifically accurate astronomical shaders"""

    @staticmethod
    def create_stellar_blackbody_shader(
        temperature: float, luminosity: float = 1.0, stellar_class: Optional[str] = None
    ) -> bpy.types.Material:
        """
        Create physically accurate stellar shader based on blackbody radiation.

        Args:
            temperature: Stellar temperature in Kelvin
            luminosity: Stellar luminosity relative to Sun
            stellar_class: Optional spectral class (O, B, A, F, G, K, M)

        Returns:
            Created stellar material
        """
        # Determine spectral class if not provided
        if stellar_class is None:
            stellar_class = AstronomicalShaders._classify_star_by_temperature(
                temperature
            )

        mat = bpy.data.materials.new(name=f"Star_{stellar_class}_{temperature}K")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        blackbody = nodes.new("ShaderNodeBlackbody")

        # Set temperature
        blackbody.inputs["Temperature"].default_value = temperature

        # Calculate emission strength from luminosity
        # Using Stefan-Boltzmann law approximation
        strength = luminosity * 10.0  # Base strength scaling
        emission.inputs["Strength"].default_value = strength

        # Add stellar surface variation
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 50.0
        noise.inputs["Detail"].default_value = 8.0

        # Color variation for stellar activity
        color_mix = nodes.new("ShaderNodeMixRGB")
        color_mix.blend_type = "ADD"
        color_mix.inputs["Fac"].default_value = 0.1

        # Hot spots for stellar activity
        math_node = nodes.new("ShaderNodeMath")
        math_node.operation = "MULTIPLY"
        math_node.inputs[1].default_value = 1.2  # Activity scaling

        # Position nodes
        blackbody.location = (-400, 100)
        noise.location = (-400, -200)
        color_mix.location = (-200, 0)
        math_node.location = (-200, -300)
        emission.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        links.new(blackbody.outputs["Color"], color_mix.inputs["Color1"])
        links.new(noise.outputs["Color"], color_mix.inputs["Color2"])
        links.new(color_mix.outputs["Color"], emission.inputs["Color"])
        links.new(noise.outputs["Fac"], math_node.inputs[0])
        links.new(math_node.outputs["Value"], emission.inputs["Strength"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        return mat

    @staticmethod
    def create_planetary_surface_shader(
        planet_type: str, composition: Dict[str, float] = None
    ) -> bpy.types.Material:
        """
        Create planetary surface shader based on composition.

        Args:
            planet_type: Type of planet ('terrestrial', 'gas_giant', 'ice_giant', 'moon')
            composition: Compositional percentages {'rock': 0.7, 'ice': 0.3, etc.}

        Returns:
            Created planetary material
        """
        mat = bpy.data.materials.new(name=f"Planet_{planet_type}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

        # Default compositions
        if composition is None:
            composition = AstronomicalShaders._get_default_composition(planet_type)

        # Base color from composition
        base_color = AstronomicalShaders._calculate_surface_color(composition)
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)

        # Surface properties based on planet type
        if planet_type == "terrestrial":
            bsdf.inputs["Roughness"].default_value = 0.8
            bsdf.inputs["Metallic"].default_value = 0.1
            bsdf.inputs["Specular"].default_value = 0.3

        elif planet_type == "gas_giant":
            # Gas giants have cloudy, banded surfaces
            bsdf.inputs["Roughness"].default_value = 0.3
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Specular"].default_value = 0.8

            # Add cloud bands
            AstronomicalShaders._add_gas_giant_bands(nodes, links, bsdf)

        elif planet_type == "ice_giant":
            bsdf.inputs["Roughness"].default_value = 0.1
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Specular"].default_value = 1.0
            bsdf.inputs["Transmission"].default_value = 0.3

        else:  # moon
            bsdf.inputs["Roughness"].default_value = 0.9
            bsdf.inputs["Metallic"].default_value = 0.05
            bsdf.inputs["Specular"].default_value = 0.1

        # Add surface texture variation
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 5.0
        noise.inputs["Detail"].default_value = 6.0

        # Mix with base color
        color_mix = nodes.new("ShaderNodeMixRGB")
        color_mix.blend_type = "MULTIPLY"
        color_mix.inputs["Fac"].default_value = 0.3

        # Position nodes
        noise.location = (-400, -200)
        color_mix.location = (-200, 0)
        bsdf.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        links.new(noise.outputs["Color"], color_mix.inputs["Color2"])
        links.new(color_mix.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        return mat

    @staticmethod
    def create_nebula_emission_shader(
        emission_lines: List[str], density_variation: float = 0.5
    ) -> bpy.types.Material:
        """
        Create nebula shader based on emission line spectra.

        Args:
            emission_lines: List of emission lines ['H_alpha', 'O_III', 'H_beta', etc.]
            density_variation: Amount of density variation (0-1)

        Returns:
            Created nebula material
        """
        mat = bpy.data.materials.new(name=f"Nebula_{'_'.join(emission_lines)}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeVolumeEmission")

        # Calculate combined color from emission lines
        combined_color = AstronomicalShaders._combine_emission_lines(emission_lines)
        emission.inputs["Color"].default_value = (*combined_color, 1.0)

        # Add density variation with multiple noise scales
        noise1 = nodes.new("ShaderNodeTexNoise")
        noise1.inputs["Scale"].default_value = 1.0
        noise1.inputs["Detail"].default_value = 8.0

        noise2 = nodes.new("ShaderNodeTexNoise")
        noise2.inputs["Scale"].default_value = 5.0
        noise2.inputs["Detail"].default_value = 4.0

        # Combine noises for complex structure
        multiply = nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"

        # Color ramp for density falloff
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.1
        ramp.color_ramp.elements[1].position = 0.9

        # Emission strength variation
        strength_variation = nodes.new("ShaderNodeMath")
        strength_variation.operation = "MULTIPLY"
        strength_variation.inputs[1].default_value = 2.0  # Base strength

        # Position nodes
        noise1.location = (-600, 0)
        noise2.location = (-600, -200)
        multiply.location = (-400, -100)
        ramp.location = (-200, 0)
        strength_variation.location = (-200, -300)
        emission.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        links.new(noise1.outputs["Fac"], multiply.inputs[0])
        links.new(noise2.outputs["Fac"], multiply.inputs[1])
        links.new(multiply.outputs["Value"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], emission.inputs["Density"])
        links.new(multiply.outputs["Value"], strength_variation.inputs[0])
        links.new(strength_variation.outputs["Value"], emission.inputs["Strength"])
        links.new(emission.outputs["Volume"], output.inputs["Volume"])

        return mat

    @staticmethod
    def create_atmospheric_scattering_shader(
        atmosphere_type: str, scale_height: float = 8.5
    ) -> bpy.types.Material:
        """
        Create atmospheric scattering shader (Rayleigh + Mie).

        Args:
            atmosphere_type: Type of atmosphere ('earth', 'mars', 'venus', 'titan')
            scale_height: Atmospheric scale height in km

        Returns:
            Created atmospheric material
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

        # Atmospheric parameters
        atmo_params = AstronomicalShaders._get_atmospheric_parameters(atmosphere_type)

        # Set scattering color (Rayleigh scattering favors blue)
        scatter.inputs["Color"].default_value = (*atmo_params["color"], 1.0)

        # Altitude-dependent density
        geometry = nodes.new("GeometryNodeInputPosition")
        vector_length = nodes.new("ShaderNodeVectorMath")
        vector_length.operation = "LENGTH"

        # Exponential density falloff
        subtract = nodes.new("ShaderNodeMath")
        subtract.operation = "SUBTRACT"
        subtract.inputs[0].default_value = atmo_params["radius"]

        divide = nodes.new("ShaderNodeMath")
        divide.operation = "DIVIDE"
        divide.inputs[1].default_value = scale_height

        power = nodes.new("ShaderNodeMath")
        power.operation = "POWER"
        power.inputs[0].default_value = math.e

        multiply_negative = nodes.new("ShaderNodeMath")
        multiply_negative.operation = "MULTIPLY"
        multiply_negative.inputs[1].default_value = -1.0

        # Base density
        multiply_density = nodes.new("ShaderNodeMath")
        multiply_density.operation = "MULTIPLY"
        multiply_density.inputs[1].default_value = atmo_params["density"]

        # Position nodes
        geometry.location = (-600, -200)
        vector_length.location = (-400, -200)
        subtract.location = (-200, -200)
        multiply_negative.location = (0, -300)
        divide.location = (0, -200)
        power.location = (200, -200)
        multiply_density.location = (400, -100)
        scatter.location = (600, 0)
        output.location = (800, 0)

        # Connect nodes
        links.new(geometry.outputs["Position"], vector_length.inputs[0])
        links.new(vector_length.outputs["Value"], subtract.inputs[1])
        links.new(subtract.outputs["Value"], divide.inputs[0])
        links.new(divide.outputs["Value"], multiply_negative.inputs[0])
        links.new(multiply_negative.outputs["Value"], power.inputs[1])
        links.new(power.outputs["Value"], multiply_density.inputs[0])
        links.new(multiply_density.outputs["Value"], scatter.inputs["Density"])
        links.new(scatter.outputs["Volume"], output.inputs["Volume"])

        return mat

    @staticmethod
    def _classify_star_by_temperature(temperature: float) -> str:
        """Classify star by temperature."""
        if temperature >= 30000:
            return "O"
        elif temperature >= 10000:
            return "B"
        elif temperature >= 7500:
            return "A"
        elif temperature >= 6000:
            return "F"
        elif temperature >= 5200:
            return "G"
        elif temperature >= 3700:
            return "K"
        else:
            return "M"

    @staticmethod
    def _get_default_composition(planet_type: str) -> Dict[str, float]:
        """Get default planetary composition."""
        compositions = {
            "terrestrial": {"rock": 0.8, "metal": 0.15, "ice": 0.05},
            "gas_giant": {"hydrogen": 0.75, "helium": 0.24, "methane": 0.01},
            "ice_giant": {"water": 0.65, "methane": 0.25, "ammonia": 0.10},
            "moon": {"rock": 0.9, "metal": 0.05, "ice": 0.05},
        }
        return compositions.get(planet_type, compositions["terrestrial"])

    @staticmethod
    def _calculate_surface_color(
        composition: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """Calculate surface color from composition."""
        # Color mapping for different materials
        material_colors = {
            "rock": (0.4, 0.3, 0.2),
            "metal": (0.6, 0.6, 0.5),
            "ice": (0.8, 0.9, 1.0),
            "hydrogen": (0.9, 0.8, 0.7),
            "helium": (0.8, 0.8, 0.9),
            "methane": (0.7, 0.9, 0.8),
            "water": (0.2, 0.4, 0.8),
            "ammonia": (0.9, 0.9, 0.7),
        }

        # Weighted average of colors
        r = g = b = 0.0
        for material, fraction in composition.items():
            if material in material_colors:
                color = material_colors[material]
                r += color[0] * fraction
                g += color[1] * fraction
                b += color[2] * fraction

        return (r, g, b)

    @staticmethod
    def _combine_emission_lines(
        emission_lines: List[str],
    ) -> Tuple[float, float, float]:
        """Combine emission line colors."""
        # Emission line wavelengths and colors
        line_colors = {
            "H_alpha": (0.8, 0.2, 0.2),  # 656.3 nm - Red
            "H_beta": (0.4, 0.6, 1.0),  # 486.1 nm - Blue
            "O_III": (0.2, 0.8, 0.3),  # 500.7 nm - Green
            "N_II": (0.8, 0.4, 0.2),  # 658.3 nm - Orange
            "S_II": (0.6, 0.2, 0.4),  # 671.6 nm - Purple
            "He_II": (0.4, 0.8, 1.0),  # 468.6 nm - Cyan
        }

        # Average the colors
        r = g = b = 0.0
        count = len(emission_lines)

        for line in emission_lines:
            if line in line_colors:
                color = line_colors[line]
                r += color[0]
                g += color[1]
                b += color[2]

        if count > 0:
            return (r / count, g / count, b / count)
        else:
            return (1.0, 0.5, 0.3)  # Default nebula color

    @staticmethod
    def _get_atmospheric_parameters(atmosphere_type: str) -> Dict[str, Any]:
        """Get atmospheric parameters for different worlds."""
        parameters = {
            "earth": {"color": (0.3, 0.6, 1.0), "density": 0.1, "radius": 1.0},
            "mars": {"color": (0.8, 0.5, 0.3), "density": 0.01, "radius": 0.53},
            "venus": {"color": (0.9, 0.8, 0.6), "density": 0.9, "radius": 0.95},
            "titan": {"color": (0.7, 0.6, 0.4), "density": 0.15, "radius": 0.4},
        }
        return parameters.get(atmosphere_type, parameters["earth"])

    @staticmethod
    def _add_gas_giant_bands(nodes, links, bsdf) -> None:
        """Add banded structure to gas giant shader."""
        # Coordinate system for bands
        coords = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")

        # Scale Y coordinate for bands
        mapping.inputs["Scale"].default_value = (1.0, 10.0, 1.0)

        # Wave texture for bands
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.inputs["Scale"].default_value = 2.0
        wave.inputs["Distortion"].default_value = 0.5

        # Color ramp for band colors
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.8, 0.6, 0.4, 1.0)  # Light band
        ramp.color_ramp.elements[1].color = (0.4, 0.3, 0.2, 1.0)  # Dark band

        # Position nodes
        coords.location = (-800, -400)
        mapping.location = (-600, -400)
        wave.location = (-400, -400)
        ramp.location = (-200, -400)

        # Connect nodes
        links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        links.new(wave.outputs["Color"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])


# Example usage functions
def create_stellar_showcase():
    """Create showcase of different stellar types."""
    stellar_types = [
        {"temp": 30000, "class": "O", "pos": (-10, 0, 0)},
        {"temp": 20000, "class": "B", "pos": (-6, 0, 0)},
        {"temp": 8500, "class": "A", "pos": (-2, 0, 0)},
        {"temp": 6500, "class": "F", "pos": (2, 0, 0)},
        {"temp": 5500, "class": "G", "pos": (6, 0, 0)},
        {"temp": 4000, "class": "K", "pos": (10, 0, 0)},
        {"temp": 3000, "class": "M", "pos": (14, 0, 0)},
    ]

    stars = []
    for star_data in stellar_types:
        # Create star
        bpy.ops.mesh.primitive_uv_sphere_add(location=star_data["pos"])
        star = bpy.context.active_object
        star.name = f"Star_{star_data['class']}"

        # Add stellar material
        stellar_mat = AstronomicalShaders.create_stellar_blackbody_shader(
            temperature=star_data["temp"],
            luminosity=1.0,
            stellar_class=star_data["class"],
        )
        star.data.materials.append(stellar_mat)

        stars.append(star)

    print("Stellar showcase created!")
    return stars


def create_planetary_system():
    """Create system with different planetary types."""
    planet_types = [
        {"type": "terrestrial", "pos": (0, -5, 0), "scale": 0.5},
        {"type": "gas_giant", "pos": (0, 0, 0), "scale": 2.0},
        {"type": "ice_giant", "pos": (0, 5, 0), "scale": 1.5},
        {"type": "moon", "pos": (3, 0, 0), "scale": 0.3},
    ]

    planets = []
    for planet_data in planet_types:
        # Create planet
        bpy.ops.mesh.primitive_uv_sphere_add(location=planet_data["pos"])
        planet = bpy.context.active_object
        planet.name = f"Planet_{planet_data['type']}"
        planet.scale = Vector([planet_data["scale"]] * 3)

        # Add planetary material
        planet_mat = AstronomicalShaders.create_planetary_surface_shader(
            planet_type=planet_data["type"]
        )
        planet.data.materials.append(planet_mat)

        planets.append(planet)

    print("Planetary system created!")
    return planets


if __name__ == "__main__":
    create_stellar_showcase()
    create_planetary_system()
