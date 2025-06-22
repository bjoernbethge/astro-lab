"""
Futuristic Materials for Astronomical Visualization
==================================================

Advanced material creation for futuristic astronomical scenes.
Includes iridescent, glass, metallic, holographic, and energy field materials.
"""

import os
import warnings
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from .. import numpy_compat  # noqa: F401
import bpy

# Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

try:
    import numpy as np
    from mathutils import Vector
    BPY_AVAILABLE = bpy is not None
except ImportError as e:
    print(f"Blender modules not available: {e}")
    BPY_AVAILABLE = False
    bpy = None
    np = None
    Vector = None

class FuturisticMaterials:
    """Create futuristic materials for astronomical visualization"""

    @staticmethod
    def create_iridescent_material(
        base_color: Tuple[float, float, float] = (0.8, 0.2, 0.8),
        iridescence_strength: float = 1.0,
        iridescence_shift: float = 0.0,
    ) -> bpy.types.Material:
        """
        Create iridescent material with chromatic shift.
        
        Args:
            base_color: Base material color
            iridescence_strength: Strength of iridescent effect
            iridescence_shift: Color shift amount
            
        Returns:
            Created iridescent material
        """
        mat = bpy.data.materials.new(name="IridescentMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        
        # Set base properties
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.8
        bsdf.inputs["Roughness"].default_value = 0.1
        bsdf.inputs["Specular"].default_value = 1.0
        
        # Add iridescence
        bsdf.inputs["Iridescence"].default_value = iridescence_strength
        bsdf.inputs["Iridescence IOR"].default_value = 1.3 + iridescence_shift
        
        # Add fresnel for enhanced effect
        fresnel = nodes.new("ShaderNodeFresnel")
        fresnel.inputs["IOR"].default_value = 1.5
        
        # Mix fresnel with base color
        mix_rgb = nodes.new("ShaderNodeMixRGB")
        mix_rgb.blend_type = "ADD"
        mix_rgb.inputs["Fac"].default_value = 0.3
        
        # Position nodes
        fresnel.location = (-400, 200)
        mix_rgb.location = (-200, 0)
        bsdf.location = (0, 0)
        output.location = (200, 0)
        
        # Connect nodes
        links.new(fresnel.outputs["Fac"], mix_rgb.inputs["Fac"])
        links.new(mix_rgb.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        
        return mat

    @staticmethod
    def create_glass_material(
        color: Tuple[float, float, float] = (0.9, 0.95, 1.0),
        transmission: float = 0.95,
        ior: float = 1.45,
        roughness: float = 0.0,
    ) -> bpy.types.Material:
        """
        Create realistic glass material.
        
        Args:
            color: Glass color
            transmission: Transmission strength
            ior: Index of refraction
            roughness: Surface roughness
            
        Returns:
            Created glass material
        """
        mat = bpy.data.materials.new(name="GlassMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        
        # Set glass properties
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Transmission"].default_value = transmission
        bsdf.inputs["IOR"].default_value = ior
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Specular"].default_value = 1.0
        
        # Add subtle color variation
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 10.0
        noise.inputs["Detail"].default_value = 2.0
        
        # Mix with base color
        mix_rgb = nodes.new("ShaderNodeMixRGB")
        mix_rgb.blend_type = "MULTIPLY"
        mix_rgb.inputs["Fac"].default_value = 0.1
        
        # Position nodes
        noise.location = (-400, 0)
        mix_rgb.location = (-200, 0)
        bsdf.location = (0, 0)
        output.location = (200, 0)
        
        # Connect nodes
        links.new(noise.outputs["Color"], mix_rgb.inputs["Color2"])
        links.new(mix_rgb.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        
        return mat

    @staticmethod
    def create_metallic_material(
        color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        metallic: float = 1.0,
        roughness: float = 0.2,
        anisotropy: float = 0.0,
    ) -> bpy.types.Material:
        """
        Create metallic material with optional anisotropy.
        
        Args:
            color: Metal color
            metallic: Metallic strength
            roughness: Surface roughness
            anisotropy: Anisotropic reflection strength
            
        Returns:
            Created metallic material
        """
        mat = bpy.data.materials.new(name="MetallicMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        
        # Set metallic properties
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Specular"].default_value = 0.5
        
        # Add anisotropy if specified
        if anisotropy > 0.0:
            bsdf.inputs["Anisotropic"].default_value = anisotropy
            bsdf.inputs["Anisotropic Rotation"].default_value = 0.0
            
            # Add anisotropic texture for variation
            tex_coord = nodes.new("ShaderNodeTexCoord")
            mapping = nodes.new("ShaderNodeMapping")
            
            # Position nodes
            tex_coord.location = (-600, 0)
            mapping.location = (-400, 0)
            bsdf.location = (0, 0)
            output.location = (200, 0)
            
            # Connect nodes
            links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
            links.new(mapping.outputs["Vector"], bsdf.inputs["Tangent"])
        else:
            # Position nodes
            bsdf.location = (0, 0)
            output.location = (200, 0)
        
        # Connect to output
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        
        return mat

    @staticmethod
    def create_holographic_material(
        base_color: Tuple[float, float, float] = (0.2, 0.8, 1.0),
        hologram_strength: float = 1.0,
        scan_speed: float = 1.0,
    ) -> bpy.types.Material:
        """
        Create holographic material with scanning effect.
        
        Args:
            base_color: Base hologram color
            hologram_strength: Strength of holographic effect
            scan_speed: Speed of scanning lines
            
        Returns:
            Created holographic material
        """
        mat = bpy.data.materials.new(name="HolographicMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        mix_shader = nodes.new("ShaderNodeMixShader")
        
        # Create scanning line effect
        tex_coord = nodes.new("ShaderNodeTexCoord")
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "SAW"
        wave.inputs["Scale"].default_value = 50.0 * scan_speed
        wave.inputs["Distortion"].default_value = 0.5
        
        # Color ramp for scan lines
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.4
        ramp.color_ramp.elements[1].position = 0.6
        
        # Add noise for holographic interference
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 100.0
        noise.inputs["Detail"].default_value = 10.0
        
        # Mix noise with scan lines
        mix_noise = nodes.new("ShaderNodeMixRGB")
        mix_noise.blend_type = "MULTIPLY"
        mix_noise.inputs["Fac"].default_value = 0.3
        
        # Position nodes
        tex_coord.location = (-800, 0)
        wave.location = (-600, 0)
        ramp.location = (-400, 0)
        noise.location = (-600, -200)
        mix_noise.location = (-200, 0)
        emission.location = (0, 100)
        transparent.location = (0, -100)
        mix_shader.location = (200, 0)
        output.location = (400, 0)
        
        # Connect nodes
        links.new(tex_coord.outputs["Generated"], wave.inputs["Vector"])
        links.new(wave.outputs["Color"], ramp.inputs["Fac"])
        links.new(noise.outputs["Color"], mix_noise.inputs["Color2"])
        links.new(ramp.outputs["Color"], mix_noise.inputs["Color1"])
        links.new(mix_noise.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], mix_shader.inputs[1])
        links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
        
        # Set emission strength
        emission.inputs["Strength"].default_value = hologram_strength * 2.0
        
        # Set mix factor
        mix_shader.inputs["Fac"].default_value = 0.7
        
        return mat

    @staticmethod
    def create_energy_field_material(
        color: Tuple[float, float, float] = (0.2, 0.8, 1.0),
        energy_strength: float = 2.0,
        pulse_speed: float = 1.0,
    ) -> bpy.types.Material:
        """
        Create energy field material with pulsing effect.
        
        Args:
            color: Energy field color
            energy_strength: Strength of energy emission
            pulse_speed: Speed of pulsing effect
            
        Returns:
            Created energy field material
        """
        mat = bpy.data.materials.new(name="EnergyFieldMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        
        # Create pulsing effect
        sine = nodes.new("ShaderNodeMath")
        sine.operation = "SINE"
        sine.inputs[1].default_value = pulse_speed
        
        # Add time input for animation
        time = nodes.new("ShaderNodeValue")
        time.outputs[0].default_value = 0.0  # Will be animated
        
        # Scale and offset sine wave
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = 0.5
        
        offset = nodes.new("ShaderNodeMath")
        offset.operation = "ADD"
        offset.inputs[1].default_value = 0.5
        
        # Add noise for energy variation
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 5.0
        noise.inputs["Detail"].default_value = 8.0
        
        # Mix noise with pulse
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 0.3
        
        # Position nodes
        time.location = (-800, 0)
        sine.location = (-600, 0)
        scale.location = (-400, 0)
        offset.location = (-200, 0)
        noise.location = (-600, -200)
        mix.location = (0, 0)
        emission.location = (200, 0)
        output.location = (400, 0)
        
        # Connect nodes
        links.new(time.outputs[0], sine.inputs[0])
        links.new(sine.outputs[0], scale.inputs[0])
        links.new(scale.outputs[0], offset.inputs[0])
        links.new(noise.outputs["Color"], mix.inputs["Color2"])
        links.new(offset.outputs[0], mix.inputs["Color1"])
        links.new(mix.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        
        # Set emission strength
        emission.inputs["Strength"].default_value = energy_strength
        
        return mat

    @staticmethod
    def create_force_field_material(
        color: Tuple[float, float, float] = (0.8, 0.2, 1.0),
        field_strength: float = 1.5,
        ripple_speed: float = 2.0,
    ) -> bpy.types.Material:
        """
        Create force field material with ripple effect.
        
        Args:
            color: Force field color
            field_strength: Strength of field emission
            ripple_speed: Speed of ripple waves
            
        Returns:
            Created force field material
        """
        mat = bpy.data.materials.new(name="ForceFieldMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        mix_shader = nodes.new("ShaderNodeMixShader")
        
        # Create ripple effect
        tex_coord = nodes.new("ShaderNodeTexCoord")
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "RINGS"
        wave.inputs["Scale"].default_value = 20.0 * ripple_speed
        wave.inputs["Distortion"].default_value = 0.3
        
        # Add time for animation
        time = nodes.new("ShaderNodeValue")
        time.outputs[0].default_value = 0.0
        
        # Mix time with coordinates
        mix_time = nodes.new("ShaderNodeMixRGB")
        mix_time.blend_type = "ADD"
        mix_time.inputs["Fac"].default_value = 0.1
        
        # Color ramp for ripple visibility
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.3
        ramp.color_ramp.elements[1].position = 0.7
        
        # Add noise for field variation
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 10.0
        noise.inputs["Detail"].default_value = 6.0
        
        # Mix noise with ripples
        mix_noise = nodes.new("ShaderNodeMixRGB")
        mix_noise.blend_type = "MULTIPLY"
        mix_noise.inputs["Fac"].default_value = 0.4
        
        # Position nodes
        tex_coord.location = (-800, 0)
        time.location = (-800, -200)
        mix_time.location = (-600, 0)
        wave.location = (-400, 0)
        ramp.location = (-200, 0)
        noise.location = (-600, -200)
        mix_noise.location = (0, 0)
        emission.location = (200, 100)
        transparent.location = (200, -100)
        mix_shader.location = (400, 0)
        output.location = (600, 0)
        
        # Connect nodes
        links.new(tex_coord.outputs["Generated"], mix_time.inputs["Color1"])
        links.new(time.outputs[0], mix_time.inputs["Color2"])
        links.new(mix_time.outputs["Color"], wave.inputs["Vector"])
        links.new(wave.outputs["Color"], ramp.inputs["Fac"])
        links.new(noise.outputs["Color"], mix_noise.inputs["Color2"])
        links.new(ramp.outputs["Color"], mix_noise.inputs["Color1"])
        links.new(mix_noise.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], mix_shader.inputs[1])
        links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
        
        # Set emission strength
        emission.inputs["Strength"].default_value = field_strength
        
        # Set mix factor
        mix_shader.inputs["Fac"].default_value = 0.6
        
        return mat

class MaterialPresets:
    """Preset material configurations"""

    @staticmethod
    def luxury_teal_material() -> bpy.types.Material:
        """Create luxury teal material preset."""
        return FuturisticMaterials.create_iridescent_material(
            base_color=(0.0, 0.8, 0.6),
            iridescence_strength=0.8,
            iridescence_shift=0.2
        )

    @staticmethod
    def golden_metallic_material() -> bpy.types.Material:
        """Create golden metallic material preset."""
        return FuturisticMaterials.create_metallic_material(
            color=(1.0, 0.8, 0.2),
            metallic=1.0,
            roughness=0.1,
            anisotropy=0.3
        )

    @staticmethod
    def crystal_glass_material() -> bpy.types.Material:
        """Create crystal glass material preset."""
        return FuturisticMaterials.create_glass_material(
            color=(0.95, 0.98, 1.0),
            transmission=0.98,
            ior=1.5,
            roughness=0.0
        )

    @staticmethod
    def holographic_blue_material() -> bpy.types.Material:
        """Create holographic blue material preset."""
        return FuturisticMaterials.create_holographic_material(
            base_color=(0.2, 0.6, 1.0),
            hologram_strength=1.2,
            scan_speed=1.5
        )

    @staticmethod
    def energy_purple_material() -> bpy.types.Material:
        """Create energy purple material preset."""
        return FuturisticMaterials.create_energy_field_material(
            color=(0.6, 0.2, 1.0),
            energy_strength=2.5,
            pulse_speed=1.8
        ) 