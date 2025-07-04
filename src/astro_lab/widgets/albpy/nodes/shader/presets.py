"""
Material Presets for AlbPy Shader Node Groups
============================================

Provides preset configurations for all shader node groups.
Centralized location for easy access and maintenance.
"""

from typing import Any, Dict

import bpy


class ShaderPresets:
    """Centralized presets for all shader node groups"""

    # Futuristic Material Presets
    @staticmethod
    def luxury_teal_iridescent() -> Dict[str, Any]:
        """Luxury teal iridescent material preset."""
        return {
            "node_group": "ALBPY_NG_Iridescent",
            "parameters": {
                "Base Color": (0.0, 0.8, 0.6, 1.0),
                "Iridescence Strength": 0.8,
                "Iridescence Shift": 0.2,
                "Metallic": 0.8,
                "Roughness": 0.1,
            },
        }

    @staticmethod
    def golden_metallic() -> Dict[str, Any]:
        """Golden metallic material preset."""
        return {
            "node_group": "ALBPY_NG_Metallic",
            "parameters": {
                "Color": (1.0, 0.8, 0.2, 1.0),
                "Metallic": 1.0,
                "Roughness": 0.1,
                "Anisotropic": 0.3,
                "Anisotropic Rotation": 0.0,
            },
        }

    @staticmethod
    def crystal_glass() -> Dict[str, Any]:
        """Crystal glass material preset."""
        return {
            "node_group": "ALBPY_NG_Glass",
            "parameters": {
                "Color": (0.95, 0.98, 1.0, 1.0),
                "Transmission": 0.98,
                "IOR": 1.5,
                "Roughness": 0.0,
                "Noise Scale": 10.0,
            },
        }

    @staticmethod
    def holographic_blue() -> Dict[str, Any]:
        """Holographic blue material preset."""
        return {
            "node_group": "ALBPY_NG_Holographic",
            "parameters": {
                "Base Color": (0.2, 0.6, 1.0, 1.0),
                "Hologram Strength": 1.2,
                "Scan Speed": 1.5,
                "Transparency": 0.7,
            },
        }

    @staticmethod
    def energy_purple() -> Dict[str, Any]:
        """Energy purple material preset."""
        return {
            "node_group": "ALBPY_NG_EnergyField",
            "parameters": {
                "Color": (0.6, 0.2, 1.0, 1.0),
                "Energy Strength": 2.5,
                "Pulse Speed": 1.8,
                "Noise Scale": 5.0,
            },
        }

    @staticmethod
    def force_field_purple() -> Dict[str, Any]:
        """Force field purple material preset."""
        return {
            "node_group": "ALBPY_NG_ForceField",
            "parameters": {
                "Color": (0.8, 0.2, 1.0, 1.0),
                "Field Strength": 1.5,
                "Ripple Speed": 2.0,
                "Transparency": 0.6,
            },
        }

    # Astronomical Material Presets
    @staticmethod
    def solar_star() -> Dict[str, Any]:
        """Solar-type star material preset."""
        return {
            "node_group": "ALBPY_NG_Star",
            "parameters": {
                "Temperature": 5778.0,
                "Luminosity": 1.0,
                "Stellar Class": "G",
                "Emission Strength": 2.0,
            },
        }

    @staticmethod
    def blue_giant() -> Dict[str, Any]:
        """Blue giant star material preset."""
        return {
            "node_group": "ALBPY_NG_Star",
            "parameters": {
                "Temperature": 20000.0,
                "Luminosity": 100.0,
                "Stellar Class": "B",
                "Emission Strength": 20.0,
            },
        }

    @staticmethod
    def red_dwarf() -> Dict[str, Any]:
        """Red dwarf star material preset."""
        return {
            "node_group": "ALBPY_NG_Star",
            "parameters": {
                "Temperature": 3000.0,
                "Luminosity": 0.01,
                "Stellar Class": "M",
                "Emission Strength": 0.3,
            },
        }

    @staticmethod
    def o_type_star() -> Dict[str, Any]:
        """O-type star material preset (Blue supergiants)."""
        return {
            "node_group": "ALBPY_NG_Star",
            "parameters": {
                "Temperature": 30000.0,
                "Luminosity": 100000.0,
                "Stellar Class": "O",
                "Emission Strength": 50.0,
            },
        }

    @staticmethod
    def a_type_star() -> Dict[str, Any]:
        """A-type star material preset (White stars)."""
        return {
            "node_group": "ALBPY_NG_Star",
            "parameters": {
                "Temperature": 8500.0,
                "Luminosity": 20.0,
                "Stellar Class": "A",
                "Emission Strength": 10.0,
            },
        }

    @staticmethod
    def f_type_star() -> Dict[str, Any]:
        """F-type star material preset (Yellow-white stars)."""
        return {
            "node_group": "ALBPY_NG_Star",
            "parameters": {
                "Temperature": 6500.0,
                "Luminosity": 5.0,
                "Stellar Class": "F",
                "Emission Strength": 5.0,
            },
        }

    @staticmethod
    def k_type_star() -> Dict[str, Any]:
        """K-type star material preset (Orange stars)."""
        return {
            "node_group": "ALBPY_NG_Star",
            "parameters": {
                "Temperature": 4000.0,
                "Luminosity": 0.5,
                "Stellar Class": "K",
                "Emission Strength": 1.0,
            },
        }

    @staticmethod
    def terrestrial_planet() -> Dict[str, Any]:
        """Terrestrial planet material preset."""
        return {
            "node_group": "ALBPY_NG_Planet",
            "parameters": {
                "Color": (0.4, 0.3, 0.2, 1.0),
                "Roughness": 0.8,
                "Metallic": 0.1,
            },
        }

    @staticmethod
    def gas_giant() -> Dict[str, Any]:
        """Gas giant planet material preset."""
        return {
            "node_group": "ALBPY_NG_Planet",
            "parameters": {
                "Color": (0.9, 0.8, 0.7, 1.0),
                "Roughness": 0.3,
                "Metallic": 0.0,
            },
        }

    @staticmethod
    def h_alpha_nebula() -> Dict[str, Any]:
        """H-alpha emission nebula material preset."""
        return {
            "node_group": "ALBPY_NG_EmissionSimple",
            "parameters": {"Color": (0.8, 0.2, 0.2, 1.0), "Strength": 3.0},
        }

    @staticmethod
    def o_iii_nebula() -> Dict[str, Any]:
        """O-III emission nebula material preset."""
        return {
            "node_group": "ALBPY_NG_EmissionSimple",
            "parameters": {"Color": (0.2, 0.8, 0.3, 1.0), "Strength": 2.5},
        }

    @staticmethod
    def earth_atmosphere() -> Dict[str, Any]:
        """Earth-like atmosphere material preset."""
        return {
            "node_group": "ALBPY_NG_Atmosphere",
            "parameters": {
                "Atmosphere Color": (0.3, 0.6, 1.0, 1.0),
                "Density": 0.1,
                "Scale Height": 8.5,
                "Planet Radius": 1.0,
            },
        }

    @staticmethod
    def mars_atmosphere() -> Dict[str, Any]:
        """Mars-like atmosphere material preset."""
        return {
            "node_group": "ALBPY_NG_Atmosphere",
            "parameters": {
                "Atmosphere Color": (0.8, 0.5, 0.3, 1.0),
                "Density": 0.01,
                "Scale Height": 11.0,
                "Planet Radius": 0.53,
            },
        }

    @staticmethod
    def h_alpha_nebula_advanced() -> Dict[str, Any]:
        """Advanced H-alpha nebula material preset."""
        return {
            "node_group": "ALBPY_NG_Nebula",
            "parameters": {
                "Primary Color": (0.8, 0.2, 0.2, 1.0),
                "Secondary Color": (0.4, 0.6, 1.0, 1.0),
                "Emission Strength": 2.0,
                "Density": 0.5,
                "Density Variation": 0.5,
                "Noise Scale 1": 1.0,
                "Noise Scale 2": 5.0,
            },
        }

    @staticmethod
    def gas_giant_advanced() -> Dict[str, Any]:
        """Advanced gas giant material preset."""
        return {
            "node_group": "ALBPY_NG_Planet",
            "parameters": {
                "Base Color": (0.9, 0.8, 0.7, 1.0),
                "Roughness": 0.3,
                "Metallic": 0.0,
                "Specular": 0.8,
                "Transmission": 0.0,
                "Planet Type": "gas_giant",
                "Surface Detail": 5.0,
                "Band Scale": 2.0,
            },
        }

    # Galaxy Configuration Presets
    @staticmethod
    def spiral_galaxy_config() -> Dict[str, Any]:
        """Spiral galaxy configuration preset."""
        return {
            "node_group": "ALBPY_NG_SpiralGalaxy",
            "parameters": {
                "Star Count": 50000,
                "Galaxy Radius": 20.0,
                "Number of Arms": 4,
            },
        }

    @staticmethod
    def elliptical_galaxy_config() -> Dict[str, Any]:
        """Elliptical galaxy configuration preset."""
        return {
            "node_group": "ALBPY_NG_EllipticalGalaxy",
            "parameters": {
                "Star Count": 25000,
                "Galaxy Radius": 15.0,
                "Ellipticity X": 0.7,
                "Ellipticity Y": 0.5,
            },
        }

    @staticmethod
    def irregular_galaxy_config() -> Dict[str, Any]:
        """Irregular galaxy configuration preset."""
        return {
            "node_group": "ALBPY_NG_IrregularGalaxy",
            "parameters": {
                "Star Count": 15000,
                "Galaxy Radius": 10.0,
                "Irregularity": 0.3,
            },
        }

    @staticmethod
    def galactic_dust_config() -> Dict[str, Any]:
        """Galactic dust configuration preset."""
        return {
            "node_group": "ALBPY_NG_GalacticDust",
            "parameters": {
                "Dust Size": 20.0,
                "Dust Density": 0.01,
                "Noise Scale": 10.0,
                "Dust Color": (0.4, 0.3, 0.2, 1.0),
            },
        }

    @staticmethod
    def star_forming_region_config() -> Dict[str, Any]:
        """Star forming region configuration preset."""
        return {
            "node_group": "ALBPY_NG_StarFormingRegion",
            "parameters": {
                "Star Count": 1000,
                "Region Size": 5.0,
                "Density Falloff": 2.0,
                "Star Size": 0.05,
            },
        }

    @staticmethod
    def planetary_nebula_config() -> Dict[str, Any]:
        """Planetary nebula configuration preset."""
        return {
            "node_group": "ALBPY_NG_PlanetaryNebula",
            "parameters": {
                "Nebula Size": 10.0,
                "Shell Thickness": 0.2,
                "Displacement Strength": 0.1,
                "Noise Scale": 2.0,
            },
        }

    @staticmethod
    def supernova_remnant_config() -> Dict[str, Any]:
        """Supernova remnant configuration preset."""
        return {
            "node_group": "ALBPY_NG_SupernovaRemnant",
            "parameters": {
                "Remnant Size": 30.0,
                "Shock Strength": 0.3,
                "Voronoi Scale": 3.0,
                "Filament Count": 20,
            },
        }

    @staticmethod
    def stellar_wind_config() -> Dict[str, Any]:
        """Stellar wind configuration preset."""
        return {
            "node_group": "ALBPY_NG_StellarWind",
            "parameters": {
                "Wind Radius": 5.0,
                "Wind Speed": 500.0,
                "Mass Loss Rate": 1e-6,
                "Density Falloff": 2.0,
            },
        }

    @staticmethod
    def orbital_trail_config() -> Dict[str, Any]:
        """Orbital trail configuration preset."""
        return {
            "node_group": "ALBPY_NG_OrbitalTrail",
            "parameters": {
                "Trail Length": 100,
                "Trail Width": 0.02,
                "Trail Opacity": 0.5,
                "Trail Color": (0.8, 0.9, 1.0, 1.0),
            },
        }

    @staticmethod
    def n_body_system_config() -> Dict[str, Any]:
        """N-body system configuration preset."""
        return {
            "node_group": "ALBPY_NG_NBodySystem",
            "parameters": {
                "Body Count": 10,
                "System Radius": 20.0,
                "Mass Range": 1.0,
                "Body Size Scale": 0.1,
            },
        }

    # HR Diagram Configuration Presets
    @staticmethod
    def hr_diagram_config() -> Dict[str, Any]:
        """HR Diagram configuration preset."""
        return {
            "node_group": "ALBPY_NG_HRDiagram",
            "parameters": {
                "Star Size": 0.1,
                "Scale Factor": 1.0,
            },
        }


def apply_preset_to_material(
    material: bpy.types.Material, preset: Dict[str, Any]
) -> None:
    """
    Apply a preset to an existing material.

    Args:
        material: The material to apply the preset to
        preset: Preset configuration dictionary
    """
    if not material.use_nodes:
        material.use_nodes = True

    # Get the node group
    node_group_name = preset["node_group"]
    node_group = bpy.data.node_groups.get(node_group_name)

    if not node_group:
        print(f"Node group {node_group_name} not found")
        return

    # Clear existing nodes
    material.node_tree.nodes.clear()

    # Add the node group
    group_node = material.node_tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = node_group

    # Add output node
    output_node = material.node_tree.nodes.new("ShaderNodeOutputMaterial")

    # Connect group to output
    material.node_tree.links.new(
        group_node.outputs["Shader"], output_node.inputs["Surface"]
    )

    # Apply preset parameters
    parameters = preset["parameters"]
    for param_name, value in parameters.items():
        if param_name in group_node.inputs:
            group_node.inputs[param_name].default_value = value


def create_material_from_preset(
    preset: Dict[str, Any], name: str = None
) -> bpy.types.Material:
    """
    Create a new material from a preset.

    Args:
        preset: Preset configuration dictionary
        name: Optional material name

    Returns:
        Created material
    """
    if name is None:
        name = f"Material_{preset['node_group']}"

    material = bpy.data.materials.new(name=name)
    apply_preset_to_material(material, preset)

    return material


# Convenience functions for common presets
def create_luxury_teal_material() -> bpy.types.Material:
    """Create luxury teal material."""
    return create_material_from_preset(
        ShaderPresets.luxury_teal_iridescent(), "LuxuryTeal"
    )


def create_golden_metallic_material() -> bpy.types.Material:
    """Create golden metallic material."""
    return create_material_from_preset(
        ShaderPresets.golden_metallic(), "GoldenMetallic"
    )


def create_crystal_glass_material() -> bpy.types.Material:
    """Create crystal glass material."""
    return create_material_from_preset(ShaderPresets.crystal_glass(), "CrystalGlass")


def create_holographic_blue_material() -> bpy.types.Material:
    """Create holographic blue material."""
    return create_material_from_preset(
        ShaderPresets.holographic_blue(), "HolographicBlue"
    )


def create_energy_purple_material() -> bpy.types.Material:
    """Create energy purple material."""
    return create_material_from_preset(ShaderPresets.energy_purple(), "EnergyPurple")


def create_solar_star_material() -> bpy.types.Material:
    """Create solar-type star material."""
    return create_material_from_preset(ShaderPresets.solar_star(), "SolarStar")


def create_blue_giant_material() -> bpy.types.Material:
    """Create blue giant star material."""
    return create_material_from_preset(ShaderPresets.blue_giant(), "BlueGiant")


def create_red_dwarf_material() -> bpy.types.Material:
    """Create red dwarf star material."""
    return create_material_from_preset(ShaderPresets.red_dwarf(), "RedDwarf")
