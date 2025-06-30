"""
AstroLab Blender Materials API
=============================

High-level API for creating astronomical materials in Blender.
Uses advanced/futuristic_materials.py for DRY, maintainable code.
"""

from .advanced.futuristic_materials import FuturisticMaterials


def create_material(material_type: str = "iridescent", **kwargs):
    """
    Create a material by type (iridescent, glass, metallic, holographic, etc.).
    Args:
        material_type: Type of material
        **kwargs: Material parameters
    Returns:
        Blender material
    """
    if material_type == "iridescent":
        return FuturisticMaterials.create_iridescent_material(**kwargs)
    elif material_type == "glass":
        return FuturisticMaterials.create_glass_material(**kwargs)
    elif material_type == "metallic":
        return FuturisticMaterials.create_metallic_material(**kwargs)
    elif material_type == "holographic":
        return FuturisticMaterials.create_holographic_material(**kwargs)
    elif material_type == "energy_field":
        return FuturisticMaterials.create_energy_field_material(**kwargs)
    elif material_type == "force_field":
        return FuturisticMaterials.create_force_field_material(**kwargs)
    else:
        raise ValueError(f"Unknown material type: {material_type}")


__all__ = ["create_material"]
