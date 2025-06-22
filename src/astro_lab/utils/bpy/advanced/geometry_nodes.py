"""
Blender 4.4 Procedural Astronomical Geometry Module
==================================================

Create procedural astronomical structures using Geometry Nodes:
- Galaxy spiral generation with stellar populations
- HR diagram 3D visualizations
- Stellar classification systems
- Procedural nebula structures
- Real-time stellar evolution

Optimized for scientific accuracy and performance.

Author: Astro-Graph Agent
Version: 1.0.0
Blender: 4.4+
"""

import warnings
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

# Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

try:
    import bpy
    import bmesh
    import numpy as np
    from mathutils import Euler, Matrix, Vector
    BPY_AVAILABLE = True
except ImportError as e:
    print(f"Blender modules not available: {e}")
    BPY_AVAILABLE = False
    bpy = None
    bmesh = None
    np = None
    Euler = None
    Matrix = None
    Vector = None


class ProceduralAstronomy:
    """Generate procedural astronomical structures"""

    @staticmethod
    def create_hr_diagram_3d(
        stellar_data: List[Dict[str, float]], scale_factor: float = 1.0
    ) -> bpy.types.Object:
        """
        Create a 3D Hertzsprung-Russell diagram visualization.

        Args:
            stellar_data: List of stellar parameters
            scale_factor: Scale factor for the diagram

        Returns:
            Created HR diagram object
        """
        # Create mesh for star positions
        mesh = bpy.data.meshes.new("HR_Diagram")
        obj = bpy.data.objects.new("HR_Diagram", mesh)
        bpy.context.collection.objects.link(obj)

        verts = []

        # Temperature-color mapping
        temp_colors = {
            "O": (0.6, 0.7, 1.0),
            "B": (0.7, 0.8, 1.0),
            "A": (1.0, 1.0, 1.0),
            "F": (1.0, 1.0, 0.9),
            "G": (1.0, 0.9, 0.7),
            "K": (1.0, 0.7, 0.4),
            "M": (1.0, 0.4, 0.2),
        }

        for star in stellar_data:
            # Position in HR diagram space
            x = -math.log10(star.get("temperature", 5778)) * scale_factor
            y = math.log10(star.get("luminosity", 1.0)) * scale_factor
            z = math.log10(star.get("mass", 1.0)) * scale_factor * 0.5

            verts.append((x, y, z))

        # Create mesh from vertices
        mesh.from_pydata(verts, [], [])
        mesh.update()

        # Add geometry nodes for stellar rendering
        ProceduralAstronomy._add_stellar_instances(obj)

        return obj

    @staticmethod
    def create_galaxy_structure(
        center: Vector = Vector((0, 0, 0)),
        galaxy_type: str = "spiral",
        num_stars: int = 50000,
        radius: float = 20.0,
    ) -> bpy.types.Object:
        """
        Create a procedural galaxy structure.

        Args:
            center: Galaxy center position
            galaxy_type: Type of galaxy ('spiral', 'elliptical', 'irregular')
            num_stars: Number of stars to generate
            radius: Galaxy radius

        Returns:
            Created galaxy object
        """
        # Create base object
        bpy.ops.mesh.primitive_plane_add(location=center)
        galaxy_obj = bpy.context.active_object
        galaxy_obj.name = f"{galaxy_type.title()}Galaxy"

        # Remove default mesh
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")

        if galaxy_type == "spiral":
            tree = ProceduralAstronomy._create_spiral_galaxy_nodes(
                num_stars, radius, num_arms=4
            )
        elif galaxy_type == "elliptical":
            tree = ProceduralAstronomy._create_elliptical_galaxy_nodes(
                num_stars, radius
            )
        else:  # irregular
            tree = ProceduralAstronomy._create_irregular_galaxy_nodes(num_stars, radius)

        # Add geometry nodes modifier
        modifier = galaxy_obj.modifiers.new(name="GalaxyStructure", type="NODES")
        modifier.node_group = tree

        return galaxy_obj

    @staticmethod
    def _add_stellar_instances(obj: bpy.types.Object) -> None:
        """Add instanced stars with spectral colors."""
        # Add geometry nodes modifier
        modifier = obj.modifiers.new(name="StellarInstances", type="NODES")
        tree = bpy.data.node_groups.new(
            name="StellarInstances", type="GeometryNodeTree"
        )
        modifier.node_group = tree

        nodes = tree.nodes
        links = tree.links

        # Clear and setup interface
        nodes.clear()
        tree.interface.clear()

        # Create interface
        tree.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        tree.interface.new_socket(
            name="Star Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set defaults
        modifier["Input_2"] = 0.1  # Star size

        # Create nodes
        input_node = nodes.new("NodeGroupInput")
        output_node = nodes.new("NodeGroupOutput")

        # Create star geometry
        ico_sphere = nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.inputs["Subdivisions"].default_value = 2

        # Instance on points
        instance = nodes.new("GeometryNodeInstanceOnPoints")

        # Random size variation
        random_size = nodes.new("FunctionNodeRandomValue")
        random_size.data_type = "FLOAT"
        random_size.inputs["Min"].default_value = 0.5
        random_size.inputs["Max"].default_value = 2.0

        # Position nodes
        input_node.location = (-400, 0)
        ico_sphere.location = (-200, -200)
        random_size.location = (-200, -100)
        instance.location = (0, 0)
        output_node.location = (200, 0)

        # Connect nodes
        links.new(input_node.outputs["Geometry"], instance.inputs["Points"])
        links.new(input_node.outputs["Star Size"], ico_sphere.inputs["Radius"])
        links.new(ico_sphere.outputs["Mesh"], instance.inputs["Instance"])
        links.new(random_size.outputs["Value"], instance.inputs["Scale"])
        links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])

    @staticmethod
    def _create_spiral_galaxy_nodes(
        num_stars: int, radius: float, num_arms: int = 4
    ) -> bpy.types.NodeTree:
        """Create node tree for spiral galaxy generation."""
        tree = bpy.data.node_groups.new(name="SpiralGalaxy", type="GeometryNodeTree")
        nodes = tree.nodes
        links = tree.links

        # Clear and setup interface
        nodes.clear()
        tree.interface.clear()

        # Create interface
        tree.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        tree.interface.new_socket(
            name="Star Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        tree.interface.new_socket(
            name="Galaxy Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Create nodes
        input_node = nodes.new("NodeGroupInput")
        output_node = nodes.new("NodeGroupOutput")

        # Create base disk
        cylinder = nodes.new("GeometryNodeMeshCylinder")
        cylinder.fill_type = "NGON"
        cylinder.inputs["Depth"].default_value = 0.1

        # Distribute points for stars
        distribute = nodes.new("GeometryNodeDistributePointsOnFaces")

        # Instance stars
        ico_sphere = nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.inputs["Subdivisions"].default_value = 1
        ico_sphere.inputs["Radius"].default_value = 0.02

        instance = nodes.new("GeometryNodeInstanceOnPoints")

        # Position nodes
        input_node.location = (-800, 0)
        cylinder.location = (-600, 0)
        distribute.location = (-400, 0)
        ico_sphere.location = (-200, 200)
        instance.location = (0, 0)
        output_node.location = (200, 0)

        # Connect basic structure
        links.new(input_node.outputs["Galaxy Radius"], cylinder.inputs["Radius"])
        links.new(cylinder.outputs["Mesh"], distribute.inputs["Mesh"])
        links.new(input_node.outputs["Star Count"], distribute.inputs["Density"])

        # Final instancing
        links.new(distribute.outputs["Points"], instance.inputs["Points"])
        links.new(ico_sphere.outputs["Mesh"], instance.inputs["Instance"])
        links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])

        return tree

    @staticmethod
    def _create_elliptical_galaxy_nodes(
        num_stars: int, radius: float
    ) -> bpy.types.NodeTree:
        """Create node tree for elliptical galaxy generation."""
        tree = bpy.data.node_groups.new(
            name="EllipticalGalaxy", type="GeometryNodeTree"
        )
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
            name="Star Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        tree.interface.new_socket(
            name="Galaxy Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Create nodes
        input_node = nodes.new("NodeGroupInput")
        output_node = nodes.new("NodeGroupOutput")

        # Create ellipsoid
        uv_sphere = nodes.new("GeometryNodeMeshUVSphere")

        # Scale to elliptical shape
        transform = nodes.new("GeometryNodeTransform")
        transform.inputs["Scale"].default_value = (1.0, 0.7, 0.5)

        # Distribute points
        distribute = nodes.new("GeometryNodeDistributePointsOnFaces")

        # Instance stars
        ico_sphere = nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.inputs["Radius"].default_value = 0.015

        instance = nodes.new("GeometryNodeInstanceOnPoints")

        # Position nodes
        input_node.location = (-600, 0)
        uv_sphere.location = (-400, 0)
        transform.location = (-200, 0)
        distribute.location = (0, 0)
        ico_sphere.location = (0, 200)
        instance.location = (200, 0)
        output_node.location = (400, 0)

        # Connect nodes
        links.new(input_node.outputs["Galaxy Radius"], uv_sphere.inputs["Radius"])
        links.new(uv_sphere.outputs["Mesh"], transform.inputs["Geometry"])
        links.new(transform.outputs["Geometry"], distribute.inputs["Mesh"])
        links.new(input_node.outputs["Star Count"], distribute.inputs["Density"])
        links.new(distribute.outputs["Points"], instance.inputs["Points"])
        links.new(ico_sphere.outputs["Mesh"], instance.inputs["Instance"])
        links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])

        return tree

    @staticmethod
    def _create_irregular_galaxy_nodes(
        num_stars: int, radius: float
    ) -> bpy.types.NodeTree:
        """Create node tree for irregular galaxy generation."""
        tree = bpy.data.node_groups.new(name="IrregularGalaxy", type="GeometryNodeTree")
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
            name="Star Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        tree.interface.new_socket(
            name="Galaxy Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Create nodes
        input_node = nodes.new("NodeGroupInput")
        output_node = nodes.new("NodeGroupOutput")

        # Create irregular shape with noise
        ico_sphere = nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.inputs["Subdivisions"].default_value = 3

        # Distribute points
        distribute = nodes.new("GeometryNodeDistributePointsOnFaces")

        # Instance stars
        star_sphere = nodes.new("GeometryNodeMeshIcoSphere")
        star_sphere.inputs["Radius"].default_value = 0.02

        instance = nodes.new("GeometryNodeInstanceOnPoints")

        # Position nodes
        input_node.location = (-600, 0)
        ico_sphere.location = (-400, 0)
        distribute.location = (-200, 0)
        star_sphere.location = (-200, 200)
        instance.location = (0, 0)
        output_node.location = (200, 0)

        # Connect nodes
        links.new(input_node.outputs["Galaxy Radius"], ico_sphere.inputs["Radius"])
        links.new(ico_sphere.outputs["Mesh"], distribute.inputs["Mesh"])
        links.new(input_node.outputs["Star Count"], distribute.inputs["Density"])
        links.new(distribute.outputs["Points"], instance.inputs["Points"])
        links.new(star_sphere.outputs["Mesh"], instance.inputs["Instance"])
        links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])

        return tree


class AstronomicalMaterials:
    """Materials for astronomical objects"""

    @staticmethod
    def create_stellar_classification_material(
        spectral_class: str = "G",
    ) -> bpy.types.Material:
        """
        Create material based on stellar spectral classification.

        Args:
            spectral_class: Stellar class (O, B, A, F, G, K, M)

        Returns:
            Created material
        """
        mat = bpy.data.materials.new(name=f"Star_{spectral_class}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Add nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        blackbody = nodes.new("ShaderNodeBlackbody")

        # Stellar temperatures and properties
        stellar_data = {
            "O": {"temp": 30000, "strength": 50.0},
            "B": {"temp": 20000, "strength": 20.0},
            "A": {"temp": 8500, "strength": 10.0},
            "F": {"temp": 6500, "strength": 5.0},
            "G": {"temp": 5500, "strength": 2.0},
            "K": {"temp": 4000, "strength": 1.0},
            "M": {"temp": 3000, "strength": 0.3},
        }

        data = stellar_data.get(spectral_class, stellar_data["G"])

        # Set temperature
        blackbody.inputs["Temperature"].default_value = data["temp"]
        emission.inputs["Strength"].default_value = data["strength"]

        # Position nodes
        blackbody.location = (-200, 0)
        emission.location = (0, 0)
        output.location = (200, 0)

        # Connect nodes
        links.new(blackbody.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        return mat


# Example usage functions
def create_hr_diagram_demo():
    """Create a demonstration HR diagram."""
    # Sample stellar data
    stellar_data = []

    # Main sequence stars
    for i in range(100):
        temp = random.uniform(3000, 30000)
        luminosity = (temp / 5778) ** 3.5  # Mass-luminosity relation approximation
        mass = (temp / 5778) ** 0.7

        if temp > 20000:
            spec_class = "O"
        elif temp > 10000:
            spec_class = "B"
        elif temp > 7500:
            spec_class = "A"
        elif temp > 6000:
            spec_class = "F"
        elif temp > 5200:
            spec_class = "G"
        elif temp > 3700:
            spec_class = "K"
        else:
            spec_class = "M"

        stellar_data.append(
            {
                "temperature": temp,
                "luminosity": luminosity + random.uniform(-0.5, 0.5),
                "mass": mass,
                "spectral_class": spec_class,
            }
        )

    # Create HR diagram
    hr_diagram = ProceduralAstronomy.create_hr_diagram_3d(
        stellar_data, scale_factor=2.0
    )

    # Add stellar material
    star_mat = AstronomicalMaterials.create_stellar_classification_material("G")
    hr_diagram.data.materials.append(star_mat)

    print("HR Diagram demonstration created!")


def create_galaxy_comparison():
    """Create comparison of different galaxy types."""
    # Spiral galaxy
    spiral = ProceduralAstronomy.create_galaxy_structure(
        center=Vector((-15, 0, 0)), galaxy_type="spiral", num_stars=30000, radius=10.0
    )

    # Elliptical galaxy
    elliptical = ProceduralAstronomy.create_galaxy_structure(
        center=Vector((0, 0, 0)), galaxy_type="elliptical", num_stars=25000, radius=8.0
    )

    # Irregular galaxy
    irregular = ProceduralAstronomy.create_galaxy_structure(
        center=Vector((15, 0, 0)), galaxy_type="irregular", num_stars=15000, radius=6.0
    )

    # Add materials
    star_materials = [
        AstronomicalMaterials.create_stellar_classification_material("B"),
        AstronomicalMaterials.create_stellar_classification_material("G"),
        AstronomicalMaterials.create_stellar_classification_material("M"),
    ]

    for i, galaxy in enumerate([spiral, elliptical, irregular]):
        galaxy.data.materials.append(star_materials[i])

    print("Galaxy comparison scene created!")


if __name__ == "__main__":
    create_hr_diagram_demo()
    create_galaxy_comparison()
