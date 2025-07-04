"""
Stellar Showcase Operator for AlbPy
==================================

Creates a showcase of different stellar types using Node Groups.
"""

import bpy
from bpy.props import BoolProperty, FloatProperty
from bpy.types import Operator

from astro_lab.widgets.albpy.nodes.shader.star import AlbpyStarShaderGroup


class ALBPY_OT_CreateStellarShowcase(Operator):
    bl_idname = "albpy.create_stellar_showcase"
    bl_label = "Create Stellar Showcase"
    bl_description = "Create showcase of different stellar types using Node Groups"
    bl_options = {"REGISTER", "UNDO"}

    # Layout settings
    spacing: FloatProperty(
        name="Spacing",
        description="Spacing between stars",
        default=4.0,
        min=1.0,
        max=20.0,
    )

    scale: FloatProperty(
        name="Scale",
        description="Scale of stars",
        default=1.0,
        min=0.1,
        max=5.0,
    )

    # Stellar types to include
    include_o_stars: BoolProperty(
        name="O-Type Stars",
        description="Include O-type stars",
        default=True,
    )

    include_b_stars: BoolProperty(
        name="B-Type Stars",
        description="Include B-type stars",
        default=True,
    )

    include_a_stars: BoolProperty(
        name="A-Type Stars",
        description="Include A-type stars",
        default=True,
    )

    include_f_stars: BoolProperty(
        name="F-Type Stars",
        description="Include F-type stars",
        default=True,
    )

    include_g_stars: BoolProperty(
        name="G-Type Stars",
        description="Include G-type stars (like our Sun)",
        default=True,
    )

    include_k_stars: BoolProperty(
        name="K-Type Stars",
        description="Include K-type stars",
        default=True,
    )

    include_m_stars: BoolProperty(
        name="M-Type Stars",
        description="Include M-type stars (red dwarfs)",
        default=True,
    )

    def execute(self, context):
        try:
            # Create stellar showcase
            stars = self._create_stellar_showcase()

            self.report({"INFO"}, f"Created stellar showcase with {len(stars)} stars!")
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to create stellar showcase: {str(e)}")
            return {"CANCELLED"}

    def _create_stellar_showcase(self):
        """Create showcase of different stellar types using Node Groups."""
        # Ensure node group is registered and created
        if "ALBPY_NG_Star" not in bpy.data.node_groups:
            bpy.utils.register_class(AlbpyStarShaderGroup)
            bpy.data.node_groups.new("ALBPY_NG_Star", "ShaderNodeTree")

        stellar_types = []
        x_offset = 0

        # Define stellar types based on user selection
        if self.include_o_stars:
            stellar_types.append({"temp": 30000, "class": "O", "pos": (x_offset, 0, 0)})
            x_offset += self.spacing

        if self.include_b_stars:
            stellar_types.append({"temp": 20000, "class": "B", "pos": (x_offset, 0, 0)})
            x_offset += self.spacing

        if self.include_a_stars:
            stellar_types.append({"temp": 8500, "class": "A", "pos": (x_offset, 0, 0)})
            x_offset += self.spacing

        if self.include_f_stars:
            stellar_types.append({"temp": 6500, "class": "F", "pos": (x_offset, 0, 0)})
            x_offset += self.spacing

        if self.include_g_stars:
            stellar_types.append({"temp": 5500, "class": "G", "pos": (x_offset, 0, 0)})
            x_offset += self.spacing

        if self.include_k_stars:
            stellar_types.append({"temp": 4000, "class": "K", "pos": (x_offset, 0, 0)})
            x_offset += self.spacing

        if self.include_m_stars:
            stellar_types.append({"temp": 3000, "class": "M", "pos": (x_offset, 0, 0)})
            x_offset += self.spacing

        if not stellar_types:
            self.report({"WARNING"}, "No stellar types selected!")
            return []

        stars = []
        for star_data in stellar_types:
            mesh = bpy.data.meshes.new(f"StarMesh_{star_data['class']}")
            obj = bpy.data.objects.new(f"Star_{star_data['class']}", mesh)
            bpy.context.scene.collection.objects.link(obj)
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=star_data["pos"])
            # Geometry-Nodes-Modifier
            mod = obj.modifiers.new("StarNodes", "NODES")
            mod.node_group = bpy.data.node_groups["ALBPY_NG_Star"]
            try:
                mod["Input_2"] = star_data["temp"]
                mod["Input_3"] = 1.0  # Luminosity
                mod["Input_4"] = star_data["class"]
            except Exception:
                pass
            obj.location = star_data["pos"]
            obj.scale = (self.scale, self.scale, self.scale)
            # Shader-Node-Group als Material
            mat = bpy.data.materials.new(name=f"Mat_Star_{star_data['class']}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            star_group = nodes.new("ShaderNodeGroup")
            star_group.node_tree = bpy.data.node_groups["ALBPY_NG_Star"]
            output = nodes.new("ShaderNodeOutputMaterial")
            links.new(star_group.outputs["Shader"], output.inputs["Surface"])
            star_group.inputs["Temperature"].default_value = star_data["temp"]
            star_group.inputs["Luminosity"].default_value = 1.0
            star_group.inputs["Stellar Class"].default_value = star_data["class"]
            obj.data.materials.append(mat)
            stars.append(obj)

        return stars


def register():
    bpy.utils.register_class(ALBPY_OT_CreateStellarShowcase)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreateStellarShowcase)
