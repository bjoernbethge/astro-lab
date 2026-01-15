"""
Nebula Showcase Operator for AlbPy
=================================

Creates a showcase of different nebula types using Node Groups.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator


class ALBPY_OT_CreateNebulaShowcase(Operator):
    bl_idname = "albpy.create_nebula_showcase"
    bl_label = "Create Nebula Showcase"
    bl_description = "Create showcase of different nebula types using Node Groups"
    bl_options = {"REGISTER", "UNDO"}

    # Layout settings
    spacing: FloatProperty(
        name="Spacing",
        description="Spacing between nebulae",
        default=6.0,
        min=2.0,
        max=20.0,
    )

    scale: FloatProperty(
        name="Scale",
        description="Scale of nebulae",
        default=1.0,
        min=0.1,
        max=5.0,
    )

    # Nebula types to include
    include_h_alpha: BoolProperty(
        name="H-Alpha Nebula",
        description="Include H-Alpha emission nebula",
        default=True,
    )

    include_o_iii: BoolProperty(
        name="O-III Nebula",
        description="Include O-III emission nebula",
        default=True,
    )

    include_mixed: BoolProperty(
        name="Mixed Nebula",
        description="Include mixed emission nebula",
        default=True,
    )

    # System layout
    layout_type: EnumProperty(
        name="Layout",
        description="Layout of the nebula showcase",
        items=[
            ("horizontal", "Horizontal", "Horizontal arrangement"),
            ("vertical", "Vertical", "Vertical arrangement"),
            ("circular", "Circular", "Circular arrangement"),
        ],
        default="horizontal",
    )

    def execute(self, context):
        try:
            # Create nebula showcase
            nebulae = self._create_nebula_showcase()

            self.report(
                {"INFO"}, f"Created nebula showcase with {len(nebulae)} nebulae!"
            )
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to create nebula showcase: {str(e)}")
            return {"CANCELLED"}

    def _create_nebula_showcase(self):
        """Create showcase of different nebula types using Node Groups."""
        nebula_types = []
        offset = 0

        # Define nebula types based on user selection
        if self.include_h_alpha:
            nebula_types.append(
                {
                    "name": "H_Alpha",
                    "primary": (0.8, 0.2, 0.2, 1.0),
                    "pos": self._get_position(offset),
                }
            )
            offset += self.spacing

        if self.include_o_iii:
            nebula_types.append(
                {
                    "name": "O_III",
                    "primary": (0.2, 0.8, 0.3, 1.0),
                    "pos": self._get_position(offset),
                }
            )
            offset += self.spacing

        if self.include_mixed:
            nebula_types.append(
                {
                    "name": "Mixed",
                    "primary": (0.4, 0.6, 1.0, 1.0),
                    "pos": self._get_position(offset),
                }
            )
            offset += self.spacing

        if not nebula_types:
            self.report({"WARNING"}, "No nebula types selected!")
            return []

        nebulae = []
        for nebula_data in nebula_types:
            obj = bpy.data.objects.new(f"Nebula_{nebula_data['name']}", None)
            bpy.context.scene.collection.objects.link(obj)
            mod = obj.modifiers.new("NebulaNodes", "NODES")
            if "ALBPY_NG_Nebula" not in bpy.data.node_groups:
                self.report({"WARNING"}, "Nebula Node Group not found!")
                continue
            mod.node_group = bpy.data.node_groups["ALBPY_NG_Nebula"]
            try:
                mod["Input_2"] = nebula_data["primary"]
                mod["Input_3"] = (0.2, 0.8, 0.3, 1.0)
                mod["Input_4"] = 2.0
            except Exception:
                pass
            obj.location = nebula_data["pos"]
            obj.scale = (2.0 * self.scale,) * 3
            nebulae.append(obj)
        return nebulae

    def _get_position(self, offset):
        """Get position based on layout type."""
        if self.layout_type == "horizontal":
            return (offset, 0, 0)
        elif self.layout_type == "vertical":
            return (0, offset, 0)
        else:  # circular
            import math

            angle = offset * 0.5
            radius = 8.0
            return (radius * math.cos(angle), radius * math.sin(angle), 0)


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreateNebulaShowcase)

    bpy.utils.register_class(ALBPY_OT_CreateNebulaShowcase)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreateNebulaShowcase)
