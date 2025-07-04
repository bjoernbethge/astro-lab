"""
Planetary System Operator for AlbPy
==================================

Creates a system with different planetary types using Node Groups.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator


class ALBPY_OT_CreatePlanetarySystem(Operator):
    bl_idname = "albpy.create_planetary_system"
    bl_label = "Create Planetary System"
    bl_description = "Create system with different planetary types using Node Groups"
    bl_options = {"REGISTER", "UNDO"}

    # Layout settings
    spacing: FloatProperty(
        name="Spacing",
        description="Spacing between planets",
        default=3.0,
        min=1.0,
        max=15.0,
    )

    scale: FloatProperty(
        name="Scale",
        description="Scale of planets",
        default=1.0,
        min=0.1,
        max=5.0,
    )

    # Planet types to include
    include_terrestrial: BoolProperty(
        name="Terrestrial Planets",
        description="Include terrestrial planets (rocky)",
        default=True,
    )

    include_gas_giant: BoolProperty(
        name="Gas Giant",
        description="Include gas giant",
        default=True,
    )

    include_ice_giant: BoolProperty(
        name="Ice Giant",
        description="Include ice giant",
        default=True,
    )

    include_moon: BoolProperty(
        name="Moon",
        description="Include moon",
        default=True,
    )

    # System layout
    layout_type: EnumProperty(
        name="Layout",
        description="Layout of the planetary system",
        items=[
            ("horizontal", "Horizontal", "Horizontal arrangement"),
            ("vertical", "Vertical", "Vertical arrangement"),
            ("circular", "Circular", "Circular arrangement"),
        ],
        default="horizontal",
    )

    def execute(self, context):
        try:
            # Create planetary system
            planets = self._create_planetary_system()

            self.report(
                {"INFO"}, f"Created planetary system with {len(planets)} objects!"
            )
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to create planetary system: {str(e)}")
            return {"CANCELLED"}

    def _create_planetary_system(self):
        """Create system with different planetary types using Node Groups."""
        planet_types = []
        offset = 0

        # Define planet types based on user selection
        if self.include_terrestrial:
            planet_types.append(
                {
                    "type": "terrestrial",
                    "pos": self._get_position(offset),
                    "scale": 0.5 * self.scale,
                }
            )
            offset += self.spacing

        if self.include_gas_giant:
            planet_types.append(
                {
                    "type": "gas_giant",
                    "pos": self._get_position(offset),
                    "scale": 2.0 * self.scale,
                }
            )
            offset += self.spacing

        if self.include_ice_giant:
            planet_types.append(
                {
                    "type": "ice_giant",
                    "pos": self._get_position(offset),
                    "scale": 1.5 * self.scale,
                }
            )
            offset += self.spacing

        if self.include_moon:
            planet_types.append(
                {
                    "type": "moon",
                    "pos": self._get_position(offset),
                    "scale": 0.3 * self.scale,
                }
            )
            offset += self.spacing

        if not planet_types:
            self.report({"WARNING"}, "No planet types selected!")
            return []

        planets = []
        for planet_data in planet_types:
            obj = bpy.data.objects.new(f"Planet_{planet_data['type']}", None)
            bpy.context.scene.collection.objects.link(obj)
            mod = obj.modifiers.new("PlanetNodes", "NODES")
            if "ALBPY_NG_Planet" not in bpy.data.node_groups:
                self.report({"WARNING"}, "Planet Node Group not found!")
                continue
            mod.node_group = bpy.data.node_groups["ALBPY_NG_Planet"]
            try:
                mod["Input_2"] = planet_data["type"]
                mod["Input_3"] = planet_data["scale"]
            except Exception:
                pass
            obj.location = planet_data["pos"]
            obj.scale = (planet_data["scale"],) * 3
            planets.append(obj)
        return planets

    def _get_position(self, offset):
        """Get position based on layout type."""
        if self.layout_type == "horizontal":
            return (offset, 0, 0)
        elif self.layout_type == "vertical":
            return (0, offset, 0)
        else:  # circular
            import math

            angle = offset * 0.5
            radius = 5.0
            return (radius * math.cos(angle), radius * math.sin(angle), 0)


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreatePlanetarySystem)

    bpy.utils.register_class(ALBPY_OT_CreatePlanetarySystem)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreatePlanetarySystem)
