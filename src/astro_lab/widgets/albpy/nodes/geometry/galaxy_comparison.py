import bpy


class AlbpyGalaxyComparisonGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_GalaxyComparison"
    bl_label = "Albpy Galaxy Comparison Geometry Group"

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == "GeometryNodeTree"

    def init(self, context):
        # Clear existing nodes
        self.nodes.clear()
        self.interface.clear()

        # Create interface
        self.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        self.interface.new_socket(
            name="Spacing", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Spiral Stars", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="Elliptical Stars", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="Irregular Stars", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 15.0  # Spacing
        self.interface.items_tree["Input_3"].default_value = 30000  # Spiral Stars
        self.interface.items_tree["Input_4"].default_value = 25000  # Elliptical Stars
        self.interface.items_tree["Input_5"].default_value = 15000  # Irregular Stars

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create separate geometry for each galaxy type
        spiral_geo = self.nodes.new("GeometryNodeGroup")
        spiral_geo.node_tree = bpy.data.node_groups.get("ALBPY_NG_SpiralGalaxy")
        # Smooth shading für Spiral Galaxy
        set_smooth_spiral = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_spiral.inputs["Shade Smooth"].default_value = True

        elliptical_geo = self.nodes.new("GeometryNodeGroup")
        elliptical_geo.node_tree = bpy.data.node_groups.get("ALBPY_NG_EllipticalGalaxy")
        # Smooth shading für Elliptical Galaxy
        set_smooth_elliptical = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_elliptical.inputs["Shade Smooth"].default_value = True

        irregular_geo = self.nodes.new("GeometryNodeGroup")
        irregular_geo.node_tree = bpy.data.node_groups.get("ALBPY_NG_IrregularGalaxy")
        # Smooth shading für Irregular Galaxy
        set_smooth_irregular = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_irregular.inputs["Shade Smooth"].default_value = True

        # Position transformations
        spiral_transform = self.nodes.new("GeometryNodeTransform")
        elliptical_transform = self.nodes.new("GeometryNodeTransform")
        irregular_transform = self.nodes.new("GeometryNodeTransform")

        # Join geometries
        join_geometry = self.nodes.new("GeometryNodeJoinGeometry")

        # Position nodes
        input_node.location = (-800, 0)
        spiral_geo.location = (-600, 200)
        set_smooth_spiral.location = (-500, 200)
        elliptical_geo.location = (-600, 0)
        set_smooth_elliptical.location = (-500, 0)
        irregular_geo.location = (-600, -200)
        set_smooth_irregular.location = (-500, -200)
        spiral_transform.location = (-400, 200)
        elliptical_transform.location = (-400, 0)
        irregular_transform.location = (-400, -200)
        join_geometry.location = (-200, 0)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(
            input_node.outputs["Spiral Stars"], spiral_geo.inputs["Star Count"]
        )
        self.links.new(
            input_node.outputs["Elliptical Stars"], elliptical_geo.inputs["Star Count"]
        )
        self.links.new(
            input_node.outputs["Irregular Stars"], irregular_geo.inputs["Star Count"]
        )

        self.links.new(
            spiral_geo.outputs["Geometry"], set_smooth_spiral.inputs["Geometry"]
        )
        self.links.new(
            set_smooth_spiral.outputs["Geometry"], spiral_transform.inputs["Geometry"]
        )
        self.links.new(
            elliptical_geo.outputs["Geometry"], set_smooth_elliptical.inputs["Geometry"]
        )
        self.links.new(
            set_smooth_elliptical.outputs["Geometry"],
            elliptical_transform.inputs["Geometry"],
        )
        self.links.new(
            irregular_geo.outputs["Geometry"], set_smooth_irregular.inputs["Geometry"]
        )
        self.links.new(
            set_smooth_irregular.outputs["Geometry"],
            irregular_transform.inputs["Geometry"],
        )

        self.links.new(
            spiral_transform.outputs["Geometry"], join_geometry.inputs["Geometry"]
        )
        self.links.new(
            elliptical_transform.outputs["Geometry"], join_geometry.inputs["Geometry"]
        )
        self.links.new(
            irregular_transform.outputs["Geometry"], join_geometry.inputs["Geometry"]
        )

        self.links.new(
            join_geometry.outputs["Geometry"], output_node.inputs["Geometry"]
        )

        # Set transform positions
        spiral_transform.inputs["Translation"].default_value = (-15, 0, 0)
        elliptical_transform.inputs["Translation"].default_value = (0, 0, 0)
        irregular_transform.inputs["Translation"].default_value = (15, 0, 0)


# Galaxy comparison presets
GALAXY_COMPARISON_PRESETS = {
    "standard_comparison": {
        "Spacing": 15.0,
        "Spiral Stars": 30000,
        "Elliptical Stars": 25000,
        "Irregular Stars": 15000,
    },
    "dense_comparison": {
        "Spacing": 10.0,
        "Spiral Stars": 50000,
        "Elliptical Stars": 40000,
        "Irregular Stars": 25000,
    },
    "sparse_comparison": {
        "Spacing": 25.0,
        "Spiral Stars": 15000,
        "Elliptical Stars": 12000,
        "Irregular Stars": 8000,
    },
    "massive_galaxies": {
        "Spacing": 20.0,
        "Spiral Stars": 100000,
        "Elliptical Stars": 80000,
        "Irregular Stars": 50000,
    },
}


def get_galaxy_comparison_preset(preset_name: str) -> dict:
    """Get preset for galaxy comparison configuration."""
    return GALAXY_COMPARISON_PRESETS.get(
        preset_name, GALAXY_COMPARISON_PRESETS["standard_comparison"]
    )


def apply_galaxy_comparison_preset(
    object_obj: bpy.types.Object, preset_name: str
) -> None:
    """Apply galaxy comparison preset to object."""
    preset = get_galaxy_comparison_preset(preset_name)

    # Find geometry nodes modifier
    for modifier in object_obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_NG_GalaxyComparison" in modifier.node_group.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in modifier:
                        modifier[param_name] = value
                break


def register():
    # REMOVED: bpy.utils.register_class(AlbpyGalaxyComparisonGeometryGroup)

    bpy.utils.register_class(AlbpyGalaxyComparisonGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyGalaxyComparisonGeometryGroup)
