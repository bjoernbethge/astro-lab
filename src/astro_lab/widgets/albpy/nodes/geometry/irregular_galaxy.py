import bpy


class AlbpyIrregularGalaxyGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_IrregularGalaxy"
    bl_label = "Albpy Irregular Galaxy Geometry Group"

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
            name="Star Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="Galaxy Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Irregularity", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 15000  # Star Count
        self.interface.items_tree["Input_3"].default_value = 10.0  # Galaxy Radius
        self.interface.items_tree["Input_4"].default_value = 0.3  # Irregularity

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create irregular shape with noise
        ico_sphere = self.nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.inputs["Subdivisions"].default_value = 3
        # Smooth shading für Grundform
        set_smooth_base = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_base.inputs["Shade Smooth"].default_value = True

        # Add noise for irregularity
        noise = self.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 2.0

        # Displace geometry
        displace = self.nodes.new("GeometryNodeSetPosition")

        # Distribute points
        distribute = self.nodes.new("GeometryNodeDistributePointsOnFaces")

        # Instance stars
        star_sphere = self.nodes.new("GeometryNodeMeshIcoSphere")
        star_sphere.inputs["Radius"].default_value = 0.02
        # Smooth shading für Sterne
        set_smooth_star = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_star.inputs["Shade Smooth"].default_value = True

        instance = self.nodes.new("GeometryNodeInstanceOnPoints")

        # Position nodes
        input_node.location = (-800, 0)
        ico_sphere.location = (-600, 0)
        set_smooth_base.location = (-500, 0)
        noise.location = (-600, 200)
        displace.location = (-400, 0)
        distribute.location = (-200, 0)
        star_sphere.location = (-200, 200)
        set_smooth_star.location = (-100, 200)
        instance.location = (0, 0)
        output_node.location = (200, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Galaxy Radius"], ico_sphere.inputs["Radius"])
        self.links.new(ico_sphere.outputs["Mesh"], set_smooth_base.inputs["Geometry"])
        self.links.new(set_smooth_base.outputs["Geometry"], displace.inputs["Geometry"])
        self.links.new(noise.outputs["Fac"], displace.inputs["Offset"])
        self.links.new(input_node.outputs["Irregularity"], noise.inputs["Scale"])
        self.links.new(displace.outputs["Geometry"], distribute.inputs["Mesh"])
        self.links.new(input_node.outputs["Star Count"], distribute.inputs["Density"])
        self.links.new(distribute.outputs["Points"], instance.inputs["Points"])
        self.links.new(star_sphere.outputs["Mesh"], set_smooth_star.inputs["Geometry"])
        self.links.new(set_smooth_star.outputs["Geometry"], instance.inputs["Instance"])
        self.links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])


def register():
    # REMOVED: bpy.utils.register_class(AlbpyIrregularGalaxyGeometryGroup)

    bpy.utils.register_class(AlbpyIrregularGalaxyGeometryGroup)


# Irregular galaxy presets
IRREGULAR_GALAXY_PRESETS = {
    "magellanic_cloud": {
        "Star Count": 15000,
        "Galaxy Radius": 10.0,
        "Irregularity": 0.3,
    },
    "dwarf_irregular": {
        "Star Count": 5000,
        "Galaxy Radius": 5.0,
        "Irregularity": 0.5,
    },
    "starburst_irregular": {
        "Star Count": 25000,
        "Galaxy Radius": 12.0,
        "Irregularity": 0.4,
    },
    "tidal_dwarf": {
        "Star Count": 8000,
        "Galaxy Radius": 7.0,
        "Irregularity": 0.6,
    },
}


def get_irregular_galaxy_preset(preset_name: str) -> dict:
    """Get preset for irregular galaxy configuration."""
    return IRREGULAR_GALAXY_PRESETS.get(
        preset_name, IRREGULAR_GALAXY_PRESETS["magellanic_cloud"]
    )


def apply_irregular_galaxy_preset(
    object_obj: bpy.types.Object, preset_name: str
) -> None:
    """Apply irregular galaxy preset to object."""
    preset = get_irregular_galaxy_preset(preset_name)

    # Find geometry nodes modifier
    for modifier in object_obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_NG_IrregularGalaxy" in modifier.node_group.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in modifier:
                        modifier[param_name] = value
                break


def unregister():
    bpy.utils.unregister_class(AlbpyIrregularGalaxyGeometryGroup)
