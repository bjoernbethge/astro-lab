import bpy


class AlbpyEllipticalGalaxyGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_EllipticalGalaxy"
    bl_label = "Albpy Elliptical Galaxy Geometry Group"

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
            name="Ellipticity X", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Ellipticity Y", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 25000  # Star Count
        self.interface.items_tree["Input_3"].default_value = 15.0  # Galaxy Radius
        self.interface.items_tree["Input_4"].default_value = 0.7  # Ellipticity X
        self.interface.items_tree["Input_5"].default_value = 0.5  # Ellipticity Y

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create ellipsoid
        uv_sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Ellipsoid
        set_smooth_ellipsoid = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_ellipsoid.inputs["Shade Smooth"].default_value = True

        # Scale to elliptical shape
        transform = self.nodes.new("GeometryNodeTransform")

        # Distribute points
        distribute = self.nodes.new("GeometryNodeDistributePointsOnFaces")

        # Instance stars
        ico_sphere = self.nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.inputs["Radius"].default_value = 0.015
        # Smooth shading für Sterne
        set_smooth_star = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_star.inputs["Shade Smooth"].default_value = True

        instance = self.nodes.new("GeometryNodeInstanceOnPoints")

        # Position nodes
        input_node.location = (-600, 0)
        uv_sphere.location = (-400, 0)
        set_smooth_ellipsoid.location = (-300, 0)
        transform.location = (-200, 0)
        distribute.location = (0, 0)
        ico_sphere.location = (0, 200)
        set_smooth_star.location = (100, 200)
        instance.location = (200, 0)
        output_node.location = (400, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Galaxy Radius"], uv_sphere.inputs["Radius"])
        self.links.new(
            uv_sphere.outputs["Mesh"], set_smooth_ellipsoid.inputs["Geometry"]
        )
        self.links.new(
            set_smooth_ellipsoid.outputs["Geometry"], transform.inputs["Geometry"]
        )
        self.links.new(transform.outputs["Geometry"], distribute.inputs["Mesh"])
        self.links.new(input_node.outputs["Star Count"], distribute.inputs["Density"])
        self.links.new(distribute.outputs["Points"], instance.inputs["Points"])
        self.links.new(ico_sphere.outputs["Mesh"], set_smooth_star.inputs["Geometry"])
        self.links.new(set_smooth_star.outputs["Geometry"], instance.inputs["Instance"])
        self.links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])

        # Connect ellipticity parameters
        self.links.new(
            input_node.outputs["Ellipticity X"],
            transform.inputs["Scale"].driver_add("default_value", 0).driver,
        )
        self.links.new(
            input_node.outputs["Ellipticity Y"],
            transform.inputs["Scale"].driver_add("default_value", 1).driver,
        )


def register():
    # REMOVED: bpy.utils.register_class(AlbpyEllipticalGalaxyGeometryGroup)

    bpy.utils.register_class(AlbpyEllipticalGalaxyGeometryGroup)


# Elliptical galaxy presets
ELLIPTICAL_GALAXY_PRESETS = {
    "giant_elliptical": {
        "Star Count": 25000,
        "Galaxy Radius": 15.0,
        "Ellipticity X": 0.7,
        "Ellipticity Y": 0.5,
    },
    "dwarf_elliptical": {
        "Star Count": 10000,
        "Galaxy Radius": 8.0,
        "Ellipticity X": 0.8,
        "Ellipticity Y": 0.6,
    },
    "cD_galaxy": {
        "Star Count": 50000,
        "Galaxy Radius": 30.0,
        "Ellipticity X": 0.9,
        "Ellipticity Y": 0.7,
    },
    "compact_elliptical": {
        "Star Count": 5000,
        "Galaxy Radius": 5.0,
        "Ellipticity X": 0.6,
        "Ellipticity Y": 0.4,
    },
}


def get_elliptical_galaxy_preset(preset_name: str) -> dict:
    """Get preset for elliptical galaxy configuration."""
    return ELLIPTICAL_GALAXY_PRESETS.get(
        preset_name, ELLIPTICAL_GALAXY_PRESETS["giant_elliptical"]
    )


def apply_elliptical_galaxy_preset(
    object_obj: bpy.types.Object, preset_name: str
) -> None:
    """Apply elliptical galaxy preset to object."""
    preset = get_elliptical_galaxy_preset(preset_name)

    # Find geometry nodes modifier
    for modifier in object_obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_NG_EllipticalGalaxy" in modifier.node_group.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in modifier:
                        modifier[param_name] = value
                break


def unregister():
    bpy.utils.unregister_class(AlbpyEllipticalGalaxyGeometryGroup)
