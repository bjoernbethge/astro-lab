import bpy


class AlbpyStarFormingRegionGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_StarFormingRegion"
    bl_label = "Albpy Star Forming Region Geometry Group"

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
            name="Region Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Density Falloff", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Star Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 1000  # Star Count
        self.interface.items_tree["Input_3"].default_value = 5.0  # Region Size
        self.interface.items_tree["Input_4"].default_value = 2.0  # Density Falloff
        self.interface.items_tree["Input_5"].default_value = 0.05  # Star Size

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create sphere for region
        sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Region
        set_smooth_region = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_region.inputs["Shade Smooth"].default_value = True

        # Distribute points on sphere
        distribute = self.nodes.new("GeometryNodeDistributePointsOnFaces")

        # Instance on points
        instance_on_points = self.nodes.new("GeometryNodeInstanceOnPoints")

        # Create star geometry (small sphere)
        star_sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        star_sphere.inputs["Radius"].default_value = 0.05
        # Smooth shading für Sterne
        set_smooth_star = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_star.inputs["Shade Smooth"].default_value = True

        # Scale instances
        scale_instances = self.nodes.new("GeometryNodeScaleInstances")

        # Add noise for density variation
        noise = self.nodes.new("GeometryNodeTexNoise")
        color_ramp = self.nodes.new("GeometryNodeValToRGB")

        # Position nodes
        input_node.location = (-1000, 0)
        sphere.location = (-800, 0)
        set_smooth_region.location = (-700, 0)
        distribute.location = (-600, 0)
        noise.location = (-800, -200)
        color_ramp.location = (-600, -200)
        instance_on_points.location = (-400, 0)
        star_sphere.location = (-600, 200)
        set_smooth_star.location = (-500, 200)
        scale_instances.location = (-200, 0)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Region Size"], sphere.inputs["Radius"])
        self.links.new(sphere.outputs["Mesh"], set_smooth_region.inputs["Geometry"])
        self.links.new(set_smooth_region.outputs["Geometry"], distribute.inputs["Mesh"])
        self.links.new(input_node.outputs["Star Count"], distribute.inputs["Density"])
        self.links.new(
            distribute.outputs["Points"], instance_on_points.inputs["Points"]
        )
        self.links.new(star_sphere.outputs["Mesh"], set_smooth_star.inputs["Geometry"])
        self.links.new(
            set_smooth_star.outputs["Geometry"], instance_on_points.inputs["Instance"]
        )
        self.links.new(
            instance_on_points.outputs["Instances"], scale_instances.inputs["Instances"]
        )
        self.links.new(
            scale_instances.outputs["Instances"], output_node.inputs["Geometry"]
        )

        # Connect noise for density
        self.links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
        self.links.new(color_ramp.outputs["Color"], distribute.inputs["Density"])

        # Connect scaling
        self.links.new(input_node.outputs["Star Size"], scale_instances.inputs["Scale"])

        # Set properties
        noise.inputs["Scale"].default_value = 5.0
        noise.inputs["Detail"].default_value = 2.0
        color_ramp.color_ramp.elements[0].position = 0.3
        color_ramp.color_ramp.elements[1].position = 0.7


# Star forming region presets
STAR_FORMING_REGION_PRESETS = {
    "dense_cluster": {
        "Star Count": 2000,
        "Region Size": 3.0,
        "Density Falloff": 3.0,
        "Star Size": 0.03,
    },
    "sparse_cluster": {
        "Star Count": 500,
        "Region Size": 8.0,
        "Density Falloff": 1.5,
        "Star Size": 0.08,
    },
    "giant_region": {
        "Star Count": 5000,
        "Region Size": 15.0,
        "Density Falloff": 2.5,
        "Star Size": 0.05,
    },
    "compact_region": {
        "Star Count": 1000,
        "Region Size": 2.0,
        "Density Falloff": 4.0,
        "Star Size": 0.02,
    },
}


def get_star_forming_region_preset(preset_name: str) -> dict:
    """Get preset for star forming region configuration."""
    return STAR_FORMING_REGION_PRESETS.get(
        preset_name, STAR_FORMING_REGION_PRESETS["dense_cluster"]
    )


def apply_star_forming_region_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply star forming region preset to node group."""
    preset = get_star_forming_region_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyStarFormingRegionGeometryGroup)

    bpy.utils.register_class(AlbpyStarFormingRegionGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyStarFormingRegionGeometryGroup)
