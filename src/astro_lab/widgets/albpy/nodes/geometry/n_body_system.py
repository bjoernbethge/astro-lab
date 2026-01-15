import bpy


class AlbpyNBodySystemGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_NBodySystem"
    bl_label = "Albpy N-Body System Geometry Group"

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
            name="Body Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="System Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Mass Range", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Body Size Scale", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 10  # Body Count
        self.interface.items_tree["Input_3"].default_value = 20.0  # System Radius
        self.interface.items_tree["Input_4"].default_value = 1.0  # Mass Range
        self.interface.items_tree["Input_5"].default_value = 0.1  # Body Size Scale

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create points for body positions
        distribute_points = self.nodes.new("GeometryNodeDistributePointsOnFaces")

        # Create sphere for distribution
        sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Verteilungssphäre
        set_smooth_dist = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_dist.inputs["Shade Smooth"].default_value = True

        # Instance on points
        instance_on_points = self.nodes.new("GeometryNodeInstanceOnPoints")

        # Create body geometry (sphere)
        body_sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Körper
        set_smooth_body = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_body.inputs["Shade Smooth"].default_value = True

        # Scale instances based on mass
        scale_instances = self.nodes.new("GeometryNodeScaleInstances")

        # Random value for mass variation
        random_value = self.nodes.new("GeometryNodeRandomValue")
        random_value.data_type = "FLOAT"

        # Math for mass scaling
        multiply_mass = self.nodes.new("GeometryNodeMath")
        multiply_mass.operation = "MULTIPLY"

        # Position nodes
        input_node.location = (-1200, 0)
        sphere.location = (-1000, 0)
        set_smooth_dist.location = (-900, 0)
        distribute_points.location = (-800, 0)
        random_value.location = (-1000, -200)
        multiply_mass.location = (-800, -200)
        instance_on_points.location = (-600, 0)
        body_sphere.location = (-800, 200)
        set_smooth_body.location = (-700, 200)
        scale_instances.location = (-400, 0)
        output_node.location = (-200, 0)

        # Connect nodes
        self.links.new(input_node.outputs["System Radius"], sphere.inputs["Radius"])
        self.links.new(sphere.outputs["Mesh"], set_smooth_dist.inputs["Geometry"])
        self.links.new(
            set_smooth_dist.outputs["Geometry"], distribute_points.inputs["Mesh"]
        )
        self.links.new(
            input_node.outputs["Body Count"], distribute_points.inputs["Density"]
        )
        self.links.new(
            distribute_points.outputs["Points"], instance_on_points.inputs["Points"]
        )
        self.links.new(body_sphere.outputs["Mesh"], set_smooth_body.inputs["Geometry"])
        self.links.new(
            set_smooth_body.outputs["Geometry"], instance_on_points.inputs["Instance"]
        )
        self.links.new(
            instance_on_points.outputs["Instances"], scale_instances.inputs["Instances"]
        )
        self.links.new(
            scale_instances.outputs["Instances"], output_node.inputs["Geometry"]
        )

        # Connect mass scaling
        self.links.new(random_value.outputs["Value"], multiply_mass.inputs[0])
        self.links.new(input_node.outputs["Mass Range"], multiply_mass.inputs[1])
        self.links.new(multiply_mass.outputs["Value"], scale_instances.inputs["Scale"])

        # Set properties
        random_value.inputs["Min"].default_value = 0.1
        random_value.inputs["Max"].default_value = 1.0


# N-body system presets
N_BODY_SYSTEM_PRESETS = {
    "solar_system": {
        "Body Count": 8,
        "System Radius": 30.0,
        "Mass Range": 0.5,
        "Body Size Scale": 0.2,
    },
    "star_cluster": {
        "Body Count": 50,
        "System Radius": 10.0,
        "Mass Range": 2.0,
        "Body Size Scale": 0.1,
    },
    "galaxy_core": {
        "Body Count": 100,
        "System Radius": 5.0,
        "Mass Range": 5.0,
        "Body Size Scale": 0.05,
    },
    "binary_system": {
        "Body Count": 2,
        "System Radius": 2.0,
        "Mass Range": 1.0,
        "Body Size Scale": 0.3,
    },
}


def get_n_body_system_preset(preset_name: str) -> dict:
    """Get preset for N-body system configuration."""
    return N_BODY_SYSTEM_PRESETS.get(preset_name, N_BODY_SYSTEM_PRESETS["solar_system"])


def apply_n_body_system_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply N-body system preset to node group."""
    preset = get_n_body_system_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyNBodySystemGeometryGroup)

    bpy.utils.register_class(AlbpyNBodySystemGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyNBodySystemGeometryGroup)
