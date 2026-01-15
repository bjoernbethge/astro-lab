import bpy


class AlbpyParticleGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_Particle"
    bl_label = "Albpy Particle Geometry Group"

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
            name="Particle Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="Particle Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="System Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Velocity Scale", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 1000  # Particle Count
        self.interface.items_tree["Input_3"].default_value = 0.02  # Particle Size
        self.interface.items_tree["Input_4"].default_value = 10.0  # System Radius
        self.interface.items_tree["Input_5"].default_value = 1.0  # Velocity Scale

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create sphere for particle distribution
        sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Verteilungssphäre
        set_smooth_dist = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_dist.inputs["Shade Smooth"].default_value = True

        # Distribute points for particles
        distribute_points = self.nodes.new("GeometryNodeDistributePointsOnFaces")

        # Instance on points
        instance_on_points = self.nodes.new("GeometryNodeInstanceOnPoints")

        # Create particle geometry (small sphere)
        particle_sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Partikel
        set_smooth_particle = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_particle.inputs["Shade Smooth"].default_value = True

        # Random size variation
        random_size = self.nodes.new("GeometryNodeRandomValue")
        random_size.data_type = "FLOAT"
        random_size.inputs["Min"].default_value = 0.5
        random_size.inputs["Max"].default_value = 2.0

        # Scale instances
        scale_instances = self.nodes.new("GeometryNodeScaleInstances")

        # Add velocity vectors
        random_velocity = self.nodes.new("GeometryNodeRandomValue")
        random_velocity.data_type = "FLOAT_VECTOR"
        random_velocity.inputs["Min"].default_value = (-1, -1, -1)
        random_velocity.inputs["Max"].default_value = (1, 1, 1)

        # Set velocity
        set_velocity = self.nodes.new("GeometryNodeSetPosition")

        # Position nodes
        input_node.location = (-1200, 0)
        sphere.location = (-1000, 0)
        set_smooth_dist.location = (-900, 0)
        distribute_points.location = (-800, 0)
        particle_sphere.location = (-800, 200)
        set_smooth_particle.location = (-700, 200)
        random_size.location = (-800, -200)
        scale_instances.location = (-600, 0)
        random_velocity.location = (-600, -200)
        set_velocity.location = (-400, 0)
        instance_on_points.location = (-200, 0)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(input_node.outputs["System Radius"], sphere.inputs["Radius"])
        self.links.new(sphere.outputs["Mesh"], set_smooth_dist.inputs["Geometry"])
        self.links.new(
            set_smooth_dist.outputs["Geometry"], distribute_points.inputs["Mesh"]
        )
        self.links.new(
            input_node.outputs["Particle Count"], distribute_points.inputs["Density"]
        )

        self.links.new(
            input_node.outputs["Particle Size"], particle_sphere.inputs["Radius"]
        )
        self.links.new(
            particle_sphere.outputs["Mesh"], set_smooth_particle.inputs["Geometry"]
        )
        self.links.new(
            set_smooth_particle.outputs["Geometry"],
            instance_on_points.inputs["Instance"],
        )

        self.links.new(random_size.outputs["Value"], scale_instances.inputs["Scale"])
        self.links.new(
            instance_on_points.outputs["Instances"], scale_instances.inputs["Instances"]
        )

        self.links.new(random_velocity.outputs["Value"], set_velocity.inputs["Offset"])
        self.links.new(
            scale_instances.outputs["Instances"], set_velocity.inputs["Geometry"]
        )

        self.links.new(
            distribute_points.outputs["Points"], instance_on_points.inputs["Points"]
        )
        self.links.new(set_velocity.outputs["Geometry"], output_node.inputs["Geometry"])


# Particle presets
PARTICLE_PRESETS = {
    "cosmic_dust": {
        "Particle Count": 5000,
        "Particle Size": 0.01,
        "System Radius": 15.0,
        "Velocity Scale": 0.5,
    },
    "stellar_wind": {
        "Particle Count": 2000,
        "Particle Size": 0.05,
        "System Radius": 8.0,
        "Velocity Scale": 2.0,
    },
    "dark_matter": {
        "Particle Count": 10000,
        "Particle Size": 0.005,
        "System Radius": 20.0,
        "Velocity Scale": 0.1,
    },
}


def get_particle_preset(preset_name: str) -> dict:
    """Get preset for particle configuration."""
    return PARTICLE_PRESETS.get(preset_name, PARTICLE_PRESETS["cosmic_dust"])


def apply_particle_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply particle preset to node group."""
    preset = get_particle_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    bpy.utils.register_class(AlbpyParticleGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyParticleGeometryGroup)
