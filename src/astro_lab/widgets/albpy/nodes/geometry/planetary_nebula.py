import bpy


class AlbpyPlanetaryNebulaGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_PlanetaryNebula"
    bl_label = "Albpy Planetary Nebula Geometry Group"

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
            name="Nebula Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Shell Thickness", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Displacement Strength", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Noise Scale", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 10.0  # Nebula Size
        self.interface.items_tree["Input_3"].default_value = 0.2  # Shell Thickness
        self.interface.items_tree[
            "Input_4"
        ].default_value = 0.1  # Displacement Strength
        self.interface.items_tree["Input_5"].default_value = 2.0  # Noise Scale

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create sphere
        sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Sphäre
        set_smooth_sphere = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_sphere.inputs["Shade Smooth"].default_value = True

        # Add displacement for realistic shape
        displace = self.nodes.new("GeometryNodeSetPosition")

        # Noise for displacement
        noise = self.nodes.new("GeometryNodeTexNoise")
        noise.inputs["Detail"].default_value = 2.0

        # Scale noise
        scale_noise = self.nodes.new("GeometryNodeVectorMath")
        scale_noise.operation = "MULTIPLY"

        # Add displacement to position
        add_displacement = self.nodes.new("GeometryNodeVectorMath")
        add_displacement.operation = "ADD"

        # Position nodes
        input_node.location = (-1000, 0)
        sphere.location = (-800, 0)
        set_smooth_sphere.location = (-700, 0)
        noise.location = (-800, -200)
        scale_noise.location = (-600, -200)
        displace.location = (-400, 0)
        add_displacement.location = (-200, 0)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Nebula Size"], sphere.inputs["Radius"])
        self.links.new(sphere.outputs["Mesh"], set_smooth_sphere.inputs["Geometry"])
        self.links.new(
            set_smooth_sphere.outputs["Geometry"], displace.inputs["Geometry"]
        )
        self.links.new(input_node.outputs["Noise Scale"], noise.inputs["Scale"])
        self.links.new(noise.outputs["Fac"], scale_noise.inputs[0])
        self.links.new(
            input_node.outputs["Displacement Strength"], scale_noise.inputs[1]
        )
        self.links.new(displace.outputs["Position"], add_displacement.inputs[0])
        self.links.new(scale_noise.outputs["Vector"], add_displacement.inputs[1])
        self.links.new(add_displacement.outputs["Vector"], displace.inputs["Offset"])
        self.links.new(displace.outputs["Geometry"], output_node.inputs["Geometry"])


# Planetary nebula presets
PLANETARY_NEBULA_PRESETS = {
    "classic_planetary": {
        "Nebula Size": 10.0,
        "Shell Thickness": 0.2,
        "Displacement Strength": 0.1,
        "Noise Scale": 2.0,
    },
    "complex_planetary": {
        "Nebula Size": 15.0,
        "Shell Thickness": 0.3,
        "Displacement Strength": 0.2,
        "Noise Scale": 3.0,
    },
    "smooth_planetary": {
        "Nebula Size": 8.0,
        "Shell Thickness": 0.15,
        "Displacement Strength": 0.05,
        "Noise Scale": 1.5,
    },
    "turbulent_planetary": {
        "Nebula Size": 12.0,
        "Shell Thickness": 0.25,
        "Displacement Strength": 0.3,
        "Noise Scale": 4.0,
    },
}


def get_planetary_nebula_preset(preset_name: str) -> dict:
    """Get preset for planetary nebula configuration."""
    return PLANETARY_NEBULA_PRESETS.get(
        preset_name, PLANETARY_NEBULA_PRESETS["classic_planetary"]
    )


def apply_planetary_nebula_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply planetary nebula preset to node group."""
    preset = get_planetary_nebula_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyPlanetaryNebulaGeometryGroup)

    bpy.utils.register_class(AlbpyPlanetaryNebulaGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyPlanetaryNebulaGeometryGroup)
