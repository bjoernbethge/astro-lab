import bpy


class AlbpyStellarWindGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_StellarWind"
    bl_label = "Albpy Stellar Wind Geometry Group"

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
            name="Wind Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Wind Speed", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Mass Loss Rate", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Density Falloff", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 5.0  # Wind Radius
        self.interface.items_tree["Input_3"].default_value = 500.0  # Wind Speed
        self.interface.items_tree["Input_4"].default_value = 1e-6  # Mass Loss Rate
        self.interface.items_tree["Input_5"].default_value = 2.0  # Density Falloff

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create wind sphere
        sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Wind-Sphäre
        set_smooth_sphere = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_sphere.inputs["Shade Smooth"].default_value = True

        # Add radial falloff for density
        set_position = self.nodes.new("GeometryNodeSetPosition")

        # Get position for radial calculation
        position = self.nodes.new("GeometryNodeInputPosition")

        # Calculate distance from center
        vector_length = self.nodes.new("GeometryNodeVectorMath")
        vector_length.operation = "LENGTH"

        # Calculate density falloff
        divide = self.nodes.new("GeometryNodeMath")
        divide.operation = "DIVIDE"

        # Power for falloff curve
        power = self.nodes.new("GeometryNodeMath")
        power.operation = "POWER"

        # Subtract from 1 for inverse falloff
        subtract = self.nodes.new("GeometryNodeMath")
        subtract.operation = "SUBTRACT"
        subtract.inputs[0].default_value = 1.0

        # Scale position based on density
        scale_position = self.nodes.new("GeometryNodeVectorMath")
        scale_position.operation = "MULTIPLY"

        # Position nodes
        input_node.location = (-1200, 0)
        sphere.location = (-1000, 0)
        set_smooth_sphere.location = (-900, 0)
        position.location = (-1000, -200)
        vector_length.location = (-800, -200)
        divide.location = (-600, -200)
        power.location = (-400, -200)
        subtract.location = (-200, -200)
        scale_position.location = (0, -200)
        set_position.location = (200, 0)
        output_node.location = (400, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Wind Radius"], sphere.inputs["Radius"])
        self.links.new(sphere.outputs["Mesh"], set_smooth_sphere.inputs["Geometry"])
        self.links.new(
            set_smooth_sphere.outputs["Geometry"], set_position.inputs["Geometry"]
        )
        self.links.new(position.outputs["Position"], vector_length.inputs[0])
        self.links.new(vector_length.outputs["Value"], divide.inputs[0])
        self.links.new(input_node.outputs["Wind Radius"], divide.inputs[1])
        self.links.new(divide.outputs["Value"], power.inputs[0])
        self.links.new(input_node.outputs["Density Falloff"], power.inputs[1])
        self.links.new(power.outputs["Value"], subtract.inputs[1])
        self.links.new(subtract.outputs["Value"], scale_position.inputs[0])
        self.links.new(position.outputs["Position"], scale_position.inputs[1])
        self.links.new(scale_position.outputs["Vector"], set_position.inputs["Offset"])
        self.links.new(set_position.outputs["Geometry"], output_node.inputs["Geometry"])


# Stellar wind presets
STELLAR_WIND_PRESETS = {
    "solar_wind": {
        "Wind Radius": 3.0,
        "Wind Speed": 400.0,
        "Mass Loss Rate": 1e-14,
        "Density Falloff": 2.0,
    },
    "massive_star_wind": {
        "Wind Radius": 10.0,
        "Wind Speed": 2000.0,
        "Mass Loss Rate": 1e-5,
        "Density Falloff": 1.5,
    },
    "red_giant_wind": {
        "Wind Radius": 20.0,
        "Wind Speed": 10.0,
        "Mass Loss Rate": 1e-7,
        "Density Falloff": 3.0,
    },
    "wolf_rayet_wind": {
        "Wind Radius": 15.0,
        "Wind Speed": 3000.0,
        "Mass Loss Rate": 1e-4,
        "Density Falloff": 1.2,
    },
}


def get_stellar_wind_preset(preset_name: str) -> dict:
    """Get preset for stellar wind configuration."""
    return STELLAR_WIND_PRESETS.get(preset_name, STELLAR_WIND_PRESETS["solar_wind"])


def apply_stellar_wind_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply stellar wind preset to node group."""
    preset = get_stellar_wind_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyStellarWindGeometryGroup)

    bpy.utils.register_class(AlbpyStellarWindGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyStellarWindGeometryGroup)
