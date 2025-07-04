import bpy


class AlbpyGalacticDustGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_GalacticDust"
    bl_label = "Albpy Galactic Dust Geometry Group"

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
            name="Dust Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Dust Density", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Noise Scale", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Dust Color", in_out="INPUT", socket_type="NodeSocketColor"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 20.0  # Dust Size
        self.interface.items_tree["Input_3"].default_value = 0.01  # Dust Density
        self.interface.items_tree["Input_4"].default_value = 10.0  # Noise Scale
        self.interface.items_tree["Input_5"].default_value = (
            0.4,
            0.3,
            0.2,
            1.0,
        )  # Dust Color

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create volume cube
        cube = self.nodes.new("GeometryNodeMeshCube")
        # Smooth shading für Cube
        set_smooth_cube = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_cube.inputs["Shade Smooth"].default_value = True

        # Scale cube
        transform = self.nodes.new("GeometryNodeTransform")

        # Add noise for dust structure
        noise = self.nodes.new("GeometryNodeTexNoise")
        color_ramp = self.nodes.new("GeometryNodeValToRGB")

        # Set material
        set_material = self.nodes.new("GeometryNodeSetMaterial")

        # Create dust material
        dust_material = self.nodes.new("GeometryNodeMaterial")
        dust_material.material = self._create_dust_material()

        # Position nodes
        input_node.location = (-800, 0)
        cube.location = (-600, 0)
        set_smooth_cube.location = (-500, 0)
        transform.location = (-400, 0)
        noise.location = (-600, -200)
        color_ramp.location = (-400, -200)
        set_material.location = (-200, 0)
        dust_material.location = (-400, 200)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Dust Size"], transform.inputs["Scale"])
        self.links.new(cube.outputs["Mesh"], set_smooth_cube.inputs["Geometry"])
        self.links.new(
            set_smooth_cube.outputs["Geometry"], transform.inputs["Geometry"]
        )
        self.links.new(transform.outputs["Geometry"], set_material.inputs["Geometry"])
        self.links.new(
            dust_material.outputs["Material"], set_material.inputs["Material"]
        )
        self.links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])

        # Connect noise parameters
        self.links.new(input_node.outputs["Noise Scale"], noise.inputs["Scale"])
        self.links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])

        # Set properties
        noise.inputs["Detail"].default_value = 2.0
        color_ramp.color_ramp.elements[0].position = 0.4
        color_ramp.color_ramp.elements[1].position = 0.6

    def _create_dust_material(self):
        """Create dust material for the node group."""
        mat = bpy.data.materials.new(name="GalacticDust")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Volume scatter
        volume_scatter = nodes.new("ShaderNodeVolumeScatter")
        volume_scatter.inputs["Color"].default_value = (0.4, 0.3, 0.2, 1.0)
        volume_scatter.inputs["Density"].default_value = 0.01

        # Output
        output = nodes.new("ShaderNodeOutputMaterial")

        # Connect
        links.new(volume_scatter.outputs["Volume"], output.inputs["Volume"])

        return mat


# Galactic dust presets
GALACTIC_DUST_PRESETS = {
    "spiral_dust": {
        "Dust Size": 20.0,
        "Dust Density": 0.01,
        "Noise Scale": 10.0,
        "Dust Color": (0.4, 0.3, 0.2, 1.0),
    },
    "irregular_dust": {
        "Dust Size": 15.0,
        "Dust Density": 0.015,
        "Noise Scale": 8.0,
        "Dust Color": (0.3, 0.2, 0.1, 1.0),
    },
    "thick_dust": {
        "Dust Size": 25.0,
        "Dust Density": 0.02,
        "Noise Scale": 12.0,
        "Dust Color": (0.5, 0.4, 0.3, 1.0),
    },
    "thin_dust": {
        "Dust Size": 18.0,
        "Dust Density": 0.005,
        "Noise Scale": 15.0,
        "Dust Color": (0.2, 0.15, 0.1, 1.0),
    },
}


def get_galactic_dust_preset(preset_name: str) -> dict:
    """Get preset for galactic dust configuration."""
    return GALACTIC_DUST_PRESETS.get(preset_name, GALACTIC_DUST_PRESETS["spiral_dust"])


def apply_galactic_dust_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply galactic dust preset to node group."""
    preset = get_galactic_dust_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyGalacticDustGeometryGroup)

    bpy.utils.register_class(AlbpyGalacticDustGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyGalacticDustGeometryGroup)
