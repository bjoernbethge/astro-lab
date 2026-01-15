import bpy


class AlbpyLensGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_Lens"
    bl_label = "Albpy Lens Geometry Group"

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
            name="Lens Strength", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Lens Radius", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Distortion Type", in_out="INPUT", socket_type="NodeSocketString"
        )
        self.interface.new_socket(
            name="Focus Distance", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 0.5  # Lens Strength
        self.interface.items_tree["Input_3"].default_value = 5.0  # Lens Radius
        self.interface.items_tree[
            "Input_4"
        ].default_value = "elliptical"  # Distortion Type
        self.interface.items_tree["Input_5"].default_value = 10.0  # Focus Distance

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create lens geometry (sphere)
        lens_sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        # Smooth shading für Linsensphäre
        set_smooth_lens = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_lens.inputs["Shade Smooth"].default_value = True

        # Get position for lensing calculation
        position = self.nodes.new("GeometryNodeInputPosition")

        # Calculate distance from lens center
        vector_length = self.nodes.new("GeometryNodeVectorMath")
        vector_length.operation = "LENGTH"

        # Create lensing distortion
        lens_distortion = self.nodes.new("GeometryNodeSetPosition")

        # Add noise for realistic lensing
        noise = self.nodes.new("GeometryNodeTexNoise")
        noise.inputs["Scale"].default_value = 3.0

        # Scale distortion based on lens strength
        scale_distortion = self.nodes.new("GeometryNodeVectorMath")
        scale_distortion.operation = "MULTIPLY"

        # Add distortion to position
        add_distortion = self.nodes.new("GeometryNodeVectorMath")
        add_distortion.operation = "ADD"

        # Position nodes
        input_node.location = (-1200, 0)
        lens_sphere.location = (-1000, 0)
        set_smooth_lens.location = (-900, 0)
        position.location = (-1000, -200)
        vector_length.location = (-800, -200)
        noise.location = (-800, -400)
        scale_distortion.location = (-600, -200)
        add_distortion.location = (-400, -200)
        lens_distortion.location = (-200, 0)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Lens Radius"], lens_sphere.inputs["Radius"])
        self.links.new(lens_sphere.outputs["Mesh"], set_smooth_lens.inputs["Geometry"])
        self.links.new(
            set_smooth_lens.outputs["Geometry"], lens_distortion.inputs["Geometry"]
        )

        self.links.new(position.outputs["Position"], vector_length.inputs[0])
        self.links.new(vector_length.outputs["Value"], scale_distortion.inputs[0])
        self.links.new(input_node.outputs["Lens Strength"], scale_distortion.inputs[1])
        self.links.new(noise.outputs["Fac"], scale_distortion.inputs[2])
        self.links.new(position.outputs["Position"], add_distortion.inputs[0])
        self.links.new(scale_distortion.outputs["Vector"], add_distortion.inputs[1])
        self.links.new(
            add_distortion.outputs["Vector"], lens_distortion.inputs["Offset"]
        )

        self.links.new(
            lens_distortion.outputs["Geometry"], output_node.inputs["Geometry"]
        )


# Lens presets
LENS_PRESETS = {
    "weak_lens": {
        "Lens Strength": 0.2,
        "Lens Radius": 3.0,
        "Distortion Type": "spherical",
        "Focus Distance": 8.0,
    },
    "strong_lens": {
        "Lens Strength": 0.8,
        "Lens Radius": 8.0,
        "Distortion Type": "elliptical",
        "Focus Distance": 15.0,
    },
    "galaxy_cluster_lens": {
        "Lens Strength": 1.0,
        "Lens Radius": 12.0,
        "Distortion Type": "complex",
        "Focus Distance": 20.0,
    },
}


def get_lens_preset(preset_name: str) -> dict:
    """Get preset for lens configuration."""
    return LENS_PRESETS.get(preset_name, LENS_PRESETS["weak_lens"])


def apply_lens_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply lens preset to node group."""
    preset = get_lens_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    bpy.utils.register_class(AlbpyLensGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyLensGeometryGroup)
