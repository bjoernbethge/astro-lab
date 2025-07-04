import bpy


class AlbpySupernovaRemnantGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_SupernovaRemnant"
    bl_label = "Albpy Supernova Remnant Geometry Group"

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
            name="Remnant Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Shock Strength", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Voronoi Scale", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Filament Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 30.0  # Remnant Size
        self.interface.items_tree["Input_3"].default_value = 0.3  # Shock Strength
        self.interface.items_tree["Input_4"].default_value = 3.0  # Voronoi Scale
        self.interface.items_tree["Input_5"].default_value = 20  # Filament Count

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create base sphere
        sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        sphere.inputs["Segments"].default_value = 64
        sphere.inputs["Ring Count"].default_value = 32
        # Smooth shading für Sphäre
        set_smooth_sphere = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_sphere.inputs["Shade Smooth"].default_value = True

        # Add displacement for shock waves
        displace = self.nodes.new("GeometryNodeSetPosition")

        # Voronoi noise for shock structure
        voronoi = self.nodes.new("GeometryNodeTexVoronoi")
        voronoi.feature = "DISTANCE_TO_EDGE"

        # Scale voronoi
        scale_voronoi = self.nodes.new("GeometryNodeVectorMath")
        scale_voronoi.operation = "MULTIPLY"

        # Add displacement to position
        add_displacement = self.nodes.new("GeometryNodeVectorMath")
        add_displacement.operation = "ADD"

        # Position nodes
        input_node.location = (-1000, 0)
        sphere.location = (-800, 0)
        set_smooth_sphere.location = (-700, 0)
        voronoi.location = (-800, -200)
        scale_voronoi.location = (-600, -200)
        displace.location = (-400, 0)
        add_displacement.location = (-200, 0)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Remnant Size"], sphere.inputs["Radius"])
        self.links.new(sphere.outputs["Mesh"], set_smooth_sphere.inputs["Geometry"])
        self.links.new(
            set_smooth_sphere.outputs["Geometry"], displace.inputs["Geometry"]
        )
        self.links.new(input_node.outputs["Voronoi Scale"], voronoi.inputs["Scale"])
        self.links.new(voronoi.outputs["Distance"], scale_voronoi.inputs[0])
        self.links.new(input_node.outputs["Shock Strength"], scale_voronoi.inputs[1])
        self.links.new(displace.outputs["Position"], add_displacement.inputs[0])
        self.links.new(scale_voronoi.outputs["Vector"], add_displacement.inputs[1])
        self.links.new(add_displacement.outputs["Vector"], displace.inputs["Offset"])
        self.links.new(displace.outputs["Geometry"], output_node.inputs["Geometry"])


# Supernova remnant presets
SUPERNOVA_REMNANT_PRESETS = {
    "young_remnant": {
        "Remnant Size": 20.0,
        "Shock Strength": 0.5,
        "Voronoi Scale": 2.0,
        "Filament Count": 30,
    },
    "mature_remnant": {
        "Remnant Size": 40.0,
        "Shock Strength": 0.2,
        "Voronoi Scale": 4.0,
        "Filament Count": 15,
    },
    "crab_like": {
        "Remnant Size": 25.0,
        "Shock Strength": 0.4,
        "Voronoi Scale": 3.0,
        "Filament Count": 25,
    },
    "giant_remnant": {
        "Remnant Size": 60.0,
        "Shock Strength": 0.1,
        "Voronoi Scale": 6.0,
        "Filament Count": 10,
    },
}


def get_supernova_remnant_preset(preset_name: str) -> dict:
    """Get preset for supernova remnant configuration."""
    return SUPERNOVA_REMNANT_PRESETS.get(
        preset_name, SUPERNOVA_REMNANT_PRESETS["young_remnant"]
    )


def apply_supernova_remnant_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply supernova remnant preset to node group."""
    preset = get_supernova_remnant_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpySupernovaRemnantGeometryGroup)

    bpy.utils.register_class(AlbpySupernovaRemnantGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpySupernovaRemnantGeometryGroup)
