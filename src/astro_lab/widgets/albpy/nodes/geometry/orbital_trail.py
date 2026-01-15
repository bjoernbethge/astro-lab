import bpy


class AlbpyOrbitalTrailGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_OrbitalTrail"
    bl_label = "Albpy Orbital Trail Geometry Group"

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
            name="Trail Length", in_out="INPUT", socket_type="NodeSocketInt"
        )
        self.interface.new_socket(
            name="Trail Width", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Trail Opacity", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Trail Color", in_out="INPUT", socket_type="NodeSocketColor"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 100  # Trail Length
        self.interface.items_tree["Input_3"].default_value = 0.02  # Trail Width
        self.interface.items_tree["Input_4"].default_value = 0.5  # Trail Opacity
        self.interface.items_tree["Input_5"].default_value = (
            0.8,
            0.9,
            1.0,
            1.0,
        )  # Trail Color

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create trail curve
        curve = self.nodes.new("GeometryNodeCurvePrimitiveBezierSegment")
        curve.mode = "POINTS"

        # Instance on points for trail effect
        instance_on_points = self.nodes.new("GeometryNodeInstanceOnPoints")

        # Create small sphere for trail particles
        trail_sphere = self.nodes.new("GeometryNodeMeshUVSphere")
        trail_sphere.inputs["Radius"].default_value = 0.01
        # Smooth shading für Trail-Sphären
        set_smooth_trail = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_trail.inputs["Shade Smooth"].default_value = True

        # Scale instances
        scale_instances = self.nodes.new("GeometryNodeScaleInstances")

        # Set material
        set_material = self.nodes.new("GeometryNodeSetMaterial")

        # Create trail material
        trail_material = self.nodes.new("GeometryNodeMaterial")
        trail_material.material = self._create_trail_material()

        # Position nodes
        input_node.location = (-1000, 0)
        curve.location = (-800, 0)
        instance_on_points.location = (-600, 0)
        trail_sphere.location = (-800, 200)
        set_smooth_trail.location = (-700, 200)
        scale_instances.location = (-400, 0)
        set_material.location = (-200, 0)
        trail_material.location = (-400, 200)
        output_node.location = (0, 0)

        # Connect nodes
        self.links.new(curve.outputs["Curve"], instance_on_points.inputs["Points"])
        self.links.new(
            trail_sphere.outputs["Mesh"], set_smooth_trail.inputs["Geometry"]
        )
        self.links.new(
            set_smooth_trail.outputs["Geometry"], instance_on_points.inputs["Instance"]
        )
        self.links.new(
            instance_on_points.outputs["Instances"], scale_instances.inputs["Instances"]
        )
        self.links.new(
            scale_instances.outputs["Instances"], set_material.inputs["Geometry"]
        )
        self.links.new(
            trail_material.outputs["Material"], set_material.inputs["Material"]
        )
        self.links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])

        # Connect scaling
        self.links.new(
            input_node.outputs["Trail Width"], scale_instances.inputs["Scale"]
        )

    def _create_trail_material(self):
        """Create trail material for the node group."""
        mat = bpy.data.materials.new(name="OrbitalTrail")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Emission shader for glowing trail
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (0.8, 0.9, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 2.0

        # Output
        output = nodes.new("ShaderNodeOutputMaterial")

        # Connect
        links.new(emission.outputs["Emission"], output.inputs["Surface"])

        return mat


# Orbital trail presets
ORBITAL_TRAIL_PRESETS = {
    "planet_trail": {
        "Trail Length": 50,
        "Trail Width": 0.01,
        "Trail Opacity": 0.3,
        "Trail Color": (0.2, 0.5, 0.8, 1.0),
    },
    "comet_trail": {
        "Trail Length": 200,
        "Trail Width": 0.05,
        "Trail Opacity": 0.8,
        "Trail Color": (1.0, 0.9, 0.7, 1.0),
    },
    "satellite_trail": {
        "Trail Length": 30,
        "Trail Width": 0.005,
        "Trail Opacity": 0.6,
        "Trail Color": (0.8, 0.8, 0.8, 1.0),
    },
    "star_trail": {
        "Trail Length": 100,
        "Trail Width": 0.02,
        "Trail Opacity": 0.4,
        "Trail Color": (1.0, 0.8, 0.6, 1.0),
    },
}


def get_orbital_trail_preset(preset_name: str) -> dict:
    """Get preset for orbital trail configuration."""
    return ORBITAL_TRAIL_PRESETS.get(preset_name, ORBITAL_TRAIL_PRESETS["planet_trail"])


def apply_orbital_trail_preset(
    node_group: bpy.types.GeometryNodeGroup, preset_name: str
) -> None:
    """Apply orbital trail preset to node group."""
    preset = get_orbital_trail_preset(preset_name)

    # Apply preset parameters
    for param_name, value in preset.items():
        if param_name in node_group.inputs:
            node_group.inputs[param_name].default_value = value


def register():
    # REMOVED: bpy.utils.register_class(AlbpyOrbitalTrailGeometryGroup)

    bpy.utils.register_class(AlbpyOrbitalTrailGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyOrbitalTrailGeometryGroup)
