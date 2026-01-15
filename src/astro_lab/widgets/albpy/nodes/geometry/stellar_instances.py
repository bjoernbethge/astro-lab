import bpy


class AlbpyStellarInstancesGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_StellarInstances"
    bl_label = "Albpy Stellar Instances Geometry Group"

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
            name="Star Size", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Spectral Class", in_out="INPUT", socket_type="NodeSocketString"
        )
        self.interface.new_socket(
            name="Temperature", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Luminosity", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 0.1  # Star Size
        self.interface.items_tree["Input_3"].default_value = "G"  # Spectral Class
        self.interface.items_tree["Input_4"].default_value = 5778.0  # Temperature
        self.interface.items_tree["Input_5"].default_value = 1.0  # Luminosity

        # Create nodes
        input_node = self.nodes.new("NodeGroupInput")
        output_node = self.nodes.new("NodeGroupOutput")

        # Create star geometry
        ico_sphere = self.nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.inputs["Subdivisions"].default_value = 2
        # Smooth shading für Sterne
        set_smooth_star = self.nodes.new("GeometryNodeSetShadeSmooth")
        set_smooth_star.inputs["Shade Smooth"].default_value = True

        # Instance on points
        instance = self.nodes.new("GeometryNodeInstanceOnPoints")

        # Random size variation based on luminosity
        random_size = self.nodes.new("FunctionNodeRandomValue")
        random_size.data_type = "FLOAT"
        random_size.inputs["Min"].default_value = 0.5
        random_size.inputs["Max"].default_value = 2.0

        # Color variation based on temperature
        blackbody = self.nodes.new("ShaderNodeBlackbody")

        # Set material properties
        set_material = self.nodes.new("GeometryNodeSetMaterial")

        # Create emission material
        emission_material = self.nodes.new("ShaderNodeEmission")

        # Position nodes
        input_node.location = (-600, 0)
        ico_sphere.location = (-400, -200)
        set_smooth_star.location = (-300, -200)
        random_size.location = (-400, 100)
        blackbody.location = (-200, 200)
        emission_material.location = (-200, 0)
        set_material.location = (0, 0)
        instance.location = (200, 0)
        output_node.location = (400, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Geometry"], instance.inputs["Points"])
        self.links.new(input_node.outputs["Star Size"], ico_sphere.inputs["Radius"])
        self.links.new(ico_sphere.outputs["Mesh"], set_smooth_star.inputs["Geometry"])
        self.links.new(set_smooth_star.outputs["Geometry"], instance.inputs["Instance"])
        self.links.new(random_size.outputs["Value"], instance.inputs["Scale"])
        self.links.new(
            input_node.outputs["Temperature"], blackbody.inputs["Temperature"]
        )
        self.links.new(blackbody.outputs["Color"], emission_material.inputs["Color"])
        self.links.new(
            input_node.outputs["Luminosity"], emission_material.inputs["Strength"]
        )
        self.links.new(
            emission_material.outputs["Emission"], set_material.inputs["Material"]
        )
        self.links.new(instance.outputs["Instances"], set_material.inputs["Geometry"])
        self.links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])


# Stellar instance presets
STELLAR_INSTANCE_PRESETS = {
    "solar_type": {
        "Star Size": 0.1,
        "Spectral Class": "G",
        "Temperature": 5778.0,
        "Luminosity": 1.0,
    },
    "blue_giant": {
        "Star Size": 0.2,
        "Spectral Class": "B",
        "Temperature": 20000.0,
        "Luminosity": 100.0,
    },
    "red_dwarf": {
        "Star Size": 0.05,
        "Spectral Class": "M",
        "Temperature": 3000.0,
        "Luminosity": 0.01,
    },
    "white_dwarf": {
        "Star Size": 0.03,
        "Spectral Class": "D",
        "Temperature": 10000.0,
        "Luminosity": 0.1,
    },
}


def get_stellar_instance_preset(preset_name: str) -> dict:
    """Get preset for stellar instance configuration."""
    return STELLAR_INSTANCE_PRESETS.get(
        preset_name, STELLAR_INSTANCE_PRESETS["solar_type"]
    )


def apply_stellar_instance_preset(
    object_obj: bpy.types.Object, preset_name: str
) -> None:
    """Apply stellar instance preset to object."""
    preset = get_stellar_instance_preset(preset_name)

    # Find geometry nodes modifier
    for modifier in object_obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group:
            if "ALBPY_NG_StellarInstances" in modifier.node_group.name:
                # Apply preset parameters
                for param_name, value in preset.items():
                    if param_name in modifier:
                        modifier[param_name] = value
                break


def register():
    # REMOVED: bpy.utils.register_class(AlbpyStellarInstancesGeometryGroup)

    bpy.utils.register_class(AlbpyStellarInstancesGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyStellarInstancesGeometryGroup)
