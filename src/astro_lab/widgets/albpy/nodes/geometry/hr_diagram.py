import bpy


class AlbpyHRDiagramGeometryGroup(bpy.types.GeometryNodeGroup):
    bl_idname = "ALBPY_NG_HRDiagram"
    bl_label = "Albpy HR Diagram Geometry Group"

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
            name="Scale Factor", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        self.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set default values
        self.interface.items_tree["Input_2"].default_value = 0.1  # Star Size
        self.interface.items_tree["Input_3"].default_value = 1.0  # Scale Factor

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

        # Random size variation
        random_size = self.nodes.new("FunctionNodeRandomValue")
        random_size.data_type = "FLOAT"
        random_size.inputs["Min"].default_value = 0.5
        random_size.inputs["Max"].default_value = 2.0

        # Position nodes
        input_node.location = (-400, 0)
        ico_sphere.location = (-200, -200)
        set_smooth_star.location = (-100, -200)
        random_size.location = (-200, -100)
        instance.location = (0, 0)
        output_node.location = (200, 0)

        # Connect nodes
        self.links.new(input_node.outputs["Geometry"], instance.inputs["Points"])
        self.links.new(input_node.outputs["Star Size"], ico_sphere.inputs["Radius"])
        self.links.new(ico_sphere.outputs["Mesh"], set_smooth_star.inputs["Geometry"])
        self.links.new(set_smooth_star.outputs["Geometry"], instance.inputs["Instance"])
        self.links.new(random_size.outputs["Value"], instance.inputs["Scale"])
        self.links.new(instance.outputs["Instances"], output_node.inputs["Geometry"])


def register():
    # REMOVED: bpy.utils.register_class(AlbpyHRDiagramGeometryGroup)

    bpy.utils.register_class(AlbpyHRDiagramGeometryGroup)


def unregister():
    bpy.utils.unregister_class(AlbpyHRDiagramGeometryGroup)
