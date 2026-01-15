import bpy
from bpy.types import Operator


class ALBPY_OT_CreateHRDiagram(Operator):
    bl_idname = "albpy.create_hr_diagram"
    bl_label = "Create HR Diagram"
    bl_description = "Create a 3D Hertzsprung-Russell diagram using Node Groups"

    def execute(self, context):
        ng_name = "ALBPY_NG_HRDiagram"
        if ng_name not in bpy.data.node_groups:
            self.report({"ERROR"}, f"Node group '{ng_name}' not found.")
            return {"CANCELLED"}
        node_group = bpy.data.node_groups[ng_name]
        obj = bpy.data.objects.new("HRDiagram", None)
        context.scene.collection.objects.link(obj)
        mod = obj.modifiers.new("HRDiagramNodes", "NODES")
        mod.node_group = node_group
        # Setze Inputs (Preset)
        mod["Input_2"] = 0.1  # Star Size
        mod["Input_3"] = 1.0  # Scale Factor
        obj.location = (0, 0, 0)
        self.report({"INFO"}, "HR Diagram created.")
        return {"FINISHED"}


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreateHRDiagram)

    bpy.utils.register_class(ALBPY_OT_CreateHRDiagram)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreateHRDiagram)
