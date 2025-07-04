import bpy
from bpy.types import Operator


class ALBPY_OT_CreateFilament(Operator):
    bl_idname = "albpy.create_filament"
    bl_label = "Create Filament"
    bl_description = "Create cosmic filaments using Node Groups"

    count: bpy.props.IntProperty(name="Filament Count", default=8, min=1, max=100)
    length: bpy.props.FloatProperty(
        name="Filament Length", default=18.0, min=1.0, max=100.0
    )
    width: bpy.props.FloatProperty(
        name="Filament Width", default=0.15, min=0.01, max=10.0
    )
    curvature: bpy.props.FloatProperty(name="Curvature", default=0.3, min=0.0, max=2.0)

    def execute(self, context):
        # Erzeuge eine neue Geometry Node Group Instanz für Filament
        ng_name = "ALBPY_NG_Filament"
        if ng_name not in bpy.data.node_groups:
            self.report({"ERROR"}, f"Node group '{ng_name}' not found.")
            return {"CANCELLED"}
        node_group = bpy.data.node_groups[ng_name]
        # Neues leeres Objekt mit Geometry Nodes Modifier
        obj = bpy.data.objects.new("Filament", None)
        context.scene.collection.objects.link(obj)
        mod = obj.modifiers.new("FilamentNodes", "NODES")
        mod.node_group = node_group
        # Setze Inputs
        try:
            mod["Input_2"] = self.count
            mod["Input_3"] = self.length
            mod["Input_4"] = self.width
            mod["Input_5"] = self.curvature
        except Exception:
            pass
        obj.location = (0, 0, 0)
        self.report(
            {"INFO"}, f"Filament created with count={self.count}, length={self.length}"
        )
        return {"FINISHED"}


def register():
    bpy.utils.register_class(ALBPY_OT_CreateFilament)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreateFilament)
