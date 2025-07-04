import bpy
from bpy.types import Operator


class ALBPY_OT_CreatePointCloud(Operator):
    bl_idname = "albpy.create_point_cloud"
    bl_label = "Create Point Cloud"
    bl_description = "Create a 3D point cloud using Node Groups"

    def execute(self, context):
        ng_name = "ALBPY_NG_Pointcloud"
        if ng_name not in bpy.data.node_groups:
            self.report({"ERROR"}, f"Node group '{ng_name}' not found.")
            return {"CANCELLED"}
        node_group = bpy.data.node_groups[ng_name]
        obj = bpy.data.objects.new("PointCloud", None)
        context.scene.collection.objects.link(obj)
        mod = obj.modifiers.new("PointCloudNodes", "NODES")
        mod.node_group = node_group
        # Setze Inputs (Preset: star_field)
        mod["Input_2"] = 10000  # Point Count
        mod["Input_3"] = 0.005  # Point Size
        mod["Input_4"] = 20.0  # Distribution Radius
        mod["Input_5"] = 1.5  # Density Falloff
        obj.location = (0, 0, 0)
        self.report({"INFO"}, "Pointcloud created.")
        return {"FINISHED"}


def register():
    # REMOVED: bpy.utils.register_class(ALBPY_OT_CreatePointCloud)

    bpy.utils.register_class(ALBPY_OT_CreatePointCloud)


def unregister():
    bpy.utils.unregister_class(ALBPY_OT_CreatePointCloud)
