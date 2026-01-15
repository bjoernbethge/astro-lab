import bpy

from astro_lab.config import get_albpy_config


def setup_scene(scene=None, config=None):
    """
    Prepare the Blender scene: set world background, units, and clean up objects.
    Args:
        scene: bpy.types.Scene (default: bpy.context.scene)
        config: dict or None, overrides defaults from albpy config
    """
    if scene is None:
        scene = bpy.context.scene
    if config is None:
        config = get_albpy_config()

    # Set units
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    # Set world background color
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg_color = config.get("background_color", (0, 0, 0, 1))
    # Set background color in nodes
    nodes = scene.world.node_tree.nodes
    bg_node = nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = bg_color

    # Remove all objects except cameras and lights
    for obj in list(scene.objects):
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
