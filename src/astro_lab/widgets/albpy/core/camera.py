import bpy
from mathutils import Vector

from astro_lab.config import get_albpy_config


def setup_camera(scene=None, config=None):
    """
    Ensure a camera exists, set its position, look_at, and focal length.
    Args:
        scene: bpy.types.Scene (default: bpy.context.scene)
        config: dict or None, overrides defaults from albpy config
    Returns:
        The camera object.
    """
    if scene is None:
        scene = bpy.context.scene
    if config is None:
        config = get_albpy_config()

    # Find or create camera
    camera = None
    for obj in scene.objects:
        if obj.type == "CAMERA":
            camera = obj
            break
    if camera is None:
        cam_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(camera)
    scene.camera = camera

    # Set camera position and look_at
    camera.location = config.get("camera_location", (0, 0, 10))
    look_at = Vector(config.get("camera_look_at", (0, 0, 0)))
    direction = look_at - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Set focal length
    camera.data.lens = config.get("camera_focal_length", 50)
    return camera
