import bpy

from astro_lab.config import get_albpy_config


def setup_lighting(scene=None, config=None):
    """
    Remove all existing lights and add new ones according to the preset in config.
    Args:
        scene: bpy.types.Scene (default: bpy.context.scene)
        config: dict or None, overrides defaults from albpy config
    Returns:
        List of created light objects.
    """
    if scene is None:
        scene = bpy.context.scene
    if config is None:
        config = get_albpy_config()
    preset = config.get("lighting_preset", "space")

    # Remove all lights
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    lights = []
    if preset == "space":
        # One sun, weak energy
        sun = bpy.data.lights.new(name="Sun", type="SUN")
        sun.energy = 2.0
        sun_obj = bpy.data.objects.new(name="Sun", object_data=sun)
        scene.collection.objects.link(sun_obj)
        sun_obj.location = (10, -10, 10)
        lights.append(sun_obj)
    elif preset == "studio":
        # Sun + fill point
        sun = bpy.data.lights.new(name="Sun", type="SUN")
        sun.energy = 5.0
        sun_obj = bpy.data.objects.new(name="Sun", object_data=sun)
        scene.collection.objects.link(sun_obj)
        sun_obj.location = (10, -10, 10)
        lights.append(sun_obj)
        point = bpy.data.lights.new(name="Fill", type="POINT")
        point.energy = 1000
        point_obj = bpy.data.objects.new(name="Fill", object_data=point)
        scene.collection.objects.link(point_obj)
        point_obj.location = (-10, 10, 5)
        lights.append(point_obj)
    else:  # default
        sun = bpy.data.lights.new(name="Sun", type="SUN")
        sun.energy = 3.0
        sun_obj = bpy.data.objects.new(name="Sun", object_data=sun)
        scene.collection.objects.link(sun_obj)
        sun_obj.location = (10, -10, 10)
        lights.append(sun_obj)
    return lights
