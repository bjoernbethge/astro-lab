import os
from pathlib import Path
from typing import Optional, Tuple

import bpy

from astro_lab.config import find_project_root, get_albpy_config


def setup_rendering(scene=None, config=None):
    """
    Configure rendering: engine, resolution, samples, output path, color management.
    Only set EEVEE Next properties that are valid for the current bpy version.
    Args:
        scene: bpy.types.Scene (default: bpy.context.scene)
        config: dict or None, overrides defaults from albpy config
    """
    if scene is None:
        scene = bpy.context.scene
    if config is None:
        config = get_albpy_config()

    scene.render.engine = config.get("engine", "BLENDER_EEVEE_NEXT")
    res = config.get("resolution", (1920, 1080))
    scene.render.resolution_x = res[0]
    scene.render.resolution_y = res[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = config.get("file_format", "PNG")

    # Fix output path: always use absolute path and ensure directory exists
    output_path = config.get("output_path", "results/nsa_albpy_render.png")
    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    scene.render.filepath = abs_output_path

    # EEVEE Next: Only set properties that are valid for your bpy version!
    eevee = scene.eevee
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = config.get("samples", 64)
    if hasattr(eevee, "use_volumetric_lights"):
        eevee.use_volumetric_lights = True
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = True
    # Do not set use_bloom, use_gtao, etc. if not present!

    # Color management
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"  # For headless/batch


def render_astronomical_scene(
    output_path: Optional[str] = None,
    render_engine: Optional[str] = None,
    resolution: Optional[Tuple[int, int]] = None,
    samples: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Render astronomical scene with professional settings.

    Args:
        output_path: Output file path
        render_engine: Render engine to use
        resolution: Render resolution (width, height)
        samples: Number of samples
        **kwargs: Additional render parameters
    """
    render_config = get_albpy_config() or {}

    # Defensive: ensure render_config values are present
    default_output_path = render_config.get(
        "output_path", "results/nsa_albpy_render.png"
    )
    default_engine = render_config.get("engine", "BLENDER_EEVEE_NEXT")
    default_resolution = render_config.get("resolution", (1920, 1080))
    default_samples = render_config.get("samples", 128)

    output_path = output_path if output_path is not None else default_output_path

    # Always render into the results directory inside the project root
    results_dir = find_project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path_path = Path(str(output_path))

    # If only a filename is given, or the path is not inside results/, force it into results/
    if not output_path_path.is_absolute() or not str(output_path_path).startswith(
        str(results_dir)
    ):
        output_path_path = results_dir / output_path_path.name
    output_path_path = output_path_path.resolve()
    output_path = str(output_path_path)

    # Set render settings
    scene = bpy.context.scene
    scene.render.engine = render_engine if render_engine else default_engine
    scene.render.resolution_x = resolution[0] if resolution else default_resolution[0]
    scene.render.resolution_y = resolution[1] if resolution else default_resolution[1]
    scene.render.resolution_percentage = 100

    # Set samples based on engine
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples if samples else default_samples
    elif scene.render.engine == "BLENDER_EEVEE_NEXT":
        if hasattr(scene, "eevee_next"):
            scene.eevee_next.taa_render_samples = (
                samples if samples else default_samples
            )

    # Render
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered to: {output_path}")
