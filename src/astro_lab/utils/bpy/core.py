"""
Core Blender utilities for astronomical data visualization.

This module consolidates all Blender functionality into a DRY, well-organized system
that handles plotting, animation, materials, lighting, cameras, and rendering.

with NumPy 2.x compatibility and robust error handling.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportGeneralTypeIssues=false

import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# Comprehensive NumPy warnings suppression
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*numpy.core.multiarray.*")
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*cannot be run in NumPy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# NumPy import
# Import bpy directly since it's available
import bpy
import numpy as np
import polars as pl
from mathutils import Vector

# =============================================================================
# SCENE MANAGEMENT
# =============================================================================


def reset_scene() -> None:
    """Reset Blender scene to clean state."""
    if bpy is None:
        return

    try:
        # Delete all objects
        bpy.ops.object.select_all(action="SELECT")  # type: ignore
        bpy.ops.object.delete(use_global=False)  # type: ignore

        # Clear materials, lights, etc.
        for collection in [bpy.data.materials, bpy.data.lights, bpy.data.curves]:  # type: ignore
            for item in collection:
                collection.remove(item)
    except Exception as e:
        warnings.warn(f"Failed to reset scene: {e}", UserWarning)


def normalize_scene(
    target_scale: float = 5.0, center: bool = True
) -> Tuple[float, Tuple[float, float, float]]:
    """Normalize scene objects to target scale."""
    if bpy is None:
        return 1.0, (0.0, 0.0, 0.0)

    try:
        # Get scene bounds
        min_coord = np.array([float("inf")] * 3)
        max_coord = np.array([float("-inf")] * 3)

        for obj in bpy.context.scene.objects:  # type: ignore
            if obj.type == "MESH" and hasattr(obj.data, "vertices"):
                mesh_data = obj.data
                for vertex in mesh_data.vertices:
                    world_coord = obj.matrix_world @ vertex.co
                    min_coord = np.minimum(min_coord, world_coord)
                    max_coord = np.maximum(max_coord, world_coord)

        # Calculate scale and offset
        size = max_coord - min_coord
        max_size = np.max(size)
        center_point = (min_coord + max_coord) / 2
        offset = -center_point if center else np.zeros(3)

        scale = target_scale / max_size if max_size > 0 else 1.0

        # Apply transformations
        for obj in bpy.context.scene.objects:  # type: ignore
            if obj.type == "MESH":
                obj.location = obj.location + Vector(offset.tolist())
                obj.scale *= scale

        bpy.context.view_layer.update()  # type: ignore
        return scale, tuple(offset)

    except Exception as e:
        warnings.warn(f"Failed to normalize scene: {e}", UserWarning)
        return 1.0, (0.0, 0.0, 0.0)


def setup_scene(
    name: str = "AstroScene",
    world_color: List[float] = [0.05, 0.05, 0.1],
    units: str = "METRIC",
) -> bool:
    """Setup scene with astronomical defaults."""
    if bpy is None:
        return False

    try:
        scene = bpy.context.scene  # type: ignore
        scene.name = name
        scene.unit_settings.system = units

        # Setup world background
        world = bpy.data.worlds.new(name=f"{name}_World")  # type: ignore
        world.use_nodes = True
        world.node_tree.nodes.clear()

        # Background shader
        bg_node = world.node_tree.nodes.new("ShaderNodeBackground")
        bg_node.inputs["Color"].default_value = (*world_color, 1.0)
        bg_node.inputs["Strength"].default_value = 0.1

        # Output
        output_node = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(
            bg_node.outputs["Background"], output_node.inputs["Surface"]
        )

        scene.world = world
        return True

    except Exception as e:
        warnings.warn(f"Failed to setup scene: {e}", UserWarning)
        return False


# =============================================================================
# MATERIALS
# =============================================================================


def create_material(  # type: ignore
    name: str,
    material_type: str = "emission",
    base_color: List[float] = [0.8, 0.8, 0.8],
    emission_strength: float = 2.0,
    alpha: float = 1.0,
    **kwargs,
) -> Optional[Any]:
    """Create material with unified API."""
    if bpy is None:
        return None

    try:
        mat = bpy.data.materials.new(name=name)  # type: ignore
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Output node
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (300, 0)

        if material_type == "emission":
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (*base_color, 1.0)
            emission.inputs["Strength"].default_value = emission_strength
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

            if alpha < 1.0:
                mat.blend_method = "BLEND"
                emission.inputs["Color"].default_value = (*base_color, alpha)

        elif material_type == "principled":
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
            bsdf.inputs["Metallic"].default_value = kwargs.get("metallic", 0.0)
            bsdf.inputs["Roughness"].default_value = kwargs.get("roughness", 0.5)
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        elif material_type == "star":
            # Temperature-based star material
            temp = kwargs.get("temperature", 5778.0)
            color = _temperature_to_color(temp)
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (*color, 1.0)
            emission.inputs["Strength"].default_value = kwargs.get("brightness", 5.0)
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        return mat

    except Exception as e:
        warnings.warn(f"Failed to create material: {e}", UserWarning)
        return None


def _temperature_to_color(temperature: float) -> Tuple[float, float, float]:
    """Convert temperature to RGB color using black-body radiation approximation."""
    # Simplified temperature to color conversion
    temp = max(1000, min(temperature, 40000))  # Clamp to reasonable range

    if temp < 3700:
        red = 1.0
        green = (temp - 1000) / 2700
        blue = 0.0
    elif temp < 5500:
        red = 1.0
        green = 0.8 + 0.2 * (temp - 3700) / 1800
        blue = (temp - 3700) / 1800
    else:
        red = 1.0 - 0.3 * (temp - 5500) / 34500
        green = 0.9 + 0.1 * (temp - 5500) / 34500
        blue = 1.0

    return (max(0, min(1, red)), max(0, min(1, green)), max(0, min(1, blue)))


def create_cosmic_grid(
    size: float = 50.0, divisions: int = 10, color: List[float] = [0.1, 0.1, 0.1]
) -> Optional[Any]:
    """Create a 3D cosmic grid."""
    if bpy is None:
        return None
    try:
        # Create a new mesh and object
        mesh = bpy.data.meshes.new("CosmicGridMesh")  # type: ignore
        obj = bpy.data.objects.new("CosmicGrid", mesh)  # type: ignore

        # Create vertices and edges
        verts = []
        edges = []
        step = size / divisions

        for i in range(divisions + 1):
            offset = i * step - size / 2
            # Lines along X
            verts.append((offset, -size / 2, 0))
            verts.append((offset, size / 2, 0))
            edges.append((len(verts) - 2, len(verts) - 1))
            # Lines along Y
            verts.append((-size / 2, offset, 0))
            verts.append((size / 2, offset, 0))
            edges.append((len(verts) - 2, len(verts) - 1))

        mesh.from_pydata(verts, edges, [])
        mesh.update()

        # Link object to scene
        bpy.context.collection.objects.link(obj)  # type: ignore

        # Create a simple material for the grid
        mat = create_material(
            name="GridMat",
            material_type="emission",
            base_color=color,
            emission_strength=0.5,
        )
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        return obj
    except Exception as e:
        warnings.warn(f"Failed to create cosmic grid: {e}", UserWarning)
        return None


def create_text_legend(
    items: Dict[str, List[float]],
    position: Tuple[float, float, float] = (-8, 5, 0),
    font_size: float = 0.5,
) -> List[Any]:
    """Create a text-based legend in the 3D scene."""
    if bpy is None:
        return []

    legend_objects = []
    try:
        for i, (name, color) in enumerate(items.items()):
            # Create text object
            bpy.ops.object.text_add(  # type: ignore
                location=(position[0], position[1] - i * (font_size * 1.5), position[2])
            )
            text_obj = bpy.context.object  # type: ignore
            text_obj.data.body = name
            text_obj.data.size = font_size
            text_obj.name = f"Legend_{name}"

            # Create a material for the text
            mat = create_material(
                name=f"LegendMat_{name}",
                material_type="emission",
                base_color=color[:3],
                emission_strength=1.0,
            )
            if text_obj.data.materials:
                text_obj.data.materials[0] = mat
            else:
                text_obj.data.materials.append(mat)

            legend_objects.append(text_obj)
        return legend_objects
    except Exception as e:
        warnings.warn(f"Failed to create text legend: {e}", UserWarning)
        return legend_objects


# =============================================================================
# LIGHTING
# =============================================================================


def create_light(
    light_type: str,
    position: List[float] = [0, 0, 5],
    power: float = 1000.0,
    color: List[float] = [1.0, 1.0, 1.0],
    name: str = "AstroLight",
    **kwargs,
) -> Optional[Any]:
    """Create light with unified API."""
    if bpy is None:
        return None

    try:
        bpy.ops.object.light_add(type=light_type.upper(), location=position)  # type: ignore
        light = bpy.context.active_object  # type: ignore
        light.name = name

        # Type-safe light data access
        if hasattr(light.data, "energy"):
            light.data.energy = power
        if hasattr(light.data, "color"):
            light.data.color = color[:3]

        # Additional properties
        if light_type.upper() == "AREA" and hasattr(light.data, "size"):
            light.data.size = kwargs.get("size", 1.0)
        elif light_type.upper() == "SPOT" and hasattr(light.data, "spot_size"):
            light.data.spot_size = kwargs.get("spot_size", math.radians(45))

        return light

    except Exception as e:
        warnings.warn(f"Failed to create light: {e}", UserWarning)
        return None


def setup_lighting_preset(
    preset: str = "three_point",
    intensity: float = 1000.0,
    color_temp: float = 6500.0,
) -> List[Any]:
    """Setup lighting presets for astronomical visualization."""
    if bpy is None:
        return []

    lights = []
    try:
        if preset == "three_point":
            # Key light
            key_light = create_light(
                "SUN", [5, -5, 8], intensity * 1.5, name="KeyLight"
            )
            if key_light:
                lights.append(key_light)

            # Fill light
            fill_light = create_light(
                "SUN", [-3, 3, 5], intensity * 0.5, name="FillLight"
            )
            if fill_light:
                lights.append(fill_light)

            # Rim light
            rim_light = create_light("SUN", [0, 8, 2], intensity * 0.3, name="RimLight")
            if rim_light:
                lights.append(rim_light)

        elif preset == "astronomical":
            # Simulate distant star illumination
            star_light = create_light(
                "SUN", [10, 10, 10], intensity * 0.1, name="StarLight"
            )
            if star_light:
                lights.append(star_light)

            # Ambient cosmic background
            ambient_light = create_light(
                "SUN", [0, 0, 20], intensity * 0.05, name="CosmicBackground"
            )
            if ambient_light:
                lights.append(ambient_light)

    except Exception as e:
        warnings.warn(f"Failed to setup lighting preset: {e}", UserWarning)

    return lights


# =============================================================================
# CAMERA
# =============================================================================


def create_camera(
    position: List[float] = [5, -5, 3],
    target: List[float] = [0, 0, 0],
    fov: float = 35.0,
    name: str = "AstroCamera",
) -> Optional[Any]:
    """Create camera with unified API."""
    if bpy is None:
        return None

    try:
        bpy.ops.object.camera_add(location=position)  # type: ignore
        camera = bpy.context.active_object  # type: ignore
        camera.name = name

        # Point camera at target
        direction = Vector(target) - Vector(position)
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        # Set properties
        camera.data.lens_unit = "FOV"
        camera.data.angle = math.radians(fov)
        camera.data.clip_start = 0.01
        camera.data.clip_end = 1000.0

        # Set as active camera
        bpy.context.scene.camera = camera  # type: ignore

        return camera

    except Exception as e:
        warnings.warn(f"Failed to create camera: {e}", UserWarning)
        return None


def animate_camera(
    camera: Any,
    keyframes: List[Dict[str, Any]],
    frame_start: int = 1,
    frame_end: int = 250,
) -> bool:
    """Animate camera with keyframes."""
    if bpy is None or not camera:
        return False

    try:
        scene = bpy.context.scene  # type: ignore
        scene.frame_start = frame_start
        scene.frame_end = frame_end

        for keyframe in keyframes:
            frame = keyframe.get("frame", 1)
            location = keyframe.get("location", [0, 0, 0])
            rotation = keyframe.get("rotation", [0, 0, 0])

            scene.frame_set(frame)
            camera.location = location
            camera.rotation_euler = rotation

            camera.keyframe_insert(data_path="location")
            camera.keyframe_insert(data_path="rotation_euler")

        return True

    except Exception as e:
        warnings.warn(f"Failed to animate camera: {e}", UserWarning)
        return False


def create_camera_path(
    points: List[List[float]],
    name: str = "CameraPath",
    smooth: bool = True,
) -> Optional[Any]:
    """Create camera path using curve."""
    if bpy is None:
        return None

    try:
        # Create curve
        curve_data = bpy.data.curves.new(name=name, type="CURVE")  # type: ignore
        curve_data.dimensions = "3D"

        # Create spline
        spline = curve_data.splines.new("NURBS")
        spline.points.add(len(points) - 1)

        for i, point in enumerate(points):
            spline.points[i].co = (*point, 1.0)

        if smooth:
            spline.use_smooth = True

        # Create object
        curve_obj = bpy.data.objects.new(name, curve_data)  # type: ignore
        bpy.context.collection.objects.link(curve_obj)  # type: ignore

        return curve_obj

    except Exception as e:
        warnings.warn(f"Failed to create camera path: {e}", UserWarning)
        return None


# =============================================================================
# ASTRONOMICAL OBJECTS
# =============================================================================


def create_astro_object(
    object_type: str,
    position: List[float] = [0, 0, 0],
    scale: float = 1.0,
    properties: Optional[Dict[str, Any]] = None,
    name: str = "AstroObject",
) -> Optional[Any]:
    """Create astronomical object with properties."""
    if bpy is None:
        return None

    obj = None
    try:
        if object_type == "star":
            bpy.ops.mesh.primitive_ico_sphere_add(location=position, radius=scale)  # type: ignore
            obj = bpy.context.active_object  # type: ignore
            obj.name = f"{name}_Star"

            # Create star material
            if properties:
                temp = properties.get("temperature", 5778)
                brightness = properties.get("magnitude", 0.0)
                mat = create_material(
                    f"{name}_StarMat",
                    "star",
                    temperature=temp,
                    brightness=10 ** (-0.4 * brightness),
                )
                if mat:
                    obj.data.materials.append(mat)

        elif object_type == "galaxy":
            bpy.ops.mesh.primitive_uv_sphere_add(location=position, radius=scale)  # type: ignore
            obj = bpy.context.active_object  # type: ignore
            obj.name = f"{name}_Galaxy"

            # Create galaxy material
            mat = create_material(
                f"{name}_GalaxyMat",
                "emission",
                base_color=[0.8, 0.6, 0.4],
                emission_strength=0.5,
            )
            if mat:
                obj.data.materials.append(mat)

        elif object_type == "nebula":
            bpy.ops.mesh.primitive_cube_add(location=position, size=scale)  # type: ignore
            obj = bpy.context.active_object  # type: ignore
            obj.name = f"{name}_Nebula"

            # Create nebula material with transparency
            mat = create_material(
                f"{name}_NebulaMat",
                "emission",
                base_color=[0.4, 0.8, 1.0],
                emission_strength=0.3,
                alpha=0.3,
            )
            if mat:
                obj.data.materials.append(mat)

        # Add custom properties
        if obj and properties:
            for key, value in properties.items():
                obj[key] = value

        return obj

    except Exception as e:
        warnings.warn(f"Failed to create astro object: {e}", UserWarning)
        return None


def setup_astronomical_scene(
    data: Union[pl.DataFrame, List[Dict[str, Any]]],
    object_type: str = "star",
    position_cols: List[str] = ["x", "y", "z"],
    scale_col: Optional[str] = None,
    color_col: Optional[str] = None,
) -> List[Any]:
    """Setup scene with astronomical data."""
    if bpy is None:
        return []

    objects = []
    try:
        # Convert to list of dicts if DataFrame
        if hasattr(data, "to_dicts"):
            data_list = data.to_dicts()  # type: ignore
        else:
            data_list = data

        for i, row in enumerate(data_list):
            position = [row.get(col, 0.0) for col in position_cols]
            scale = row.get(scale_col, 1.0) if scale_col else 1.0

            properties = {k: v for k, v in row.items() if k not in position_cols}

            obj = create_astro_object(
                object_type, position, scale, properties, f"{object_type}_{i:04d}"
            )

            if obj:
                objects.append(obj)

        return objects

    except Exception as e:
        warnings.warn(f"Failed to setup astronomical scene: {e}", UserWarning)
        return []


# =============================================================================
# RENDERING
# =============================================================================


def setup_render_settings(
    engine: str = "BLENDER_EEVEE_NEXT",
    resolution: Tuple[int, int] = (1920, 1080),
    samples: int = 64,
    output_path: str = "results/render.png",
) -> bool:
    """Setup render settings."""
    if bpy is None:
        return False

    try:
        scene = bpy.context.scene  # type: ignore
        render = scene.render

        # Engine settings
        scene.render.engine = engine.upper()

        # Resolution
        render.resolution_x, render.resolution_y = resolution
        render.resolution_percentage = 100

        # Sampling (for Cycles)
        if engine.upper() == "CYCLES":
            scene.cycles.samples = samples

        # Output - ensure absolute path
        import os

        if not os.path.isabs(output_path):
            output_path = str(Path.cwd() / output_path)

        # Ensure directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        render.filepath = output_path
        render.image_settings.file_format = Path(output_path).suffix[1:].upper()

        return True

    except Exception as e:
        warnings.warn(f"Failed to setup render settings: {e}", UserWarning)
        return False


def render_scene(
    output_path: str = "results/render.png", animation: bool = False
) -> bool:
    """Render scene or animation."""
    if bpy is None:
        return False

    try:
        bpy.context.scene.render.filepath = output_path  # type: ignore

        if animation:
            bpy.ops.render.render(animation=True)  # type: ignore
        else:
            bpy.ops.render.render(write_still=True)  # type: ignore

        return True

    except Exception as e:
        warnings.warn(f"Failed to render scene: {e}", UserWarning)
        return False


# =============================================================================
# MAIN PLOTTER CLASSES
# =============================================================================


class AstroPlotter:
    """Main astronomical data plotter using Blender."""

    def __init__(self, scene_name: str = "AstroScene"):
        """Initialize plotter."""
        self.scene_name = scene_name
        self.objects = []
        self.lights = []
        self.camera = None

        if bpy is not None:
            self.setup_scene()

    def setup_scene(self) -> bool:
        """Setup the Blender scene."""
        if bpy is None:
            return False

        try:
            reset_scene()
            setup_scene(self.scene_name)
            self.camera = create_camera()
            self.lights = setup_lighting_preset("astronomical")
            return True
        except Exception as e:
            warnings.warn(f"Failed to setup scene: {e}", UserWarning)
            return False

    def plot_data(
        self,
        data: Union[pl.DataFrame, List[Dict[str, Any]]],
        object_type: str = "star",
        **kwargs,
    ) -> List[Any]:
        """Plot astronomical data."""
        if bpy is None:
            return []

        try:
            objects = setup_astronomical_scene(data, object_type, **kwargs)
            self.objects.extend(objects)
            return objects
        except Exception as e:
            warnings.warn(f"Failed to plot data: {e}", UserWarning)
            return []

    def render(self, output_path: str = "results/astro_render.png") -> bool:
        """Render the scene."""
        if bpy is None:
            return False

        try:
            setup_render_settings(output_path=output_path)
            return render_scene(output_path)
        except Exception as e:
            warnings.warn(f"Failed to render: {e}", UserWarning)
            return False


class FuturisticAstroPlotter(AstroPlotter):
    """Futuristic-styled astronomical plotter."""

    def setup_scene(self) -> bool:
        """Setup futuristic scene."""
        if bpy is None:
            return False

        try:
            reset_scene()
            setup_scene(self.scene_name, world_color=[0.0, 0.05, 0.1])
            self.camera = create_camera(fov=45.0)
            self.lights = setup_lighting_preset("three_point", intensity=2000.0)
            return True
        except Exception as e:
            warnings.warn(f"Failed to setup futuristic scene: {e}", UserWarning)
            return False


class GeometryNodesVisualizer:
    """geometry nodes-based visualizer."""

    def __init__(self):
        """Initialize geometry nodes visualizer."""
        self.node_groups = []

    def create_procedural_galaxy(self, name: str = "ProceduralGalaxy") -> Optional[Any]:
        """Create procedural galaxy using geometry nodes."""
        if bpy is None:
            return None

        try:
            # This would require extensive geometry nodes setup
            # For now, return a placeholder
            bpy.ops.mesh.primitive_ico_sphere_add()  # type: ignore
            obj = bpy.context.active_object  # type: ignore
            obj.name = name
            return obj
        except Exception as e:
            warnings.warn(f"Failed to create procedural galaxy: {e}", UserWarning)
            return None


class GreasePencilPlotter:
    """Grease Pencil-based astronomical plotter."""

    def __init__(self):
        """Initialize Grease Pencil plotter."""
        self.gp_objects = []

    def create_constellation_lines(
        self, star_positions: List[List[float]], connections: List[Tuple[int, int]]
    ) -> Optional[Any]:
        """Create constellation lines using Grease Pencil."""
        if bpy is None:
            return None

        try:
            # Create grease pencil object
            bpy.ops.object.gpencil_add(location=(0, 0, 0))  # type: ignore
            gp_obj = bpy.context.active_object  # type: ignore
            gp_obj.name = "ConstellationLines"

            # Get grease pencil data
            gp_data = gp_obj.data

            # Create layer
            layer = gp_data.layers.new("Constellation")

            # Create frame
            frame = layer.frames.new(1)

            # Create strokes for connections
            for start_idx, end_idx in connections:
                if start_idx < len(star_positions) and end_idx < len(star_positions):
                    stroke = frame.strokes.new()
                    stroke.points.add(2)

                    # Set points
                    stroke.points[0].co = star_positions[start_idx]
                    stroke.points[1].co = star_positions[end_idx]

            return gp_obj

        except Exception as e:
            warnings.warn(f"Failed to create constellation lines: {e}", UserWarning)
            return None
