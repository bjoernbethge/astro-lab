"""
Core Blender utilities for astronomical data visualization.

This module consolidates all Blender functionality into a DRY, well-organized system
that handles plotting, animation, materials, lighting, cameras, and rendering.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None
    mathutils = None


# =============================================================================
# SCENE MANAGEMENT
# =============================================================================


def reset_scene() -> None:
    """Reset Blender scene to clean state."""
    if not BLENDER_AVAILABLE:
        return

    # Delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Clear materials, lights, etc.
    for collection in [bpy.data.materials, bpy.data.lights, bpy.data.curves]:
        for item in collection:
            collection.remove(item)


def normalize_scene(
    target_scale: float = 5.0, center: bool = True
) -> Tuple[float, Tuple[float, float, float]]:
    """Normalize scene objects to target scale."""
    if not BLENDER_AVAILABLE:
        return 1.0, (0.0, 0.0, 0.0)

    # Get scene bounds
    min_coord = np.array([float("inf")] * 3)
    max_coord = np.array([float("-inf")] * 3)

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and hasattr(obj.data, "vertices"):
            mesh_data = obj.data
            for vertex in mesh_data.vertices:
                world_coord = obj.matrix_world @ vertex.co
                min_coord = np.minimum(min_coord, world_coord)
                max_coord = np.maximum(max_coord, world_coord)

    # Calculate scale and offset
    size = max_coord - min_coord
    max_size = np.max(size)
    scale = target_scale / max_size if max_size > 0 else 1.0

    center_point = (min_coord + max_coord) / 2
    offset = -center_point if center else np.zeros(3)

    # Apply transformations
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.location = obj.location + mathutils.Vector(offset.tolist())
            obj.scale *= scale

    bpy.context.view_layer.update()
    return scale, tuple(offset)


def setup_scene(name: str = "AstroScene", engine: str = "BLENDER_EEVEE_NEXT") -> None:
    """Setup scene with proper settings."""
    if not BLENDER_AVAILABLE:
        return

    scene = bpy.context.scene
    scene.name = name
    scene.render.engine = engine
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = False

    # Enable advanced features for EEVEE Next
    if engine == "BLENDER_EEVEE_NEXT" and hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = True
        scene.eevee.bloom_intensity = 0.8
        scene.eevee.bloom_radius = 6.5

    # Color management
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "High Contrast"


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
    if not BLENDER_AVAILABLE:
        return None

    mat = bpy.data.materials.new(name=name)
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


def _temperature_to_color(temperature: float) -> List[float]:
    """Convert stellar temperature to RGB color."""
    if temperature < 3700:
        return [1.0, 0.4, 0.0]  # Red
    elif temperature < 5200:
        return [1.0, 0.7, 0.4]  # Orange
    elif temperature < 6000:
        return [1.0, 1.0, 0.8]  # Yellow-white
    elif temperature < 7500:
        return [1.0, 1.0, 1.0]  # White
    elif temperature < 10000:
        return [0.8, 0.9, 1.0]  # Blue-white
    else:
        return [0.6, 0.7, 1.0]  # Blue


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
    if not BLENDER_AVAILABLE:
        return None

    bpy.ops.object.light_add(type=light_type.upper(), location=position)
    light = bpy.context.active_object
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


def setup_lighting_preset(preset_name: str, **kwargs) -> List[Any]:
    """Setup lighting preset."""
    if not BLENDER_AVAILABLE:
        return []

    lights = []

    if preset_name == "three_point":
        # Key light
        key = create_light("SUN", [5, -5, 8], 3.0, [1.0, 0.9, 0.8], "KeyLight")
        # Fill light
        fill = create_light("SUN", [-3, 3, 5], 1.5, [0.8, 0.9, 1.0], "FillLight")
        # Rim light
        rim = create_light("SUN", [-5, -3, 2], 2.0, [1.0, 0.8, 0.6], "RimLight")
        lights = [key, fill, rim]

    elif preset_name == "deep_space":
        # Ambient space lighting
        ambient = create_light("SUN", [0, 0, 10], 0.5, [0.2, 0.3, 0.8], "SpaceAmbient")
        # Star field simulation
        star1 = create_light("POINT", [10, 10, 10], 500.0, [1.0, 1.0, 0.9], "Star1")
        star2 = create_light("POINT", [-8, 6, -5], 300.0, [0.9, 0.8, 1.0], "Star2")
        lights = [ambient, star1, star2]

    elif preset_name == "orbital":
        # Dramatic orbital lighting for futuristic interfaces
        key = create_light("SUN", [20, -20, 30], 3.0, [0.8, 0.9, 1.0], "OrbitalKey")
        rim = create_light("SUN", [-30, 20, 10], 2.0, [1.0, 0.6, 0.2], "OrbitalRim")
        ambient = create_light(
            "SUN", [0, 0, 50], 0.5, [0.2, 0.4, 0.8], "OrbitalAmbient"
        )
        lights = [key, rim, ambient]

    return [light for light in lights if light]


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
    if not BLENDER_AVAILABLE:
        return None

    bpy.ops.object.camera_add(location=position)
    camera = bpy.context.active_object
    camera.name = name

    # Point camera at target
    direction = mathutils.Vector(target) - mathutils.Vector(position)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Set properties
    camera.data.lens_unit = "FOV"
    camera.data.angle = math.radians(fov)
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0

    # Set as active camera
    bpy.context.scene.camera = camera

    return camera


def create_camera_path(
    path_type: str = "orbit",
    center: List[float] = [0, 0, 0],
    radius: float = 8.0,
    num_positions: int = 8,
    **kwargs,
) -> List[List[float]]:
    """Generate camera path for animation."""
    positions = []
    center_array = np.array(center)

    if path_type == "orbit":
        elevation = math.radians(kwargs.get("elevation", 30.0))
        start_angle = kwargs.get("start_angle", 0.0)

        for i in range(num_positions):
            azimuth = math.radians(start_angle + (360.0 * i / num_positions))
            x = radius * math.cos(elevation) * math.cos(azimuth)
            y = radius * math.cos(elevation) * math.sin(azimuth)
            z = radius * math.sin(elevation)
            position = center_array + np.array([x, y, z])
            positions.append(position.tolist())

    elif path_type == "flyby":
        direction = np.array(kwargs.get("direction", [1, 0, 0]))
        direction = direction / np.linalg.norm(direction)

        for i in range(num_positions):
            t = (i / (num_positions - 1)) * 2 - 1  # -1 to 1
            position = center_array + direction * t * radius * 2
            positions.append(position.tolist())

    return positions


def animate_camera(
    camera: Any,
    positions: List[List[float]],
    target: List[float] = [0, 0, 0],
    frame_duration: int = 30,
) -> None:
    """Animate camera along path."""
    if not BLENDER_AVAILABLE or not camera:
        return

    # Clear existing animation
    if camera.animation_data:
        camera.animation_data_clear()

    for i, position in enumerate(positions):
        frame = i * frame_duration + 1

        # Set position
        camera.location = position

        # Point at target
        direction = mathutils.Vector(target) - mathutils.Vector(position)
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        # Insert keyframes
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Set timeline
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(positions) * frame_duration


# =============================================================================
# ASTRONOMICAL OBJECTS
# =============================================================================


def create_astro_object(
    object_type: str,
    name: str,
    position: Tuple[float, float, float] = (0, 0, 0),
    scale: float = 1.0,
    material_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Optional[Any]:
    """Create astronomical object with unified API."""
    if not BLENDER_AVAILABLE:
        return None

    # Create base geometry
    if object_type in ["planet", "star", "exoplanet"]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=scale, location=position)
    elif object_type == "galaxy":
        bpy.ops.mesh.primitive_cylinder_add(
            radius=scale, depth=scale * 0.1, location=position
        )
    elif object_type in ["satellite", "asteroid"]:
        if object_type == "satellite":
            bpy.ops.mesh.primitive_cube_add(size=scale, location=position)
        else:
            bpy.ops.mesh.primitive_ico_sphere_add(radius=scale, location=position)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=scale, location=position)

    obj = bpy.context.active_object
    obj.name = name

    # Apply material
    if material_config:
        material = create_material(f"{name}_material", **material_config)
        if material:
            obj.data.materials.append(material)

    return obj


def setup_astronomical_scene(
    datasets: Dict[str, pl.DataFrame],
    scene_name: str = "AstroViz",
    max_objects_per_type: int = 100,
) -> Dict[str, List[Any]]:
    """Setup complete astronomical scene."""
    if not BLENDER_AVAILABLE:
        return {}

    reset_scene()
    setup_scene(scene_name)

    created_objects = {}

    for obj_type, data in datasets.items():
        if data.is_empty():
            continue

        objects = []
        limited_data = data.head(max_objects_per_type)

        for i, row in enumerate(limited_data.iter_rows(named=True)):
            # Extract position
            if all(col in row for col in ["x", "y", "z"]):
                position = (row["x"], row["y"], row["z"])
            else:
                position = (i * 2.0, 0, 0)

            # Extract scale
            scale = row.get("scale", row.get("radius", 0.1))

            # Create object
            obj_name = row.get("name", f"{obj_type}_{i}")
            material_config = _get_material_config(obj_type, row)

            obj = create_astro_object(
                obj_type, obj_name, position, scale, material_config
            )

            if obj:
                objects.append(obj)

        created_objects[obj_type] = objects

    # Setup lighting and camera
    setup_lighting_preset("deep_space")
    create_camera([15, -15, 10], [0, 0, 0])

    return created_objects


def _get_material_config(obj_type: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Get material configuration for object type."""
    if obj_type in ["star"]:
        return {
            "material_type": "star",
            "temperature": row.get("temperature", 5778.0),
            "brightness": row.get("brightness", 5.0),
        }
    elif obj_type in ["planet", "exoplanet"]:
        return {
            "material_type": "principled",
            "base_color": [0.2, 0.4, 0.8],
            "roughness": 0.7,
        }
    else:
        return {
            "material_type": "emission",
            "base_color": [0.8, 0.8, 0.8],
            "emission_strength": 2.0,
        }


# =============================================================================
# RENDERING
# =============================================================================


def setup_render_settings(
    engine: str = "BLENDER_EEVEE_NEXT",
    resolution: Tuple[int, int] = (1920, 1080),
    samples: int = 128,
) -> None:
    """Setup render settings."""
    if not BLENDER_AVAILABLE:
        return

    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]

    if engine == "CYCLES":
        scene.cycles.samples = samples
        # Try to use GPU if available
        try:
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            scene.cycles.device = "GPU"
        except:
            pass


def render_scene(output_path: str, animation: bool = False) -> bool:
    """Render scene to file."""
    if not BLENDER_AVAILABLE:
        return False

    # Convert to absolute path
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set render filepath
    output_path = output_path.replace("\\", "/")
    bpy.context.scene.render.filepath = output_path

    # Render
    if animation:
        bpy.ops.render.render(animation=True)
    else:
        bpy.ops.render.render(write_still=True)

    return True


# =============================================================================
# MAIN PLOTTING CLASSES
# =============================================================================


class AstroPlotter:
    """Main astronomical data plotter."""

    def __init__(self, scene_name: str = "AstroPlot"):
        self.scene_name = scene_name
        if BLENDER_AVAILABLE:
            setup_scene(scene_name)

    def scatter_3d(
        self,
        data: pl.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: Optional[str] = None,
        size_col: Optional[str] = None,
        name_col: Optional[str] = None,
        max_points: int = 1000,
    ) -> List[Any]:
        """Create 3D scatter plot."""
        if not BLENDER_AVAILABLE:
            return []

        reset_scene()
        setup_scene(self.scene_name)

        # Limit data
        plot_data = data.head(max_points)

        # Extract coordinates
        coords = plot_data.select([x_col, y_col, z_col]).to_numpy()
        coords = self._normalize_coordinates(coords)

        # Extract colors and sizes
        colors = self._get_colors(plot_data, color_col)
        sizes = self._get_sizes(plot_data, size_col)
        names = self._get_names(plot_data, name_col)

        objects = []
        for i, (coord, color, size, name) in enumerate(
            zip(coords, colors, sizes, names)
        ):
            # Create sphere
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=coord)
            obj = bpy.context.active_object
            obj.name = name

            # Create material
            material = create_material(
                f"{name}_material",
                material_type="emission",
                base_color=color,
                emission_strength=20.0,
            )

            if material:
                obj.data.materials.append(material)

            objects.append(obj)

        # Setup camera and lighting
        self._setup_plot_camera(coords)
        setup_lighting_preset("three_point")

        return objects

    def render_plot(self, output_path: str, animation: bool = False) -> bool:
        """Render the plot."""
        return render_scene(output_path, animation)

    def _normalize_coordinates(
        self, coords: np.ndarray, target_scale: float = 10.0
    ) -> np.ndarray:
        """Normalize coordinates to target scale."""
        min_vals = np.min(coords, axis=0)
        max_vals = np.max(coords, axis=0)
        ranges = max_vals - min_vals

        # Avoid division by zero
        ranges = np.where(ranges == 0, 1, ranges)

        # Normalize to [-target_scale/2, target_scale/2]
        normalized = (coords - min_vals) / ranges
        normalized = (normalized - 0.5) * target_scale

        return normalized

    def _get_colors(
        self, data: pl.DataFrame, color_col: Optional[str]
    ) -> List[List[float]]:
        """Get colors for data points."""
        if color_col and color_col in data.columns:
            values = data[color_col].to_numpy()
            # Normalize to [0, 1]
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
                # Map to color (blue to red)
                colors = [[1 - n, 0.0, n] for n in normalized]
            else:
                colors = [[0.8, 0.8, 0.8]] * len(values)
            return colors
        else:
            return [[0.8, 0.8, 0.8]] * len(data)

    def _get_sizes(
        self, data: pl.DataFrame, size_col: Optional[str], base_size: float = 0.3
    ) -> List[float]:
        """Get sizes for data points."""
        if size_col and size_col in data.columns:
            values = data[size_col].to_numpy()
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
                sizes = base_size + normalized * base_size * 4
            else:
                sizes = [base_size] * len(values)
            return list(sizes)
        else:
            return [base_size] * len(data)

    def _get_names(self, data: pl.DataFrame, name_col: Optional[str]) -> List[str]:
        """Get names for data points."""
        if name_col and name_col in data.columns:
            return data[name_col].to_list()
        else:
            return [f"Point_{i}" for i in range(len(data))]

    def _setup_plot_camera(self, coords: np.ndarray) -> None:
        """Setup camera for plot viewing."""
        # Calculate bounding box
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        center = (min_coords + max_coords) / 2
        size = np.max(max_coords - min_coords)

        # Position camera
        camera_distance = size * 2
        camera_pos = center + np.array(
            [camera_distance, -camera_distance, camera_distance]
        )

        create_camera(camera_pos.tolist(), center.tolist())


class FuturisticAstroPlotter:
    """Futuristic orbital interface plotter."""

    def __init__(self, scene_name: str = "FuturisticAstro"):
        self.scene_name = scene_name
        if BLENDER_AVAILABLE:
            setup_scene(scene_name, "BLENDER_EEVEE_NEXT")

    def create_orbital_interface(
        self,
        central_object: Dict[str, Any],
        orbital_data: pl.DataFrame,
        interface_scale: float = 10.0,
    ) -> Dict[str, List[Any]]:
        """Create futuristic orbital interface."""
        if not BLENDER_AVAILABLE:
            return {}

        reset_scene()
        setup_scene(self.scene_name, "BLENDER_EEVEE_NEXT")

        created_objects = {
            "central_body": [],
            "orbital_rings": [],
            "trajectories": [],
            "hud_elements": [],
        }

        # Create central body
        central_body = create_astro_object(
            "star",
            central_object.get("name", "CentralBody"),
            (0, 0, 0),
            central_object.get("radius", 1.0) * interface_scale * 0.1,
            {
                "material_type": "emission",
                "base_color": [0.2, 0.6, 1.0],
                "emission_strength": 15.0,
            },
        )
        created_objects["central_body"].append(central_body)

        # Create orbital rings
        ring_radii = [2.0, 4.0, 6.0, 8.0, 10.0]
        ring_colors = [
            [0.0, 0.8, 1.0],  # Cyan
            [0.2, 1.0, 0.8],  # Teal
            [0.8, 1.0, 0.2],  # Yellow-green
            [1.0, 0.6, 0.2],  # Orange
            [1.0, 0.2, 0.4],  # Pink
        ]

        for i, (radius, color) in enumerate(zip(ring_radii, ring_colors)):
            bpy.ops.mesh.primitive_torus_add(
                major_radius=radius * interface_scale,
                minor_radius=0.02,
                location=(0, 0, 0),
            )
            ring = bpy.context.active_object
            ring.name = f"OrbitalRing_{i}"

            material = create_material(
                f"ring_{i}_material",
                material_type="emission",
                base_color=color,
                emission_strength=8.0,
                alpha=0.3 - i * 0.05,
            )

            if material:
                ring.data.materials.append(material)

            created_objects["orbital_rings"].append(ring)

        # Setup dramatic lighting and camera
        setup_lighting_preset("orbital")
        create_camera(
            [interface_scale * 25, -interface_scale * 15, interface_scale * 12],
            [0, 0, 0],
            35.0,
        )

        return created_objects

    def render_orbital_interface(self, output_path: str) -> bool:
        """Render the orbital interface."""
        return render_scene(output_path)


class GeometryNodesVisualizer:
    """Geometry Nodes based procedural visualizer."""

    def __init__(self, name: str = "GeoNodesViz"):
        self.name = name

    def create_procedural_scatter_plot(
        self,
        data: np.ndarray,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        title: str = "Procedural Scatter",
        animated: bool = False,
    ) -> Optional[Any]:
        """Create procedural scatter plot using Geometry Nodes."""
        if not BLENDER_AVAILABLE:
            return None

        # Create base mesh with vertices at data points
        mesh = bpy.data.meshes.new(f"{title}_data")
        obj = bpy.data.objects.new(f"{title}_scatter", mesh)
        bpy.context.collection.objects.link(obj)

        # Create vertices
        vertices = [(float(x), float(y), float(z)) for x, y, z in data]
        mesh.from_pydata(vertices, [], [])
        mesh.update()

        # Add Geometry Nodes modifier (simplified)
        modifier = obj.modifiers.new("ProceduralViz", "NODES")

        return obj


class GreasePencilPlotter:
    """Grease Pencil based 2D/3D plotter."""

    def __init__(self, name: str = "AstroGP"):
        self.name = name

    def create_line_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        z_data: Optional[np.ndarray] = None,
        title: str = "Line Plot",
        style: str = "neon",
    ) -> Optional[Any]:
        """Create line plot with Grease Pencil v3."""
        if not BLENDER_AVAILABLE:
            return None

        try:
            # Method 1: Use Grease Pencil v3 API (Blender 4.4+)
            bpy.ops.object.grease_pencil_add(type="EMPTY")
            gp_obj = bpy.context.active_object
            gp_obj.name = f"{title}_Line"

            # Get grease pencil v3 data
            gp_data = gp_obj.data

            # Create layer (GPv3 API)
            layer = gp_data.layers.new("LineData")

            # Create frame (GPv3 API)
            frame = layer.frames.new(1)

            # Create drawing (GPv3 API)
            drawing = bpy.data.grease_pencils_v3.new(f"{title}_drawing")
            frame.drawing = drawing

            # Add stroke to drawing
            stroke = drawing.strokes.new()

            # Prepare coordinates
            if z_data is None:
                z_data = np.zeros_like(x_data)

            # Add points to stroke
            stroke.points.add(len(x_data))
            for i, (x, y, z) in enumerate(zip(x_data, y_data, z_data)):
                stroke.points[i].position = (x, y, z)
                stroke.points[i].radius = 0.02
                stroke.points[i].opacity = 1.0

            print(f"   ✅ GPv3 line plot created: {gp_obj.name}")
            return gp_obj

        except Exception as e:
            print(f"   ⚠️  GPv3 creation failed: {e}")
            # Fallback: Create curve instead
            try:
                curve_data = bpy.data.curves.new(f"{title}_curve", type="CURVE")
                curve_data.dimensions = "3D"
                curve_data.bevel_depth = 0.02

                # Create spline
                spline = curve_data.splines.new("NURBS")
                if z_data is None:
                    z_data = np.zeros_like(x_data)

                spline.points.add(len(x_data) - 1)
                for i, (x, y, z) in enumerate(zip(x_data, y_data, z_data)):
                    spline.points[i].co = (x, y, z, 1.0)

                # Create object
                curve_obj = bpy.data.objects.new(f"{title}_curve", curve_data)
                bpy.context.collection.objects.link(curve_obj)

                # Apply emission material
                mat = create_material(
                    f"{title}_mat",
                    material_type="emission",
                    base_color=[0.2, 0.8, 1.0] if style == "neon" else [1.0, 0.4, 0.2],
                    emission_strength=3.0,
                )
                if mat:
                    curve_obj.data.materials.append(mat)

                print(f"   ✅ Curve fallback created: {curve_obj.name}")
                return curve_obj

            except Exception as e2:
                print(f"   ❌ Curve fallback failed: {e2}")
                return None
