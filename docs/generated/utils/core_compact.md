# Core Module

Auto-generated documentation for `utils.blender.core`

## Functions

### animate_camera(camera: Any, positions: List[List[float]], target: List[float] = [0, 0, 0], frame_duration: int = 30) -> None

Animate camera along path.

### create_astro_object(object_type: str, name: str, position: Tuple[float, float, float] = (0, 0, 0), scale: float = 1.0, material_config: Optional[Dict[str, Any]] = None, **kwargs) -> Optional[Any]

Create astronomical object with unified API.

### create_camera(position: List[float] = [5, -5, 3], target: List[float] = [0, 0, 0], fov: float = 35.0, name: str = 'AstroCamera') -> Optional[Any]

Create camera with unified API.

### create_camera_path(path_type: str = 'orbit', center: List[float] = [0, 0, 0], radius: float = 8.0, num_positions: int = 8, **kwargs) -> List[List[float]]

Generate camera path for animation.

### create_light(light_type: str, position: List[float] = [0, 0, 5], power: float = 1000.0, color: List[float] = [1.0, 1.0, 1.0], name: str = 'AstroLight', **kwargs) -> Optional[Any]

Create light with unified API.

### create_material(name: str, material_type: str = 'emission', base_color: List[float] = [0.8, 0.8, 0.8], emission_strength: float = 2.0, alpha: float = 1.0, **kwargs) -> Optional[Any]

Create material with unified API.

### normalize_scene(target_scale: float = 5.0, center: bool = True) -> Tuple[float, Tuple[float, float, float]]

Normalize scene objects to target scale.

### render_scene(output_path: str, animation: bool = False) -> bool

Render scene to file.

### reset_scene() -> None

Reset Blender scene to clean state.

### setup_astronomical_scene(datasets: Dict[str, polars.dataframe.frame.DataFrame], scene_name: str = 'AstroViz', max_objects_per_type: int = 100) -> Dict[str, List[Any]]

Setup complete astronomical scene.

### setup_lighting_preset(preset_name: str, **kwargs) -> List[Any]

Setup lighting preset.

### setup_render_settings(engine: str = 'BLENDER_EEVEE_NEXT', resolution: Tuple[int, int] = (1920, 1080), samples: int = 128) -> None

Setup render settings.

### setup_scene(name: str = 'AstroScene', engine: str = 'BLENDER_EEVEE_NEXT') -> None

Setup scene with proper settings.

## Classes

### AstroPlotter

Main astronomical data plotter.

### FuturisticAstroPlotter

Futuristic orbital interface plotter.

### GeometryNodesVisualizer

Geometry Nodes based procedural visualizer.

### GreasePencilPlotter

Grease Pencil based 2D/3D plotter.

## Constants

- **BLENDER_AVAILABLE** (bool): `True`
