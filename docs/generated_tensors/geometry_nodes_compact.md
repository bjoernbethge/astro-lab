# Geometry_Nodes Module

Auto-generated documentation for `astro_lab.utils.blender.advanced.geometry_nodes`

## Functions

### create_galaxy_comparison()

Create comparison of different galaxy types.

### create_hr_diagram_demo()

Create a demonstration HR diagram.

## Classes

### AstronomicalMaterials

Materials for astronomical objects

#### Methods

**`create_stellar_classification_material(spectral_class: str = 'G') -> bpy.types.Material`**

Create material based on stellar spectral classification.

Args:
spectral_class: Stellar class (O, B, A, F, G, K, M)

Returns:
Created material

### ProceduralAstronomy

Generate procedural astronomical structures

#### Methods

**`create_hr_diagram_3d(stellar_data: List[Dict[str, float]], scale_factor: float = 1.0) -> bpy_types.Object`**

Create a 3D Hertzsprung-Russell diagram visualization.

Args:
stellar_data: List of stellar parameters
scale_factor: Scale factor for the diagram

Returns:
Created HR diagram object

**`create_galaxy_structure(center: Vector = Vector((0.0, 0.0, 0.0)), galaxy_type: str = 'spiral', num_stars: int = 50000, radius: float = 20.0) -> bpy_types.Object`**

Create a procedural galaxy structure.

Args:
center: Galaxy center position
galaxy_type: Type of galaxy ('spiral', 'elliptical', 'irregular')
num_stars: Number of stars to generate
radius: Galaxy radius

Returns:
Created galaxy object
