# Grease_Pencil_3D Module

Auto-generated documentation for `utils.blender.grease_pencil_3d`

## Classes

### GreasePencil3DPlotter

Advanced 3D plotting using curves for Grease Pencil-style visualizations.

#### Methods

**`create_3d_scatter_plot(self, data: polars.dataframe.frame.DataFrame, x_col: str, y_col: str, z_col: str, color_col: Optional[str] = None, size_col: Optional[str] = None, title: str = '3D Galaxy Distribution', max_points: int = 1000) -> List[Any]`**

Create 3D scatter plot like galaxy distribution visualization.

**`create_3d_trajectory(self, trajectory_data: numpy.ndarray, title: str = 'Orbital Trajectory', color: List[float] = [0.2, 0.8, 1.0], line_width: float = 0.05) -> List[Any]`**

Create 3D trajectory visualization.

**`create_3d_vector_field(self, grid_points: numpy.ndarray, vectors: numpy.ndarray, title: str = 'Gravitational Field', scale: float = 5.0, max_vectors: int = 200) -> List[Any]`**

Create 3D vector field visualization.

**`create_3d_surface_plot(self, x_grid: numpy.ndarray, y_grid: numpy.ndarray, z_values: numpy.ndarray, title: str = 'Surface Plot', color_map: str = 'viridis') -> List[Any]`**

Create 3D surface plot using curve wireframe.

**`clear_objects(self) -> None`**

Clear all created objects.

## Constants

- **BLENDER_AVAILABLE** (bool): `True`
