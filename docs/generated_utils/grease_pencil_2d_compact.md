# Grease_Pencil_2D Module

Auto-generated documentation for `utils.blender.grease_pencil_2d`

## Classes

### GreasePencil2DPlotter

Professional 2D plotting using Grease Pencil v3 with curve fallbacks.

#### Methods

**`create_radar_chart(self, data: Dict[str, float], title: str = 'Data Format Comparison', colors: Optional[List[List[float]]] = None, scale: float = 5.0) -> List[Any]`**

Create radar chart like the data format comparison example.

**`create_multi_panel_plot(self, datasets: List[polars.dataframe.frame.DataFrame], panel_titles: List[str], plot_types: List[str], layout: Tuple[int, int] = (2, 2), panel_size: float = 3.0) -> List[Any]`**

Create multi-panel plot like NSA galaxy analysis.

**`create_comparison_plot(self, data1: numpy.ndarray, data2: numpy.ndarray, labels: List[str], title: str = 'Mass Distribution Comparison', colors: Optional[List[List[float]]] = None) -> List[Any]`**

Create comparison histogram plot.

**`clear_objects(self) -> None`**

Clear all created objects.

## Constants

- **BLENDER_AVAILABLE** (bool): `True`
