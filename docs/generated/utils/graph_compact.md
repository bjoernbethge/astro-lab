# Graph Module

Auto-generated documentation for `utils.graph`

## Spatial3DTensor

Proper 3D Spatial Tensor for astronomical coordinates.

Uses unified [N, 3] tensor structure for efficient spatial operations.
Compatible with astroML, poliastro, and astropy.

The tensor data is always stored as 3D Cartesian coordinates [x, y, z]
in the specified coordinate system with proper units.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import Spatial3DTensor

config = Spatial3DTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## Functions

### calculate_graph_metrics(data: torch_geometric.data.data.Data) -> dict

Calculate basic graph metrics.

Args:
    data: PyTorch Geometric Data object

Returns:
    Dictionary of graph metrics

### create_spatial_graph(spatial_tensor: astro_lab.tensors.spatial_3d.Spatial3DTensor, method: str = 'knn', k: int = 5, radius: float = 1.0, **kwargs) -> torch_geometric.data.data.Data

Create a spatial graph from Spatial3DTensor.

Args:
    spatial_tensor: Input spatial tensor
    method: Graph construction method ('knn', 'radius')
    k: Number of neighbors for KNN
    radius: Radius for radius graph
    **kwargs: Additional arguments

Returns:
    PyTorch Geometric Data object

### spatial_distance_matrix(coords: torch.Tensor, metric: str = 'euclidean') -> torch.Tensor

Compute pairwise distance matrix.

Args:
    coords: Coordinate tensor [N, 3]
    metric: Distance metric ('euclidean', 'angular')

Returns:
    Distance matrix [N, N]

## Constants

- **TORCH_GEOMETRIC_AVAILABLE** (bool): `True`
