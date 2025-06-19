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

## Pydantic Model Methods

### Spatial3DTensor Methods

**`to_spherical(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**

Convert to spherical coordinates (RA, Dec, Distance).

Returns:
Tuple of (ra, dec, distance) in degrees and original units

**`to_astropy(self) -> Any`**

Convert to astropy SkyCoord object.

**`query_neighbors(self, query_point: Union[torch.Tensor, numpy.ndarray], radius: float, max_neighbors: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]`**

Fast neighbor query using spatial index.

Args:
query_point: Query coordinates [3] or [1, 3]
radius: Search radius in same units as tensor
max_neighbors: Maximum number of neighbors

Returns:
Tuple of (distances, indices)

**`angular_separation(self, other: 'Spatial3DTensor') -> torch.Tensor`**

Calculate angular separation using dot product.
More efficient than haversine for 3D coordinates.

**`cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor`**

Cone search around center position.

Args:
center: Center position [3] (Cartesian)
radius_deg: Search radius in degrees

Returns:
Boolean mask of objects within cone

**`cross_match(self, other: 'Spatial3DTensor', radius_deg: float = 0.0002777777777777778) -> Dict[str, torch.Tensor]`**

Cross-match with another catalog.

Args:
other: Other spatial tensor to match against
radius_deg: Matching radius in degrees

Returns:
Dictionary with match results

**`to_torch_geometric(self, k: int = 8, radius: Optional[float] = None) -> 'Data'`**

Convert to PyTorch Geometric Data object for GNN processing.

**`transform_coordinates(self, target_system: str) -> 'Spatial3DTensor'`**

Transform to different coordinate system.

Args:
target_system: Target coordinate system ('icrs', 'galactic', 'ecliptic')

**`model_post_init(self: 'BaseModel', context: 'Any', /) -> 'None'`**

This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
self: The BaseModel instance.
context: The context.

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
