# Transforms Module

Auto-generated documentation for `astro_lab.data.transforms`

## Functions

### get_default_astro_transforms(add_colors: bool = True, add_distances: bool = True, normalize: bool = True, coordinate_system: str = 'icrs') -> torch_geometric.transforms.compose.Compose

Get default astronomical transforms for most use cases.

Parameters
----------
add_colors : bool, default True
    Whether to add color indices
add_distances : bool, default True
    Whether to add distance features
normalize : bool, default True
    Whether to normalize features
coordinate_system : str, default "icrs"
    Target coordinate system

Returns
-------
Compose
    Composed transform pipeline

### get_exoplanet_transforms() -> torch_geometric.transforms.compose.Compose

Get transforms optimized for exoplanet data.

### get_galaxy_transforms() -> torch_geometric.transforms.compose.Compose

Get transforms optimized for galaxy data.

### get_stellar_transforms() -> torch_geometric.transforms.compose.Compose

Get transforms optimized for stellar data.

## Classes

### AddAstronomicalColors

Add astronomical color indices to node features.

#### Methods

**`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

Compute and add color indices.

### AddDistanceFeatures

Add distance-based features for astronomical objects.

#### Methods

**`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

Add distance-based features.

### AddRedshiftFeatures

Add redshift-derived features for extragalactic objects.

#### Methods

**`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

Add redshift-derived features.

### CoordinateSystemTransform

Transform between different astronomical coordinate systems.

#### Methods

**`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

Apply coordinate system transformation.

### NormalizeAstronomicalFeatures

Normalize astronomical features (magnitudes, colors, etc.).

#### Methods

**`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

Normalize astronomical features.
