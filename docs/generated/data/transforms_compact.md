# Transforms Module

Auto-generated documentation for `data.transforms`

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

### AddDistanceFeatures

Add distance-based features for astronomical objects.

### AddRedshiftFeatures

Add redshift-derived features for extragalactic objects.

### CoordinateSystemTransform

Transform between different astronomical coordinate systems.

### NormalizeAstronomicalFeatures

Normalize astronomical features (magnitudes, colors, etc.).
