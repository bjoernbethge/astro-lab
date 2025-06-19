# Exoplanets Module

Auto-generated documentation for `astro_lab.data.datasets.exoplanets`

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
from astro_lab.data.datasets.exoplanets import Spatial3DTensor

config = Spatial3DTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## SurveyTensor

Main coordinator tensor for astronomical survey data using composition.

Coordinates all specialized tensors and provides unified access to:
- Photometric measurements (via PhotometricTensor)
- Time series data (via LightcurveTensor)
- 3D spatial coordinates (via Spatial3DTensor)
- Spectroscopic data (via SpectralTensor)
- Astrometric measurements (via AstrometricTensor)
- Survey transformations and metadata

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.data.datasets.exoplanets import SurveyTensor

config = SurveyTensor(

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

### SurveyTensor Methods

**`get_photometric_tensor(self, band_columns: Optional[List[str]] = None, force_recreate: bool = False)`**

Get or create PhotometricTensor for this survey.

Args:
band_columns: Specific bands to extract (auto-detect if None)
force_recreate: Force recreation even if cached

Returns:
PhotometricTensor with photometric measurements

**`get_spatial_tensor(self, include_distances: bool = True, force_recreate: bool = False)`**

Get or create Spatial3DTensor for coordinate analysis.

Args:
include_distances: Whether to include distance information
force_recreate: Force recreation even if cached

Returns:
Spatial3DTensor with 3D coordinates

**`get_astrometric_tensor(self, include_radial_velocity: bool = True, force_recreate: bool = False) -> 'Spatial3DTensor'`**

Get or create Spatial3DTensor with astrometric data for astrometric analysis.

Args:
include_radial_velocity: Whether to include radial velocity if available
force_recreate: Force recreation even if cached

Returns:
Spatial3DTensor with astrometric data

**`create_lightcurve_tensor(self, time_column: str, magnitude_columns: List[str], error_columns: Optional[List[str]] = None, object_id_column: Optional[str] = None)`**

Create LightcurveTensor from time-series survey data.

Args:
time_column: Name of time/date column
magnitude_columns: Names of magnitude columns
error_columns: Names of error columns (optional)
object_id_column: Name of object ID column (optional)

Returns:
LightcurveTensor with time-series data

**`get_unified_catalog(self) -> Dict[str, Any]`**

Get unified catalog with all specialized tensor data.

Returns:
Dictionary with all available tensor data

**`transform_to_survey(self, target_survey: str, target_release: Optional[str] = None) -> 'SurveyTensor'`**

Transform data to another survey system.

Args:
target_survey: Target survey name
target_release: Target data release

Returns:
Transformed SurveyTensor

**`register_transformation(self, source_survey: str, target_survey: str, transform_func: Callable) -> None`**

Register a custom transformation function.

Args:
source_survey: Source survey name
target_survey: Target survey name
transform_func: Transformation function

**`apply_quality_cuts(self, criteria: Dict[str, Tuple[Optional[float], Optional[float]]]) -> 'SurveyTensor'`**

Apply quality cuts based on survey-specific criteria.

Args:
criteria: Dictionary mapping column names to (min, max) values

Returns:
Filtered SurveyTensor

**`get_column(self, column_name: str) -> torch.Tensor`**

Get values for a specific column.

**`add_derived_columns(self, derived: Dict[str, torch.Tensor]) -> 'SurveyTensor'`**

Add derived columns to the survey data.

Args:
derived: Dictionary mapping column names to tensors

Returns:
New SurveyTensor with added columns

**`get_catalog_data(self) -> Dict[str, Any]`**

Get catalog data in standardized format.

Returns:
Dictionary with standardized catalog fields

**`compute_survey_statistics(self) -> Dict[str, Any]`**

Compute survey-specific statistics.

**`match_to_reference(self, reference: 'SurveyTensor', radius: float = 1.0, unit: str = 'arcsec') -> Dict[str, torch.Tensor]`**

Match to a reference survey catalog using spatial coordinates.

Args:
reference: Reference SurveyTensor
radius: Matching radius
unit: Unit of radius

Returns:
Match results with indices and separations

**`exoplanet_habitability_score(self) -> Optional[torch.Tensor]`**

Calculate habitability score for exoplanets in the survey.

Returns:
Habitability score [0, 1] or None if no exoplanet data

**`atmospheric_escape_rate(self) -> Optional[torch.Tensor]`**

Estimate atmospheric escape rate for exoplanets.

Returns:
Escape rate in kg/s (logarithmic scale) or None if no data

**`biosignature_potential(self) -> Optional[Dict[str, torch.Tensor]]`**

Assess potential for detecting biosignatures in exoplanet survey.

Returns:
Dictionary with biosignature detection metrics or None if no data

**`filter_habitable_exoplanets(self, min_habitability_score: float = 0.5, max_distance_ly: float = 100.0, min_biosignature_potential: float = 0.1) -> 'SurveyTensor'`**

Filter survey to only include potentially habitable exoplanets.

Args:
min_habitability_score: Minimum habitability score
max_distance_ly: Maximum distance in light-years
min_biosignature_potential: Minimum biosignature detection potential

Returns:
Filtered SurveyTensor with habitable exoplanets

**`exoplanet_summary_statistics(self) -> Optional[Dict[str, torch.Tensor]]`**

Calculate summary statistics for exoplanet survey data.

Returns:
Dictionary with summary statistics or None if no exoplanet data

**`dim(self) -> int`**

Number of dimensions.

**`model_post_init(self: 'BaseModel', context: 'Any', /) -> 'None'`**

This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
self: The BaseModel instance.
context: The context.

## Classes

### ExoplanetGraphDataset

PyTorch Geometric InMemoryDataset for Exoplanet data.

Creates spatial graphs from exoplanet systems with connections
based on stellar distances and system properties.

#### Methods

**`download(self)`**

Download exoplanet data if needed.

**`process(self)`**

Process exoplanet catalog into graph format.

**`to_survey_tensor(self) -> Optional[astro_lab.tensors.survey.SurveyTensor]`**

Convert dataset to SurveyTensor format.

**`get_spatial_tensor(self) -> Optional[astro_lab.tensors.spatial_3d.Spatial3DTensor]`**

Extract spatial coordinates as Spatial3DTensor.

## Constants

- **ASTRO_LAB_TENSORS_AVAILABLE** (bool): `True`
