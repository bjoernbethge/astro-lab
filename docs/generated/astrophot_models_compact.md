# Astrophot_Models Module

Auto-generated documentation for `astro_lab.models.astrophot_models`

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
from docs.auto.schemas.data_schemas import SurveyTensor

config = SurveyTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## Pydantic Model Methods

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

### AstroPhotGNN

Graph Neural Network with AstroPhot integration for galaxy modeling.

#### Methods

**`extract_galaxy_features(self, survey_tensor: astro_lab.tensors.survey.SurveyTensor) -> torch.Tensor`**

Extract galaxy features from SurveyTensor using existing encoders.

**`forward(self, x: Union[torch.Tensor, astro_lab.tensors.survey.SurveyTensor], edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]`**

Forward pass with AstroPhot integration.

### BulgeParameterHead

Output head for bulge component parameters.

#### Methods

**`forward(self, x: torch.Tensor) -> torch.Tensor`**

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### DiskParameterHead

Output head for exponential disk parameters.

#### Methods

**`forward(self, x: torch.Tensor) -> torch.Tensor`**

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### GlobalGalaxyHead

Output head for global galaxy parameters.

#### Methods

**`forward(self, x: torch.Tensor) -> torch.Tensor`**

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### NSAGalaxyModeler

Specialized model for NSA galaxy catalog.

### SersicParameterHead

Output head for Sersic profile parameters.

#### Methods

**`forward(self, x: torch.Tensor) -> torch.Tensor`**

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

## Constants

- **ASTROPHOT_AVAILABLE** (bool): `True`
