# Time_Series Module

Auto-generated documentation for `astro_lab.data.datasets.time_series`

## LightcurveTensor

Tensor for astronomical time series and lightcurve data.

Handles time-dependent photometric measurements with physical
properties like periods, amplitudes, and variability characteristics.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.data.datasets.time_series import LightcurveTensor

config = LightcurveTensor(

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
from astro_lab.data.datasets.time_series import SurveyTensor

config = SurveyTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## Pydantic Model Methods

### LightcurveTensor Methods

**`dim(self) -> int`**

Number of dimensions.

**`compute_period_folded(self, period: float, epoch: float = 0.0) -> 'LightcurveTensor'`**

Compute period-folded lightcurve.

Args:
period: Folding period
epoch: Reference epoch

Returns:
Period-folded LightcurveTensor

**`compute_statistics(self) -> Dict[str, torch.Tensor]`**

Compute basic lightcurve statistics.

**`filter_by_band(self, band: str) -> 'LightcurveTensor'`**

Filter lightcurve to specific band.

**`time_bin(self, bin_size: float) -> 'LightcurveTensor'`**

Bin lightcurve data in time.

Args:
bin_size: Size of time bins

Returns:
Binned LightcurveTensor

**`get_period(self, object_idx: int = 0) -> Optional[float]`**

Get period for specific object.

**`get_amplitude(self, object_idx: int = 0) -> Optional[float]`**

Get amplitude for specific object.

**`compute_variability_stats(self) -> Dict[str, torch.Tensor]`**

Compute variability statistics for each lightcurve.

Returns:
Dictionary with variability metrics

**`fold_lightcurve(self, period: Optional[float] = None, object_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]`**

Fold lightcurve with given period.

Args:
period: Period for folding (uses stored period if None)
object_idx: Object index if multiple objects

Returns:
Tuple of (folded_times, magnitudes)

**`detect_periods(self, min_period: float = 0.1, max_period: float = 100.0) -> torch.Tensor`**

Detect periods using Lomb-Scargle periodogram.

Args:
min_period: Minimum period to search
max_period: Maximum period to search

Returns:
Detected periods for each object

**`classify_variability(self) -> List[str]`**

Classify variability type based on lightcurve properties.

Returns:
List of variability classifications

**`phase_lightcurve(self, period: Optional[float] = None, epoch: float = 0.0) -> torch.Tensor`**

Calculate phase for each observation.

Args:
period: Period for phasing
epoch: Reference epoch

Returns:
Phase values (0-1)

**`bin_lightcurve(self, n_bins: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**

Bin lightcurve in phase.

Args:
n_bins: Number of phase bins

Returns:
Tuple of (bin_centers, binned_mags, bin_errors)

**`to_dict(self) -> Dict[str, Any]`**

Convert to dictionary representation.

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

### LINEARLightcurveDataset

PyTorch Geometric InMemoryDataset for LINEAR lightcurve data.

Creates spatial graphs from variable star lightcurves with connections
based on sky coordinates and period similarity.

#### Methods

**`download(self)`**

Download LINEAR data if needed.

**`process(self)`**

Process LINEAR lightcurve data into graph format.

**`to_survey_tensor(self) -> Optional[astro_lab.tensors.survey.SurveyTensor]`**

Convert dataset to SurveyTensor format.

**`get_lightcurve_tensor(self) -> Optional[astro_lab.tensors.lightcurve.LightcurveTensor]`**

Extract lightcurve data as LightcurveTensor.

### RRLyraeDataset

PyTorch Geometric InMemoryDataset for RR Lyrae variable star data.

Creates spatial graphs from RR Lyrae stars with connections
based on sky coordinates and period similarity.

#### Methods

**`download(self)`**

Download RR Lyrae data if needed.

**`process(self)`**

Process RR Lyrae data into graph format.

**`to_survey_tensor(self) -> Optional[astro_lab.tensors.survey.SurveyTensor]`**

Convert dataset to SurveyTensor format.

**`get_lightcurve_tensor(self) -> Optional[astro_lab.tensors.lightcurve.LightcurveTensor]`**

Extract lightcurve data as LightcurveTensor.

## Constants

- **ASTRO_LAB_TENSORS_AVAILABLE** (bool): `True`
