# Base Module

Auto-generated documentation for `astro_lab.data.datasets.base`

## LightcurveTensor

Tensor for astronomical time series and lightcurve data.

Handles time-dependent photometric measurements with physical
properties like periods, amplitudes, and variability characteristics.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.data.datasets.base import LightcurveTensor

config = LightcurveTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## PhotometricTensor

Tensor for multi-band photometric measurements using composition.

Stores magnitudes/fluxes across multiple bands with associated metadata
like measurement errors, extinction coefficients, and zero points.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.data.datasets.base import PhotometricTensor

config = PhotometricTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

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
from astro_lab.data.datasets.base import Spatial3DTensor

config = Spatial3DTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## SpectralTensor

Tensor for astronomical spectral data with wavelength-dependent operations.

Handles spectroscopic data with wavelength calibration, redshift corrections,
and basic spectral analysis operations.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.data.datasets.base import SpectralTensor

config = SpectralTensor(

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
from astro_lab.data.datasets.base import SurveyTensor

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

### PhotometricTensor Methods

**`get_band_data(self, band: str) -> torch.Tensor`**

Get data for a specific band.

**`get_band_error(self, band: str) -> Optional[torch.Tensor]`**

Get measurement error for a specific band.

**`compute_colors(self, band1: str, band2: str) -> torch.Tensor`**

Compute color (magnitude difference) between two bands.

**`to_flux(self, band: Optional[str] = None) -> 'PhotometricTensor'`**

Convert magnitudes to flux units.

Args:
band: Specific band to convert (None for all bands)

Returns:
PhotometricTensor with flux data

**`to_magnitude(self, band: Optional[str] = None) -> 'PhotometricTensor'`**

Convert flux to magnitude units.

Args:
band: Specific band to convert (None for all bands)

Returns:
PhotometricTensor with magnitude data

**`apply_extinction_correction(self, extinction_values: Union[float, torch.Tensor, Dict[str, float]]) -> 'PhotometricTensor'`**

Apply extinction correction to photometric data.

Args:
extinction_values: Extinction values (A_V or per-band)

Returns:
Extinction-corrected PhotometricTensor

**`compute_synthetic_colors(self) -> Dict[str, torch.Tensor]`**

Compute common synthetic colors from available bands.

**`filter_by_bands(self, selected_bands: List[str]) -> 'PhotometricTensor'`**

Create new PhotometricTensor with only selected bands.

**`to_dict(self) -> Dict[str, Any]`**

Convert to dictionary representation.

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

### SpectralTensor Methods

**`apply_redshift(self, z: float) -> 'SpectralTensor'`**

Apply redshift to spectrum.

Args:
z: Redshift value

Returns:
New SpectralTensor with redshift applied

**`deredshift(self) -> 'SpectralTensor'`**

Remove redshift, returning to rest frame.

**`to_velocity_space(self, rest_wavelength: float) -> Tuple[torch.Tensor, torch.Tensor]`**

Convert to velocity space around a rest wavelength.

Args:
rest_wavelength: Rest wavelength for velocity calculation

Returns:
Tuple of (velocities, flux_data)

**`measure_line(self, wavelength: float, window: float = 50.0) -> Dict[str, torch.Tensor]`**

Measure spectral line properties.

Args:
wavelength: Central wavelength
window: Window size around line (in wavelength units)

Returns:
Dictionary with line measurements

**`atmospheric_transmission_spectrum(self, planet_radius: float, stellar_radius: float, atmospheric_scale_height: float, molecular_species: Optional[List[str]] = None) -> 'SpectralTensor'`**

Calculate atmospheric transmission spectrum for exoplanet transit.

Args:
planet_radius: Planet radius in Earth radii
stellar_radius: Stellar radius in solar radii
atmospheric_scale_height: Atmospheric scale height in km
molecular_species: List of molecular species to include

Returns:
SpectralTensor with transmission spectrum

**`biosignature_detection(self, snr_threshold: float = 5.0, observation_time: float = 10.0) -> Dict[str, torch.Tensor]`**

Assess biosignature detection potential in spectrum.

Args:
snr_threshold: Minimum SNR for detection
observation_time: Total observation time in hours

Returns:
Dictionary with detection metrics for biosignatures

**`interstellar_reddening_correction(self, distance_ly: float, av_per_kpc: float = 1.0) -> 'SpectralTensor'`**

Correct spectrum for interstellar reddening.

Args:
distance_ly: Distance to object in light-years
av_per_kpc: Visual extinction per kiloparsec

Returns:
Dereddened SpectralTensor

**`stellar_classification(self) -> Dict[str, Union[str, torch.Tensor, Dict]]`**

Classify stellar spectrum and determine stellar parameters.

Returns:
Dictionary with stellar classification results

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

## Functions

### get_device()

Get the best available device (GPU if available, else CPU).

### gpu_knn_graph(coords, k_neighbors=8, max_distance=None, device=None)

Create k-NN graph using GPU acceleration when available.

### to_device(tensor, device=None)

Move tensor to device with memory optimization.

## Classes

### AstroLabDataset

General astronomical dataset using PyTorch Geometric's InMemoryDataset.

Provides easy access to astronomical catalogs for machine learning.
Creates tensor-based datasets from magnitude columns.

#### Methods

**`download(self)`**

Download not needed - using local catalog file.

**`process(self)`**

Process catalog into graph format.

**`get_dataset_info(self) -> Dict[str, Any]`**

Get information about the dataset.

## Constants

- **ASTRO_LAB_TENSORS_AVAILABLE** (bool): `True`
