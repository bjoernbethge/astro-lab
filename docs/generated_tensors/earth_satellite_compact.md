# Earth_Satellite Module

Auto-generated documentation for `astro_lab.tensors.earth_satellite`

## AstroTensorBase

Base class for all astronomical tensor types using composition.

This class wraps a PyTorch tensor and provides common astronomical
functionality without the complications of tensor subclassing.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.tensors.earth_satellite import AstroTensorBase

config = AstroTensorBase(

    # Optional parameters:
    # tensor_type="example"
)
```

## AttitudeTensor

Tensor for satellite attitude and orientation.

Handles quaternions, Euler angles, and attitude propagation.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.tensors.earth_satellite import AttitudeTensor

config = AttitudeTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## EarthSatelliteTensor

Tensor for Earth satellites with Earth-specific operations using composition.

Handles Earth-specific functionality like ground tracks,
field of view calculations, and pass predictions.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.tensors.earth_satellite import EarthSatelliteTensor

config = EarthSatelliteTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## OrbitTensor

Tensor for orbital elements and state vectors in celestial mechanics.

Handles Keplerian orbital elements, Cartesian state vectors, and orbital
propagation for satellites, planets, and interstellar objects.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.tensors.earth_satellite import OrbitTensor

config = OrbitTensor(

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
from astro_lab.tensors.earth_satellite import Spatial3DTensor

config = Spatial3DTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## Pydantic Model Methods

### AstroTensorBase Methods

**`dim(self) -> int`**

Number of dimensions.

**`unsqueeze(self, dim: int) -> 'AstroTensorBase'`**

Unsqueeze tensor preserving metadata.

**`squeeze(self, dim: Optional[int] = None) -> 'AstroTensorBase'`**

Squeeze tensor preserving metadata.

**`update_metadata(self, **kwargs) -> None`**

Update metadata.

**`get_metadata(self, key: str, default: Any = None) -> Any`**

Get metadata value.

**`to_dict(self) -> Dict[str, Any]`**

Convert to dictionary representation.

**`clone(self) -> 'AstroTensorBase'`**

Clone tensor preserving all metadata.

**`detach(self) -> 'AstroTensorBase'`**

Detach tensor preserving all metadata.

**`to(self, *args, **kwargs) -> 'AstroTensorBase'`**

Move tensor to device/dtype preserving metadata.

**`cpu(self) -> 'AstroTensorBase'`**

Move tensor to CPU preserving metadata.

**`cuda(self, device: Union[int, str, torch.device, NoneType] = None) -> 'AstroTensorBase'`**

Move tensor to CUDA preserving metadata.

**`numpy(self)`**

Convert to numpy array (data only).

**`has_uncertainties(self) -> bool`**

Check if tensor has associated uncertainties.

**`apply_mask(self, mask: torch.Tensor) -> 'AstroTensorBase'`**

Apply a boolean mask to the tensor.

### AttitudeTensor Methods

**`to_rotation_matrix(self) -> torch.Tensor`**

Convert to rotation matrix.

**`pointing_vector(self, body_vector: Optional[torch.Tensor] = None) -> torch.Tensor`**

Calculate pointing vector in inertial frame.

Args:
body_vector: Vector in body frame (default: [0, 0, 1] - z-axis)

Returns:
Pointing vector in inertial frame

**`model_post_init(self: 'BaseModel', context: 'Any', /) -> 'None'`**

This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
self: The BaseModel instance.
context: The context.

### EarthSatelliteTensor Methods

**`ground_track(self, time_span: torch.Tensor, earth_rotation_rate: float = 7.2921159e-05)`**

Calculate ground track coordinates.

Args:
time_span: Time values (seconds from epoch)
earth_rotation_rate: Earth rotation rate (rad/s)

Returns:
Spatial3DTensor with ground track coordinates

**`field_of_view_footprint(self, altitude: float, fov_angle: float, time_span: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]`**

Calculate field of view footprint on Earth surface.

Args:
altitude: Target altitude above Earth surface (km)
fov_angle: Field of view half-angle (degrees)
time_span: Time values for propagation

Returns:
Dictionary with footprint coordinates and area

**`ground_range(self, elevation_angle: float = 0.0) -> torch.Tensor`**

Calculate maximum ground range for given elevation angle.

Args:
elevation_angle: Minimum elevation angle (degrees)

Returns:
Ground range (km)

**`pass_prediction(self, ground_station_coords, time_span: torch.Tensor, min_elevation: float = 10.0) -> Dict[str, torch.Tensor]`**

Predict satellite passes over a ground station.

Args:
ground_station_coords: Ground station coordinates (Spatial3DTensor)
time_span: Time values to check (seconds from epoch)
min_elevation: Minimum elevation angle (degrees)

Returns:
Dictionary with pass information

**`sun_illumination(self, time_span: torch.Tensor) -> Dict[str, torch.Tensor]`**

Calculate satellite illumination conditions.

Args:
time_span: Time values (seconds from epoch)

Returns:
Dictionary with illumination information

**`orbital_decay_prediction(self, atmospheric_density: float = 1e-12, drag_coefficient: float = 2.2, satellite_area: float = 1.0, satellite_mass: float = 100.0) -> Dict[str, torch.Tensor]`**

Predict orbital decay due to atmospheric drag.

Args:
atmospheric_density: Atmospheric density at satellite altitude
drag_coefficient: Satellite drag coefficient
satellite_area: Cross-sectional area
satellite_mass: Satellite mass

Returns:
Dictionary with decay information

**`model_post_init(self: 'BaseModel', context: 'Any', /) -> 'None'`**

This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
self: The BaseModel instance.
context: The context.

### OrbitTensor Methods

**`to_cartesian(self) -> 'OrbitTensor'`**

Convert Keplerian elements to Cartesian state vectors.

**`to_keplerian(self) -> 'OrbitTensor'`**

Convert Cartesian state vectors to Keplerian elements.

**`propagate(self, time_span: torch.Tensor) -> 'OrbitTensor'`**

Propagate orbit using specified propagator.

Args:
time_span: Time values to propagate to (seconds from epoch)

Returns:
Propagated OrbitTensor

**`orbital_period(self) -> torch.Tensor`**

Calculate orbital period (seconds).

**`apoapsis_periapsis(self) -> Tuple[torch.Tensor, torch.Tensor]`**

Calculate apoapsis and periapsis distances.

**`habitable_zone_distance(self) -> Tuple[torch.Tensor, torch.Tensor]`**

Calculate habitable zone boundaries for exoplanets.

Returns:
Tuple of (inner_edge, outer_edge) distances in AU

**`is_in_habitable_zone(self) -> torch.Tensor`**

Check if orbit is within the habitable zone.

Returns:
Boolean tensor indicating if orbit is in HZ

**`transit_probability(self) -> torch.Tensor`**

Calculate transit probability for exoplanets.

Returns:
Transit probability [0, 1]

**`transit_duration(self, planet_radius: float = 6371.0) -> torch.Tensor`**

Calculate transit duration for exoplanets.

Args:
planet_radius: Planet radius in km

Returns:
Transit duration in hours

**`equilibrium_temperature(self, albedo: float = 0.3) -> torch.Tensor`**

Calculate equilibrium temperature for exoplanets.

Args:
albedo: Planetary albedo [0, 1]

Returns:
Equilibrium temperature in Kelvin

**`model_post_init(self: 'BaseModel', context: 'Any', /) -> 'None'`**

This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
self: The BaseModel instance.
context: The context.

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
