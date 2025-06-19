# Orbital Module

Auto-generated documentation for `astro_lab.tensors.orbital`

## AstroTensorBase

Base class for all astronomical tensor types using composition.

This class wraps a PyTorch tensor and provides common astronomical
functionality without the complications of tensor subclassing.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import AstroTensorBase

config = AstroTensorBase(

    # Optional parameters:
    # tensor_type="example"
)
```

## ManeuverTensor

Tensor for orbital maneuvers and delta-V calculations.

Handles impulsive maneuvers, transfer calculations, and fuel optimization.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import ManeuverTensor

config = ManeuverTensor(

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
from docs.auto.schemas.data_schemas import OrbitTensor

config = OrbitTensor(

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

### ManeuverTensor Methods

**`fuel_mass_ratio(self, specific_impulse: float) -> torch.Tensor`**

Calculate fuel mass ratio using rocket equation.

Args:
specific_impulse: Specific impulse in seconds

Returns:
Mass ratio (m_initial / m_final)

**`apply_to_orbit(self, orbit: astro_lab.tensors.orbital.OrbitTensor) -> astro_lab.tensors.orbital.OrbitTensor`**

Apply maneuver to orbit.

Args:
orbit: Initial orbit

Returns:
Orbit after maneuver

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
