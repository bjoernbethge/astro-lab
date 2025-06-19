# Spectral Module

Auto-generated documentation for `tensors.spectral`

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

## SpectralTensor

Tensor for astronomical spectral data with wavelength-dependent operations.

Handles spectroscopic data with wavelength calibration, redshift corrections,
and basic spectral analysis operations.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import SpectralTensor

config = SpectralTensor(

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

**`cuda(self, device: Union[str, torch.device, int, NoneType] = None) -> 'AstroTensorBase'`**

Move tensor to CUDA preserving metadata.

**`numpy(self)`**

Convert to numpy array (data only).

**`has_uncertainties(self) -> bool`**

Check if tensor has associated uncertainties.

**`apply_mask(self, mask: torch.Tensor) -> 'AstroTensorBase'`**

Apply a boolean mask to the tensor.

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
