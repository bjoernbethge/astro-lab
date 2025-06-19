# Photometric Module

Auto-generated documentation for `astro_lab.tensors.photometric`

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

## PhotometricTensor

Tensor for multi-band photometric measurements using composition.

Stores magnitudes/fluxes across multiple bands with associated metadata
like measurement errors, extinction coefficients, and zero points.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import PhotometricTensor

config = PhotometricTensor(

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
