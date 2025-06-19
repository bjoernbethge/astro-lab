# Lightcurve Module

Auto-generated documentation for `astro_lab.tensors.lightcurve`

## AstroTensorBase

Base class for all astronomical tensor types using composition.

This class wraps a PyTorch tensor and provides common astronomical
functionality without the complications of tensor subclassing.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.tensors.lightcurve import AstroTensorBase

config = AstroTensorBase(

    # Optional parameters:
    # tensor_type="example"
)
```

## LightcurveTensor

Tensor for astronomical time series and lightcurve data.

Handles time-dependent photometric measurements with physical
properties like periods, amplitudes, and variability characteristics.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from astro_lab.tensors.lightcurve import LightcurveTensor

config = LightcurveTensor(

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
