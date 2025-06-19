# Processing Module

Auto-generated documentation for `astro_lab.data.processing`

## ProcessingConfig

Type-safe configuration for tensor processing.

### Parameters

**`device`** *(string)* = `auto`
  Device for tensor operations

**`batch_size`** *(integer)* = `32`
  Batch size for processing
  *≥1, ≤10000*

**`max_samples`** *(Optional[object])* = `None`
  Maximum samples per survey

**`surveys`** *(Optional[array])* = `None`
  List of surveys to process

### Usage

```python
from astro_lab.data.processing import ProcessingConfig

config = ProcessingConfig(

    # Optional parameters:
    # device="example"
    # batch_size=1
    # max_samples=None
    # surveys=None
)
```

## Pydantic Model Methods

## Functions

### create_simple_processor(device: str = 'auto') -> astro_lab.data.processing.AstroTensorProcessor

Create a simple tensor processor.

### normalize_astronomical_data(data: Dict[str, torch.Tensor], device: str = 'auto') -> Dict[str, torch.Tensor]

Normalize astronomical data for ML processing.

### process_coordinate_dict(coordinates: Dict[str, torch.Tensor], device: str = 'auto') -> Dict[str, torch.Tensor]

Process a dictionary of coordinate tensors.

## Classes

### AstroTensorProcessor

Simplified processor for astronomical tensors.

Focuses on core tensor operations without complex batch processing.

#### Methods

**`to_device(self, tensor: torch.Tensor) -> torch.Tensor`**

Move tensor to processing device.

**`create_spatial_data(self, coordinates: torch.Tensor, ra: Optional[torch.Tensor] = None, dec: Optional[torch.Tensor] = None, distance: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]`**

Create spatial data dictionary from coordinate data.

Parameters
----------
coordinates : torch.Tensor
Coordinate tensor [N, 3]
ra : torch.Tensor, optional
Right ascension
dec : torch.Tensor, optional
Declination
distance : torch.Tensor, optional
Distance values

Returns
-------
Dict[str, torch.Tensor]
Dictionary with spatial data

**`create_survey_data(self, features: torch.Tensor, survey_name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`**

Create survey data dictionary from feature data.

Parameters
----------
features : torch.Tensor
Feature tensor
survey_name : str
Name of the survey
metadata : Dict[str, Any], optional
Additional metadata

Returns
-------
Dict[str, Any]
Dictionary with survey data

**`process_coordinate_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`**

Process astronomical coordinate data.

Parameters
----------
data : Dict[str, torch.Tensor]
Dictionary containing coordinate tensors

Returns
-------
Dict[str, torch.Tensor]
Processed coordinate data

**`normalize_magnitudes(self, magnitudes: torch.Tensor, mag_zero: float = 0.0, mag_range: float = 30.0) -> torch.Tensor`**

Normalize magnitude values for ML processing.

Parameters
----------
magnitudes : torch.Tensor
Magnitude tensor
mag_zero : float
Zero point for normalization
mag_range : float
Magnitude range for scaling

Returns
-------
torch.Tensor
Normalized magnitudes

**`compute_colors(self, magnitudes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`**

Compute astronomical colors from magnitudes.

Parameters
----------
magnitudes : Dict[str, torch.Tensor]
Dictionary of magnitude tensors

Returns
-------
Dict[str, torch.Tensor]
Dictionary of color tensors

**`save_tensors(self, tensors: Dict[str, Any], output_dir: Union[str, pathlib.Path]) -> pathlib.Path`**

Save processed tensors to disk.

Parameters
----------
tensors : Dict[str, Any]
Tensor dictionary to save
output_dir : Union[str, Path]
Output directory

Returns
-------
Path
Path to saved tensors

**`load_tensors(self, tensor_file: Union[str, pathlib.Path]) -> Dict[str, torch.Tensor]`**

Load processed tensors from disk.

Parameters
----------
tensor_file : Union[str, Path]
Path to tensor file

Returns
-------
Dict[str, torch.Tensor]
Loaded tensors

## Constants

- **ASTRO_LAB_TENSORS_AVAILABLE** (bool): `False`
