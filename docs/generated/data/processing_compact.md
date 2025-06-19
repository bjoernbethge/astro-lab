# Processing Module

Auto-generated documentation for `data.processing`

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
from docs.auto.schemas.data_schemas import ProcessingConfig

config = ProcessingConfig(

    # Optional parameters:
    # device="example"
    # batch_size=1
    # max_samples=None
    # surveys=None
)
```

## Functions

### create_simple_processor(device: str = 'auto') -> data.processing.AstroTensorProcessor

Create a simple tensor processor.

### normalize_astronomical_data(data: Dict[str, torch.Tensor], device: str = 'auto') -> Dict[str, torch.Tensor]

Normalize astronomical data for ML processing.

### process_coordinate_dict(coordinates: Dict[str, torch.Tensor], device: str = 'auto') -> Dict[str, torch.Tensor]

Process a dictionary of coordinate tensors.

## Classes

### AstroTensorProcessor

Simplified processor for astronomical tensors.

Focuses on core tensor operations without complex batch processing.

## Constants

- **ASTRO_LAB_TENSORS_AVAILABLE** (bool): `False`
