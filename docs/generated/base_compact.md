# Base Module

Auto-generated documentation for `astro_lab.tensors.base`

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
