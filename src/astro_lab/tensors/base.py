"""
Base Tensor Classes - Core Tensor Infrastructure
===============================================

Provides base classes and interfaces for all tensor types in the AstroLab framework.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self


class AstroTensorBase(BaseModel):
    """
    A base class for tensor-like objects in astrophysics.

    This class uses Pydantic for data validation and management. It ensures that the
    input data is always converted to a torch.Tensor. Metadata can be passed as
    keyword arguments during initialization.
    """

    data: torch.Tensor
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        json_encoders={torch.Tensor: lambda t: t.tolist()},
    )

    @model_validator(mode="before")
    @classmethod
    def _convert_data_to_tensor_and_collect_meta(
        cls, values: Union[Dict, Any]
    ) -> Dict:
        """
        Validator to convert input data to a torch.Tensor and handle metadata.

        - If the input is not a dictionary (e.g., a raw tensor or array), it wraps it.
        - It converts various data types (pd.DataFrame, np.ndarray, list) to torch.Tensor.
        - It collects any non-field keyword arguments into the 'meta' dictionary.
        """
        if not isinstance(values, dict):
            # If a raw tensor/array is passed, assume it's the data.
            values = {"data": values}

        # Convert data to tensor
        data_input = values.get("data")
        if data_input is not None:
            if isinstance(data_input, pd.DataFrame):
                tensor_data = torch.from_numpy(data_input.values).float()
            elif isinstance(data_input, np.ndarray):
                tensor_data = torch.from_numpy(data_input).float()
            elif isinstance(data_input, list):
                tensor_data = torch.tensor(data_input, dtype=torch.float32)
            elif isinstance(data_input, torch.Tensor):
                tensor_data = data_input
            else:
                raise TypeError(
                    f"Unsupported data type for tensor conversion: {type(data_input)}"
                )
            values["data"] = tensor_data
        else:
            raise ValueError("'data' field is required.")

        # Collect extra fields into meta
        if "meta" not in values:
            values["meta"] = {}
        
        model_fields = cls.model_fields.keys()
        meta_kwargs = {
            k: v for k, v in values.items() if k not in model_fields and k != "data"
        }
        
        values["meta"].update(meta_kwargs)
        
        # Remove original meta kwargs from the root to avoid Pydantic errors
        for key in meta_kwargs:
            del values[key]

        return values

    @field_validator("data")
    @classmethod
    def _validate_tensor_properties(cls, v: torch.Tensor) -> torch.Tensor:
        """Validate tensor properties after conversion."""
        if v.numel() == 0:
            raise ValueError("Tensor cannot be empty.")
        if not torch.isfinite(v).all():
            raise ValueError("Tensor contains non-finite values (NaN or Inf).")
        return v

    def _create_new_instance(self, new_data: torch.Tensor, **extra_meta) -> Self:
        """
        Helper to create a new instance with updated data, preserving all other
        Pydantic fields and metadata from the original instance.
        """
        # Dump the current model's data, excluding the 'data' field itself,
        # to preserve all other attributes like 'bands', 'coordinate_system', etc.
        new_instance_data = self.model_dump(exclude={"data"})
        
        # Update with the new tensor data
        new_instance_data["data"] = new_data
        
        # Update metadata if any extra is provided
        if extra_meta:
            if "meta" not in new_instance_data:
                new_instance_data["meta"] = {}
            new_instance_data["meta"].update(extra_meta)
            
        return self.__class__(**new_instance_data)

    # =========================================================================
    # Core Properties (redirect to the tensor)
    # =========================================================================
    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.dim()

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        return self.data.size() if dim is None else self.data.size(dim)

    def numel(self) -> int:
        return self.data.numel()

    # =========================================================================
    # Metadata access
    # =========================================================================
    def get_metadata(self, key: Optional[str] = None, default: Any = None) -> Any:
        return self.meta if key is None else self.meta.get(key, default)

    def update_metadata(self, **kwargs) -> Self:
        self.meta.update(kwargs)
        return self # Allow chaining

    def add_history_entry(self, description: str, **details) -> Self:
        """Adds an entry to the tensor's metadata history."""
        if "history" not in self.meta:
            self.meta["history"] = []
        self.meta["history"].append({"description": description, "details": details})
        return self

    def has_metadata(self, key: str) -> bool:
        return key in self.meta

    # =========================================================================
    # Tensor operations that preserve class and metadata
    # =========================================================================
    def clone(self) -> Self:
        return self._create_new_instance(self.data.clone())

    def copy(self) -> Self:
        return self.clone()

    def detach(self) -> Self:
        return self._create_new_instance(self.data.detach())

    def to(self, *args, **kwargs) -> Self:
        return self._create_new_instance(self.data.to(*args, **kwargs))

    def cpu(self) -> Self:
        return self.to(device=torch.device("cpu"))

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> Self:
        # Simplifies finding the default CUDA device.
        cuda_device = device or "cuda"
        return self.to(device=cuda_device)

    def numpy(self) -> np.ndarray:
        return self.data.cpu().numpy()

    def apply_mask(self, mask: torch.Tensor) -> Self:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError("Mask must be a boolean torch.Tensor.")
        return self._create_new_instance(self.data[mask])

    # =========================================================================
    # Python Dunder Methods
    # =========================================================================
    def __getitem__(self, item: Any) -> Self:
        """
        Allows indexing into the tensor, returning a new instance of the class.
        """
        return self._create_new_instance(self.data[item])

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        """
        Provides a detailed representation of the tensor object, including its
        class, shape, device, and metadata.
        """
        class_name = self.__class__.__name__
        shape_str = f"shape={self.shape}"
        device_str = f"device='{self.device}'"
        meta_str = f"meta={self.meta}"
        data_preview = str(self.data)
        return f"{class_name}({shape_str}, {device_str}, {meta_str})\nTensor:\n{data_preview}"

    # By using Pydantic's BaseModel, we get reasonable default serialization.
    # Custom __getstate__ and __setstate__ are often not needed unless you
    # have complex, non-serializable objects that Pydantic can't handle.
    # The `json_encoders` in model_config helps with serialization to JSON.
