"""
Base TensorDict for AstroLab
============================

Base class for all astronomical TensorDicts with common functionality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class AstroTensorDict(TensorDict):
    """Base class for all astronomical TensorDicts.

    Extends TensorDict with astronomical-specific functionality
    while maintaining native PyTorch performance.

    Note: Metadata is stored separately from tensor data to maintain
    TensorDict compatibility and performance.
    """

    def __init__(self, data: Dict[str, Any], **kwargs):
        """Initialize AstroTensorDict.

        Args:
            data: Dictionary of tensors and metadata
            **kwargs: Additional TensorDict arguments (batch_size, device, etc.)
        """
        # Extract metadata before passing to parent
        metadata = data.pop("meta", {})

        # Initialize parent TensorDict
        super().__init__(data, **kwargs)

        # Store metadata separately
        self._metadata = metadata

        # Add default metadata
        if "tensor_type" not in self._metadata:
            self._metadata["tensor_type"] = self.__class__.__name__
        if "creation_time" not in self._metadata:
            from datetime import datetime

            self._metadata["creation_time"] = datetime.now().isoformat()

        logger.debug(f"Created {self.__class__.__name__} with {self.n_objects} objects")

    @property
    def meta(self) -> Dict[str, Any]:
        """Access to metadata dictionary."""
        return self._metadata

    @property
    def n_objects(self) -> int:
        """Number of astronomical objects in this tensor."""
        return self.batch_size[0] if self.batch_size else 0

    def add_history(self, operation: str, **details) -> "AstroTensorDict":
        """Add operation to processing history.

        Args:
            operation: Name of the operation
            **details: Additional details about the operation

        Returns:
            Self for chaining
        """
        if "history" not in self._metadata:
            self._metadata["history"] = []

        from datetime import datetime

        self._metadata["history"].append(
            {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "details": details,
            }
        )

        return self

    def validate(self) -> bool:
        """Validate tensor structure.

        Returns:
            True if valid, raises ValueError if not
        """
        # Check that all tensors have compatible batch dimensions
        batch_size = self.batch_size

        for key, value in self.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                # Check first dimensions match batch size
                value_batch_dims = value.shape[: len(batch_size)]
                if value_batch_dims != batch_size:
                    raise ValueError(
                        f"Tensor '{key}' has incompatible batch dimensions: "
                        f"expected {batch_size}, got {value_batch_dims}"
                    )

        return True

    def select_subset(
        self, indices: Union[torch.Tensor, List[int], slice]
    ) -> "AstroTensorDict":
        """
        Select a subset of objects.

        Args:
            indices: Indices to select

        Returns:
            New AstroTensorDict with selected objects
        """
        if isinstance(indices, (list, slice)):
            indices = torch.tensor(indices) if isinstance(indices, list) else indices

        # Select data
        subset_data = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) == len(self.batch_size):
                    # Same dimensionality as batch
                    subset_data[key] = value[indices]
                else:
                    # Higher dimensionality, select along first dimension
                    subset_data[key] = value[indices]
            else:
                subset_data[key] = value

        # Copy metadata
        subset_data["meta"] = self._metadata.copy()

        # Create new instance
        result = type(self)(subset_data)
        n_selected = len(indices) if isinstance(indices, (list, torch.Tensor)) else 1
        result.add_history(
            "select_subset",
            n_selected=n_selected,
        )

        return result

    def filter_by_condition(self, condition: torch.Tensor) -> "AstroTensorDict":
        """
        Filter objects by boolean condition.

        Args:
            condition: Boolean tensor [N] indicating which objects to keep

        Returns:
            Filtered AstroTensorDict
        """
        if not isinstance(condition, torch.Tensor) or condition.dtype != torch.bool:
            raise ValueError("Condition must be a boolean tensor")

        indices = torch.where(condition)[0]
        result = self.select_subset(indices)
        result.add_history(
            "filter_by_condition", n_kept=len(indices), n_total=len(condition)
        )

        return result

    def split_by_key(
        self, split_key: str, values: Optional[List] = None
    ) -> Dict[str, "AstroTensorDict"]:
        """
        Split tensor dict by values in a specific key.

        Args:
            split_key: Key to split by
            values: Specific values to split by (if None, use unique values)

        Returns:
            Dictionary mapping values to AstroTensorDicts
        """
        if split_key not in self:
            raise ValueError(f"Split key '{split_key}' not found")

        split_tensor = self[split_key]
        if values is None:
            values = torch.unique(split_tensor).tolist()
        if values is None:
            return {}
        result = {}
        for value in values:
            mask = split_tensor == value
            subset = self.filter_by_condition(mask)
            result[str(value)] = subset
        return result

    def concatenate(self, other: "AstroTensorDict", dim: int = 0) -> "AstroTensorDict":
        """
        Concatenate with another AstroTensorDict.

        Args:
            other: Other tensor dict to concatenate
            dim: Dimension to concatenate along

        Returns:
            Concatenated AstroTensorDict
        """
        if not isinstance(other, AstroTensorDict):
            raise ValueError("Can only concatenate with another AstroTensorDict")

        # Find common keys
        common_keys = set(list(self.keys())) & set(list(other.keys()))

        concat_data = {}
        for key in common_keys:
            if isinstance(self[key], torch.Tensor) and isinstance(
                other[key], torch.Tensor
            ):
                concat_data[key] = torch.cat([self[key], other[key]], dim=dim)
            else:
                # Keep the first value for non-tensor data
                concat_data[key] = self[key]

        # Merge metadata
        merged_meta = self._metadata.copy()
        merged_meta.update(other._metadata)
        concat_data["meta"] = merged_meta

        result = type(self)(concat_data)
        result.add_history(
            "concatenate", n_self=self.n_objects, n_other=other.n_objects
        )

        return result

    def extract_features(self, feature_types: Optional[List[str]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Extract features from the TensorDict.
        
        This is the central API for feature extraction that should be overridden 
        by subclasses for domain-specific logic.
        
        Args:
            feature_types: List of feature types to extract (e.g., ['spatial', 'temporal'])
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            Dictionary mapping feature names to tensors
        """
        # Default implementation: return all tensor data except metadata
        features = {}

        for key, value in self.items():
            if key == "meta":
                continue

            if isinstance(value, torch.Tensor):
                # Filter by feature types if specified
                if feature_types is None:
                    features[key] = value
                else:
                    # Default classification of keys by type
                    if self._classify_feature_type(key) in feature_types:
                        features[key] = value

        return features

    def _classify_feature_type(self, key: str) -> str:
        """
        Classify a tensor key into a feature type.
        
        Args:
            key: Tensor key name
            
        Returns:
            Feature type classification
        """
        # Default classification logic
        spatial_keys = {"coordinates", "pos", "position", "x", "y", "z", "ra", "dec"}
        temporal_keys = {"time", "epoch", "mjd", "jd", "timestamp"}
        photometric_keys = {"magnitudes", "flux", "colors", "g", "r", "i", "z", "bp", "rp"}
        kinematic_keys = {"pmra", "pmdec", "proper_motion", "radial_velocity", "parallax"}

        key_lower = key.lower()

        if any(spatial_key in key_lower for spatial_key in spatial_keys):
            return "spatial"
        elif any(temporal_key in key_lower for temporal_key in temporal_keys):
            return "temporal"
        elif any(photo_key in key_lower for photo_key in photometric_keys):
            return "photometric"
        elif any(kinematic_key in key_lower for kinematic_key in kinematic_keys):
            return "kinematic"
        else:
            return "generic"

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available features and their properties.
        
        Returns:
            Dictionary with feature summary statistics
        """
        features = self.extract_features()
        summary = {
            "total_features": len(features),
            "feature_types": {},
            "feature_shapes": {},
            "feature_dtypes": {},
        }

        # Classify and summarize features
        for key, tensor in features.items():
            feature_type = self._classify_feature_type(key)

            if feature_type not in summary["feature_types"]:
                summary["feature_types"][feature_type] = []
            summary["feature_types"][feature_type].append(key)

            summary["feature_shapes"][key] = list(tensor.shape)
            summary["feature_dtypes"][key] = str(tensor.dtype)

        return summary

    def clone(self) -> "AstroTensorDict":
        """Create a deep copy of the tensor dict."""
        cloned = super().clone()
        # Clone metadata
        cloned._metadata = self._metadata.copy()
        return cloned
