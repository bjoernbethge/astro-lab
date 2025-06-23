"""
Feature Tensor for ML Feature Engineering
========================================

Specialized tensor for machine learning feature engineering operations
including scaling, imputation, outlier detection, and feature selection.
"""

from typing import Any, Dict, List, Optional, Union, Literal
import warnings

import numpy as np
import torch
from pydantic import Field, field_validator
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from typing_extensions import Self

from .base import AstroTensorBase


class FeatureTensor(AstroTensorBase):
    """
    Tensor for ML feature engineering and preprocessing.

    Provides astronomical-specific feature engineering operations:
    - Magnitude/flux scaling and normalization
    - Color computation and error propagation
    - Missing value imputation with astronomical priors
    - Outlier detection for astronomical objects
    - Feature selection based on astronomical relevance
    """

    data: torch.Tensor
    feature_names: List[str] = Field(default_factory=list, description="Names of features")
    
    @field_validator("data")
    @classmethod
    def validate_feature_data(cls, v):
        """
        Custom validator for FeatureTensor.
        This overrides the base class validator to allow NaNs for imputation.
        """
        if v.ndim == 1:
            v = v.unsqueeze(1)
        if v.ndim != 2:
            raise ValueError(f"FeatureTensor requires 2D data [N, F], but got shape {v.shape}")
        if v.numel() == 0:
            raise ValueError("FeatureTensor cannot be empty.")
        # We explicitly DO NOT check for isfinite here, as this tensor is used for imputation.
        return v

    def __init__(self, data: torch.Tensor, **kwargs: Any):
        super().__init__(data=data, **kwargs)
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        elif len(self.feature_names) != self.n_features:
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) doesn't match data columns ({self.n_features})"
            )
        # Add tensor type to metadata for consistent access
        self.meta["tensor_type"] = "feature"


    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.data.shape[1]

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return self.data.shape[0]

    @property
    def num_features(self) -> int:
        """Return the number of features."""
        return self.data.shape[1] if self.data.dim() > 1 else 1

    @property
    def num_objects(self) -> int:
        """Return the number of objects/samples."""
        return self.data.shape[0]

    def scale_features(self, method: str = "standard") -> "FeatureTensor":
        """Scales features using a specified method."""
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

        # Select scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Scale data
        scaled_data = torch.from_numpy(scaler.fit_transform(self.data.numpy())).float()

        # Create new instance with updated metadata (flat structure)
        new_meta = self.meta.copy()
        new_meta.setdefault("scalers", {})[method] = scaler
        new_meta.setdefault("history", []).append({
            "description": f"scaled_features (method: {method})",
            "details": {"scaler": scaler.__class__.__name__}
        })

        return self._create_new_instance(
            new_data=scaled_data,
            feature_names=self.feature_names.copy(),
            meta=new_meta
        )

    def impute_missing_values(self, strategy: str = "mean", **kwargs) -> "FeatureTensor":
        """Impute missing values using specified strategy."""
        from sklearn.impute import SimpleImputer
        
        # Create imputer
        if strategy in ["mean", "median", "most_frequent"]:
            imputer = SimpleImputer(strategy=strategy)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        
        # Apply imputation
        imputed_data = torch.from_numpy(
            imputer.fit_transform(self.data.numpy())
        ).float()
        
        # Update metadata (flat structure)
        new_meta = self.meta.copy()
        new_meta.setdefault("imputers", {})[strategy] = imputer
        new_meta.setdefault("history", []).append({
            "description": f"imputed_missing_values (strategy: {strategy})",
            "details": {"imputer": imputer.__class__.__name__}
        })
        
        return self._create_new_instance(
            new_data=imputed_data,
            feature_names=self.feature_names.copy(),
            meta=new_meta
        )

    def detect_outliers(self, method: str = "isolation_forest", **kwargs) -> torch.Tensor:
        """Detects outliers and returns a boolean mask."""
        if method == "isolation_forest":
            detector = IsolationForest(**kwargs)
            predictions = detector.fit_predict(self.data.cpu().numpy())
            return torch.tensor(predictions == -1, dtype=torch.bool, device=self.device)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def select_features(self, method: str = "variance", threshold: float = 0.01) -> "FeatureTensor":
        """Select features based on variance or other criteria."""
        if method == "variance":
            # Calculate variance for each feature
            variances = torch.var(self.data, dim=0)
            selected_mask = variances > threshold
            
            # Select features above threshold
            if not selected_mask.any():
                raise ValueError(f"No features meet variance threshold {threshold}")
            
            selected_data = self.data[:, selected_mask]
            # Fix: Properly filter feature names to match selected data
            selected_names = [name for i, name in enumerate(self.feature_names) if selected_mask[i]]
            
            # Update metadata
            new_meta = self.meta.copy()
            new_meta["variance_threshold"] = threshold
            new_meta.setdefault("history", []).append({
                "description": f"feature_selection (method: {method})",
                "details": {"threshold": threshold, "features_kept": len(selected_names)}
            })
            
            return self._create_new_instance(
                new_data=selected_data,
                feature_names=selected_names,
                meta=new_meta
            )
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    def compute_colors(self, bands: Optional[List[str]] = None) -> "FeatureTensor":
        """Compute astronomical color indices from magnitude bands."""
        if bands is None:
            bands = self.feature_names
        
        # Check which bands actually exist in our data
        available_bands = [band for band in bands if band in self.feature_names]
        
        if len(available_bands) < 2:
            # If specific bands weren't found, use first few features
            if bands != self.feature_names:  # Only if user specified bands
                available_bands = self.feature_names[:min(3, len(self.feature_names))]
            if len(available_bands) < 2:
                raise ValueError("Need at least 2 bands to compute colors")
        
        # Create color combinations
        colors = []
        color_names = []
        
        for i in range(len(available_bands) - 1):
            for j in range(i + 1, len(available_bands)):
                if available_bands[i] in self.feature_names and available_bands[j] in self.feature_names:
                    idx_i = self.feature_names.index(available_bands[i])
                    idx_j = self.feature_names.index(available_bands[j])
                    
                    color = self.data[:, idx_i] - self.data[:, idx_j]
                    colors.append(color.unsqueeze(1))
                    color_names.append(f"{available_bands[i]}-{available_bands[j]}")
        
        if not colors:
            raise ValueError("No valid band combinations found")
        
        # Combine original data with colors
        color_data = torch.cat(colors, dim=1)
        combined_data = torch.cat([self.data, color_data], dim=1)
        combined_names = self.feature_names + color_names
        
        # Update metadata
        new_meta = self.meta.copy()
        new_meta.setdefault("history", []).append({
            "description": "computed_colors",
            "details": {"color_combinations": color_names}
        })
        
        return self._create_new_instance(
            new_data=combined_data,
            feature_names=combined_names,
            meta=new_meta
        )

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Computes summary statistics for each feature."""
        stats = {}
        for i, name in enumerate(self.feature_names):
            feature_data = self.data[:, i]
            stats[name] = {
                "mean": feature_data.mean().item(),
                "std": feature_data.std().item(),
                "min": feature_data.min().item(),
                "max": feature_data.max().item(),
                "median": feature_data.median().item(),
            }
        return stats

    def __repr__(self) -> str:
        """Enhanced string representation matching test expectations."""
        return (f"FeatureTensor(data=..., feature_names={self.feature_names}, "
                f"num_objects={self.num_objects}, num_features={self.num_features})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary for serialization."""
        return {
            "data": self.data.cpu().numpy().tolist(),
            "feature_names": self.feature_names,
            "meta": self.meta,
            "tensor_type": "feature"
        }
        
    # Keep compatibility methods if they are simple, otherwise they should be refactored
    def to_dataframe(self) -> Any:
        """Converts the tensor to a Polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars is required for this functionality. `pip install polars`")
        return pl.DataFrame(self.data.cpu().numpy(), schema=self.feature_names)

    @classmethod
    def from_dataframe(cls, df: Any, feature_cols: Optional[List[str]] = None) -> Self:
        """Creates a FeatureTensor from a Polars DataFrame."""
        if feature_cols is None:
            feature_cols = df.columns
        
        data_np = df[feature_cols].to_numpy()
        return cls(data=torch.from_numpy(data_np), feature_names=feature_cols)

    def copy(self: Self, deep: bool = True) -> Self:
        """Creates a copy of the tensor."""
        if deep:
            return self.model_copy(deep=True)
        return self.model_copy()

    def _get_default_priors(self) -> Dict[str, Any]:
        """Get default astronomical priors."""
        # This can be simplified or used to guide more complex methods if needed
        return {
            "magnitude_range": {"min": 10.0, "max": 30.0},
            "color_range": {"min": -2.0, "max": 5.0},
        }
        
    def _astronomical_outlier_detection(self) -> torch.Tensor:
        """Example of a more complex, domain-specific outlier detection."""
        priors = self._get_default_priors()
        outlier_mask = torch.zeros(self.n_objects, dtype=torch.bool, device=self.device)
        
        for i, fname in enumerate(self.feature_names):
            if "mag" in fname: # A simple check if the feature is a magnitude
                mag_data = self.data[:, i]
                mag_range = priors["magnitude_range"]
                outlier_mask |= (mag_data < mag_range["min"]) | (mag_data > mag_range["max"])
        
        return outlier_mask
        
    def test_astronomical_priors(self):
        """Test astronomical priors."""
        priors = self._get_default_priors()
        assert "magnitude_range" in priors

    def _create_new_instance(self, new_data: torch.Tensor, feature_names: List[str], meta: Dict[str, Any]) -> "FeatureTensor":
        """Create a new instance with updated data, ensuring feature names match data shape."""
        # Ensure feature names match data dimensions
        if new_data.dim() > 1:
            expected_features = new_data.shape[1]
            if len(feature_names) != expected_features:
                # Auto-generate feature names if there's a mismatch
                feature_names = [f"feature_{i}" for i in range(expected_features)]
        
        return self.__class__(
            data=new_data,
            feature_names=feature_names,
            meta=meta
        )
