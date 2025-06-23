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

    def scale_features(self, method: str = "standard") -> "FeatureTensor":
        """Scales features using a specified method."""
        scaler_map = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        if method not in scaler_map:
            raise ValueError(f"Unknown scaling method: {method}")

        scaler = scaler_map[method]()
        # sklearn transformers expect numpy arrays
        scaled_data_np = scaler.fit_transform(self.data.cpu().numpy())
        scaled_data = torch.tensor(scaled_data_np, dtype=self.dtype, device=self.device)

        new_tensor = self.model_copy(update={"data": scaled_data})
        new_tensor.add_history_entry(f"scaled_features (method: {method})", scaler=scaler.__class__.__name__)
        return new_tensor

    def impute_missing_values(self, method: str = "mean", **kwargs) -> "FeatureTensor":
        """Imputes missing values (NaNs) using a specified method."""
        imputer_map = {
            "mean": SimpleImputer,
            "median": SimpleImputer,
            "knn": KNNImputer,
        }
        if method not in imputer_map:
            raise ValueError(f"Unknown imputation method: {method}")

        imputer = imputer_map[method](strategy=method if method != 'knn' else None, **kwargs)
        imputed_data_np = imputer.fit_transform(self.data.cpu().numpy())
        imputed_data = torch.tensor(imputed_data_np, dtype=self.dtype, device=self.device)

        new_tensor = self.model_copy(update={"data": imputed_data})
        new_tensor.add_history_entry(f"imputed_missing (method: {method})")
        return new_tensor

    def detect_outliers(self, method: str = "isolation_forest", **kwargs) -> torch.Tensor:
        """Detects outliers and returns a boolean mask."""
        if method == "isolation_forest":
            detector = IsolationForest(**kwargs)
            predictions = detector.fit_predict(self.data.cpu().numpy())
            return torch.tensor(predictions == -1, dtype=torch.bool, device=self.device)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def select_features(self, method: str = "variance", k: int = 10, target: Optional[torch.Tensor] = None) -> "FeatureTensor":
        """Selects features based on a specified method."""
        if method == "variance":
            selector = VarianceThreshold()
            selector.fit(self.data.cpu().numpy())
            selected_indices = np.where(selector.variances_ > 0)[0][:k]
        elif method == "k_best":
            if target is None:
                raise ValueError("Target tensor must be provided for k-best selection")
            selector = SelectKBest(f_classif, k=k)
            selector.fit(self.data.cpu().numpy(), target.cpu().numpy())
            selected_indices = selector.get_support(indices=True)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        new_data = self.data[:, selected_indices]
        new_feature_names = [self.feature_names[i] for i in selected_indices]
        
        new_tensor = self.model_copy(update={"data": new_data, "feature_names": new_feature_names})
        new_tensor.add_history_entry(f"selected_features (method: {method}, k: {k})")
        return new_tensor

    def compute_colors(self, magnitude_bands: List[str]) -> "FeatureTensor":
        """Computes colors from magnitude bands (e.g., 'g-r', 'r-i')."""
        band_indices = {band: i for i, band in enumerate(self.feature_names) if band in magnitude_bands}
        
        if len(band_indices) < 2:
            raise ValueError("At least two magnitude bands must be provided and found in feature_names")

        colors = []
        color_names = []
        
        # Create colors from adjacent bands in the provided list
        for i in range(len(magnitude_bands) - 1):
            band1_name = magnitude_bands[i]
            band2_name = magnitude_bands[i+1]
            if band1_name in band_indices and band2_name in band_indices:
                idx1 = band_indices[band1_name]
                idx2 = band_indices[band2_name]
                color = self.data[:, idx1] - self.data[:, idx2]
                colors.append(color.unsqueeze(1))
                color_names.append(f"{band1_name}-{band2_name}")

        if not colors:
            raise ValueError("Could not compute any colors from the provided bands.")

        new_color_data = torch.cat(colors, dim=1)
        
        # Create a new FeatureTensor for the colors
        new_tensor = self.__class__(data=new_color_data, feature_names=color_names)
        new_tensor.add_history_entry(f"computed_colors (bands: {magnitude_bands})")
        return new_tensor

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
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, objects={self.n_objects}, features={self.n_features})"
        
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
