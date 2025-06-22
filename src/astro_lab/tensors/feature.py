"""
Feature Tensor for ML Feature Engineering
========================================

Specialized tensor for machine learning feature engineering operations
including scaling, imputation, outlier detection, and feature selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Core dependencies - should always be available
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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
    - Categorical encoding for survey flags
    """

    _metadata_fields = [
        "feature_names",
        "scalers",
        "imputers",
        "outlier_detectors",
        "feature_selectors",
        "preprocessing_history",
        "astronomical_priors",
        "survey_flags",
        "quality_masks",
    ]

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        feature_names: Optional[List[str]] = None,
        astronomical_priors: Optional[Dict[str, Any]] = None,
        survey_flags: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Initialize feature tensor.

        Args:
            data: Feature matrix [N, F]
            feature_names: Names of features
            astronomical_priors: Prior knowledge for astronomical features
            survey_flags: Survey-specific quality flags
        """
        # Validate shape
        tensor_data = torch.as_tensor(data, dtype=torch.float32)
        if tensor_data.dim() == 1:
            tensor_data = tensor_data.unsqueeze(1)  # Convert 1D to 2D
        elif tensor_data.dim() != 2:
            raise ValueError(
                f"FeatureTensor requires 2D data [N, F], got {tensor_data.shape}"
            )

        # Set default feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(tensor_data.shape[1])]
        elif len(feature_names) != tensor_data.shape[1]:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) doesn't match data columns ({tensor_data.shape[1]})"
            )

        # Set astronomical priors
        if astronomical_priors is None:
            astronomical_priors = self._get_default_priors()

        # Initialize metadata
        metadata = {
            "feature_names": feature_names,
            "astronomical_priors": astronomical_priors,
            "survey_flags": survey_flags or {},
            "scalers": {},
            "imputers": {},
            "outlier_detectors": {},
            "feature_selectors": {},
            "preprocessing_history": [],
            "quality_masks": {},
            "tensor_type": "feature",
        }
        metadata.update(kwargs)

        super().__init__(tensor_data, **metadata)

    def validate_finite_values(self) -> None:
        """Override to allow non-finite values for preprocessing."""
        # FeatureTensor allows non-finite values for preprocessing
        pass

    def _get_default_priors(self) -> Dict[str, Any]:
        """Get default astronomical priors."""
        return {
            "magnitude_range": {"min": 10.0, "max": 30.0},
            "color_range": {"min": -2.0, "max": 5.0},
            "parallax_range": {"min": 0.001, "max": 100.0},  # mas
            "proper_motion_range": {"min": -1000.0, "max": 1000.0},  # mas/yr
            "redshift_range": {"min": 0.0, "max": 10.0},
            "missing_value_codes": [99.0, 999.0, -999.0, float("inf"), float("-inf")],
            "magnitude_error_threshold": 0.5,  # mag
            "color_error_threshold": 0.1,  # mag
        }

    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return self.get_metadata("feature_names", [])

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._data.shape[1]

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return self._data.shape[0]

    def scale_features(
        self,
        method: str = "standard",
        feature_types: Optional[Dict[str, str]] = None,
        fit: bool = True,
    ) -> "FeatureTensor":
        """
        Scale features with astronomical-aware scaling.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'astronomical')
            feature_types: Map feature names to types ('magnitude', 'color', 'flux', etc.)
            fit: Whether to fit the scaler

        Returns:
            Scaled FeatureTensor
        """

        if feature_types is None:
            feature_types = self._detect_feature_types()

        scaled_data = self._data.clone()
        scalers = {}

        for i, feature_name in enumerate(self.feature_names):
            feature_type = feature_types.get(feature_name, "unknown")
            feature_data = self._data[:, i].cpu().numpy()

            # Choose scaler based on feature type
            if method == "astronomical":
                scaler = self._get_astronomical_scaler(feature_type)
            elif method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            # Fit and transform
            if fit:
                # Mask invalid values for fitting
                valid_mask = self._get_valid_mask(feature_data, feature_type)
                if valid_mask.sum() > 0:
                    scaler.fit(feature_data[valid_mask].reshape(-1, 1))
                    scalers[feature_name] = scaler

            if feature_name in scalers:
                scaled_feature = scaler.transform(feature_data.reshape(-1, 1)).flatten()
                scaled_data[:, i] = torch.from_numpy(scaled_feature)

        # Create new tensor with scaling history
        new_tensor = self._create_copy(scaled_data)
        new_tensor.update_metadata(scalers=scalers)
        new_tensor._add_to_history(f"scaled_features_{method}")

        return new_tensor

    def impute_missing_values(
        self,
        method: str = "astronomical",
        feature_types: Optional[Dict[str, str]] = None,
    ) -> "FeatureTensor":
        """
        Impute missing values with astronomical priors.

        Args:
            method: Imputation method ('astronomical', 'mean', 'median', 'knn')
            feature_types: Map feature names to types

        Returns:
            FeatureTensor with imputed values
        """
        # Check for KNN method - sklearn is required
        if method in ["knn"]:
            raise ImportError("sklearn required for advanced imputation")

        if feature_types is None:
            feature_types = self._detect_feature_types()

        imputed_data = self._data.clone()
        imputers = {}

        for i, feature_name in enumerate(self.feature_names):
            feature_type = feature_types.get(feature_name, "unknown")
            feature_data = self._data[:, i].cpu().numpy()

            # Detect missing values
            missing_mask = self._get_missing_mask(feature_data, feature_type)

            if missing_mask.sum() == 0:
                continue  # No missing values

            if method == "astronomical":
                imputed_values = self._astronomical_imputation(
                    feature_data, feature_type, missing_mask
                )
            elif method == "mean":
                imputer = SimpleImputer(strategy="mean")
                imputed_values = imputer.fit_transform(
                    feature_data.reshape(-1, 1)
                ).flatten()
                imputers[feature_name] = imputer
            elif method == "median":
                imputer = SimpleImputer(strategy="median")
                imputed_values = imputer.fit_transform(
                    feature_data.reshape(-1, 1)
                ).flatten()
                imputers[feature_name] = imputer
            elif method == "knn":
                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(
                    feature_data.reshape(-1, 1)
                ).flatten()
                imputers[feature_name] = imputer
            else:
                raise ValueError(f"Unknown imputation method: {method}")

            imputed_data[:, i] = torch.from_numpy(imputed_values)

        # Create new tensor
        new_tensor = self._create_copy(imputed_data)
        new_tensor.update_metadata(imputers=imputers)
        new_tensor._add_to_history(f"imputed_missing_{method}")

        return new_tensor

    def detect_outliers(
        self, method: str = "astronomical", contamination: float = 0.1
    ) -> torch.Tensor:
        """
        Detect outliers with astronomical knowledge.

        Args:
            method: Detection method ('astronomical', 'isolation_forest', 'statistical')
            contamination: Expected fraction of outliers

        Returns:
            Boolean mask (True = outlier)
        """
        if method == "astronomical":
            return self._astronomical_outlier_detection()
        elif method == "isolation_forest":
            detector = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = detector.fit_predict(self._data.cpu().numpy())
            return torch.from_numpy(outlier_labels == -1)
        elif method == "statistical":
            return self._statistical_outlier_detection()
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def select_features(
        self,
        method: str = "astronomical",
        k: int = 10,
        target: Optional[torch.Tensor] = None,
    ) -> "FeatureTensor":
        """
        Select most relevant features.

        Args:
            method: Selection method ('astronomical', 'variance', 'univariate')
            k: Number of features to select
            target: Target variable for supervised selection

        Returns:
            FeatureTensor with selected features
        """
        if method == "astronomical":
            selected_indices = self._astronomical_feature_selection(k)
        elif method == "variance":
            selector = VarianceThreshold()
            selector.fit(self._data.cpu().numpy())
            selected_indices = torch.from_numpy(selector.get_support())
        elif method == "univariate":
            if not SKLEARN_AVAILABLE or target is None:
                raise ImportError(
                    "sklearn and target required for univariate selection"
                )
            selector = SelectKBest(f_classif, k=k)
            selector.fit(self._data.cpu().numpy(), target.cpu().numpy())
            selected_indices = torch.from_numpy(selector.get_support())
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        # Select features
        selected_data = self._data[:, selected_indices]
        selected_names = [
            self.feature_names[i]
            for i in range(len(self.feature_names))
            if selected_indices[i]
        ]

        # Create new tensor with correct feature names
        new_tensor = FeatureTensor(selected_data, feature_names=selected_names)
        new_tensor.update_metadata(
            **{k: v for k, v in self._metadata.items() if k != "feature_names"}
        )
        new_tensor._add_to_history(f"selected_features_{method}_{k}")

        return new_tensor

    def compute_colors(self, magnitude_bands: List[str]) -> "FeatureTensor":
        """
        Compute astronomical colors from magnitudes.

        Args:
            magnitude_bands: List of magnitude band names

        Returns:
            FeatureTensor with color features
        """
        color_data = []
        color_names = []

        # Find magnitude columns
        mag_indices = {}
        for band in magnitude_bands:
            for i, name in enumerate(self.feature_names):
                if band.lower() in name.lower() and "mag" in name.lower():
                    mag_indices[band] = i
                    break

        # Compute colors
        for i in range(len(magnitude_bands) - 1):
            band1, band2 = magnitude_bands[i], magnitude_bands[i + 1]
            if band1 in mag_indices and band2 in mag_indices:
                color = (
                    self._data[:, mag_indices[band1]]
                    - self._data[:, mag_indices[band2]]
                )
                color_data.append(color)
                color_names.append(f"{band1}_{band2}_color")

        if not color_data:
            raise ValueError("No magnitude bands found for color computation")

        # Combine with original data
        color_tensor = torch.stack(color_data, dim=1)
        combined_data = torch.cat([self._data, color_tensor], dim=1)
        combined_names = self.feature_names + color_names

        # Create new tensor with correct feature names
        new_tensor = FeatureTensor(combined_data, feature_names=combined_names)
        new_tensor.update_metadata(
            **{k: v for k, v in self._metadata.items() if k != "feature_names"}
        )
        new_tensor._add_to_history("computed_colors")

        return new_tensor

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive feature statistics."""
        stats = {}

        for i, name in enumerate(self.feature_names):
            feature_data = self._data[:, i]
            valid_mask = torch.isfinite(feature_data)
            valid_data = feature_data[valid_mask]

            if len(valid_data) > 0:
                stats[name] = {
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "median": float(valid_data.median()),
                    "q25": float(valid_data.quantile(0.25)),
                    "q75": float(valid_data.quantile(0.75)),
                    "missing_fraction": float((~valid_mask).sum()) / len(feature_data),
                    "n_valid": int(valid_mask.sum()),
                }
            else:
                stats[name] = {"error": "No valid data"}

        return stats

    # Helper methods
    def _detect_feature_types(self) -> Dict[str, str]:
        """Automatically detect feature types from names."""
        types = {}
        for name in self.feature_names:
            name_lower = name.lower()
            if "mag" in name_lower:
                types[name] = "magnitude"
            elif (
                "color" in name_lower
                or "_" in name_lower
                and any(band in name_lower for band in ["u", "g", "r", "i", "z"])
            ):
                types[name] = "color"
            elif "flux" in name_lower:
                types[name] = "flux"
            elif "parallax" in name_lower:
                types[name] = "parallax"
            elif "pm" in name_lower or "proper_motion" in name_lower:
                types[name] = "proper_motion"
            elif "ra" in name_lower or "dec" in name_lower:
                types[name] = "coordinate"
            elif "z" == name_lower or "redshift" in name_lower:
                types[name] = "redshift"
            else:
                types[name] = "unknown"
        return types

    def _get_astronomical_scaler(self, feature_type: str):
        """Get appropriate scaler for astronomical feature type."""
        if feature_type == "magnitude":
            return MinMaxScaler(feature_range=(0, 1))  # Magnitudes have natural bounds
        elif feature_type == "color":
            return StandardScaler()  # Colors can be negative
        elif feature_type == "flux":
            return RobustScaler()  # Fluxes have outliers
        else:
            return StandardScaler()

    def _get_valid_mask(self, data: np.ndarray, feature_type: str) -> np.ndarray:
        """Get mask for valid astronomical values."""
        priors = self.get_metadata("astronomical_priors", {})
        missing_codes = priors.get("missing_value_codes", [99.0, 999.0, -999.0])

        valid_mask = np.isfinite(data)
        for code in missing_codes:
            valid_mask &= data != code

        return valid_mask

    def _get_missing_mask(self, data: np.ndarray, feature_type: str) -> np.ndarray:
        """Get mask for missing values."""
        return ~self._get_valid_mask(data, feature_type)

    def _astronomical_imputation(
        self, data: np.ndarray, feature_type: str, missing_mask: np.ndarray
    ) -> np.ndarray:
        """Impute missing values with astronomical priors."""
        imputed_data = data.copy()
        priors = self.get_metadata("astronomical_priors", {})

        if feature_type == "magnitude":
            # Use magnitude limit for missing magnitudes
            mag_range = priors.get("magnitude_range", {"min": 10.0, "max": 30.0})
            imputed_data[missing_mask] = mag_range["max"]  # Faint limit
        elif feature_type == "color":
            # Use median color for missing colors
            valid_data = data[~missing_mask]
            if len(valid_data) > 0:
                imputed_data[missing_mask] = np.median(valid_data)
        elif feature_type == "parallax":
            # Use small positive value for missing parallax
            imputed_data[missing_mask] = 0.001  # 1000 pc
        else:
            # Use median for other types
            valid_data = data[~missing_mask]
            if len(valid_data) > 0:
                imputed_data[missing_mask] = np.median(valid_data)

        return imputed_data

    def _astronomical_outlier_detection(self) -> torch.Tensor:
        """Detect outliers using astronomical knowledge."""
        outlier_mask = torch.zeros(self.n_objects, dtype=torch.bool)
        priors = self.get_metadata("astronomical_priors", {})

        for i, name in enumerate(self.feature_names):
            feature_data = self._data[:, i]
            feature_type = self._detect_feature_types()[name]

            if feature_type == "magnitude":
                mag_range = priors.get("magnitude_range", {"min": 10.0, "max": 30.0})
                outlier_mask |= (feature_data < mag_range["min"]) | (
                    feature_data > mag_range["max"]
                )
            elif feature_type == "color":
                color_range = priors.get("color_range", {"min": -2.0, "max": 5.0})
                outlier_mask |= (feature_data < color_range["min"]) | (
                    feature_data > color_range["max"]
                )

        return outlier_mask

    def _statistical_outlier_detection(self) -> torch.Tensor:
        """Detect outliers using statistical methods."""
        outlier_mask = torch.zeros(self.n_objects, dtype=torch.bool)

        for i in range(self.n_features):
            feature_data = self._data[:, i]
            valid_mask = torch.isfinite(feature_data)

            if valid_mask.sum() > 10:  # Need enough data
                valid_data = feature_data[valid_mask]
                q1, q3 = valid_data.quantile(0.25), valid_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                feature_outliers = (feature_data < lower_bound) | (
                    feature_data > upper_bound
                )
                outlier_mask |= feature_outliers

        return outlier_mask

    def _astronomical_feature_selection(self, k: int) -> torch.Tensor:
        """Select features based on astronomical importance."""
        importance_scores = torch.zeros(self.n_features)
        feature_types = self._detect_feature_types()

        # Score based on astronomical relevance
        for i, name in enumerate(self.feature_names):
            feature_type = feature_types[name]
            if feature_type == "magnitude":
                importance_scores[i] = 10.0  # High importance
            elif feature_type == "color":
                importance_scores[i] = 8.0
            elif feature_type == "coordinate":
                importance_scores[i] = 9.0
            elif feature_type == "parallax":
                importance_scores[i] = 7.0
            else:
                importance_scores[i] = 5.0

        # Select top k features
        _, top_indices = importance_scores.topk(min(k, self.n_features))
        selected_mask = torch.zeros(self.n_features, dtype=torch.bool)
        selected_mask[top_indices] = True

        return selected_mask

    def _create_copy(self, new_data: torch.Tensor) -> "FeatureTensor":
        """Create a copy with new data but same metadata."""
        metadata = self._metadata.copy()
        # Remove feature_names from metadata since they might not match new data
        metadata.pop("feature_names", None)
        return FeatureTensor(new_data, **metadata)

    def _add_to_history(self, operation: str):
        """Add operation to preprocessing history."""
        history = self.get_metadata("preprocessing_history", [])
        history.append(operation)
        self.update_metadata(preprocessing_history=history)

    def __repr__(self) -> str:
        return (
            f"FeatureTensor(objects={self.n_objects}, features={self.n_features}, "
            f"history={len(self.get_metadata('preprocessing_history', []))} ops)"
        )
