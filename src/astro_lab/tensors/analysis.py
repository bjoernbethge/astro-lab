"""
Unified Analysis TensorDict for AstroLab
========================================

Consolidated TensorDict for data analysis operations including feature extraction,
statistical computations, and clustering algorithms.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from tensordict import TensorDict

from .base import AstroTensorDict
from .mixins import FeatureExtractionMixin, NormalizationMixin, ValidationMixin


class AnalysisTensorDict(
    AstroTensorDict, NormalizationMixin, FeatureExtractionMixin, ValidationMixin
):
    """
    Unified TensorDict for data analysis operations.

    Structure:
    {
        "data": Tensor[N, D],         # Input data
        "features": Tensor[N, F],     # Extracted features (optional)
        "statistics": TensorDict,     # Computed statistics (optional)
        "labels": Tensor[N],          # Cluster labels (optional)
        "centroids": Tensor[K, D],    # Cluster centroids (optional)
        "meta": {
            "analysis_type": str,     # "features", "statistics", "clustering", "combined"
            "feature_names": List[str],
            "computed_stats": List[str],
            "algorithm": str,
            "n_clusters": int,
            "parameters": Dict,
        }
    }
    """

    def __init__(
        self,
        data: torch.Tensor,
        analysis_type: str = "features",
        feature_names: Optional[List[str]] = None,
        algorithm: str = "manual",
        n_clusters: int = 3,
        **kwargs,
    ):
        """
        Initialize AnalysisTensorDict.

        Args:
            data: [N, D] Input data tensor
            analysis_type: Type of analysis (
                "features", "statistics", "clustering", "combined"
            )
            feature_names: Names for extracted features
            algorithm: Analysis algorithm
            n_clusters: Number of clusters (for clustering)
        """
        n_objects, n_features = data.shape

        # Default feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        data_dict = {
            "data": data,
            "meta": {
                "analysis_type": analysis_type,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "algorithm": algorithm,
                "computed_stats": [],
                "parameters": {},
            },
        }

        # Add analysis-specific components
        if analysis_type in ["statistics", "combined"]:
            data_dict["statistics"] = TensorDict({}, batch_size=(n_objects,))

        if analysis_type in ["clustering", "combined"]:
            data_dict["labels"] = torch.zeros(n_objects, dtype=torch.long)
            data_dict["centroids"] = torch.zeros(n_clusters, n_features)
            data_dict["meta"]["n_clusters"] = n_clusters
            data_dict["meta"]["fitted"] = False

        super().__init__(data_dict, batch_size=(n_objects,), **kwargs)

    @property
    def analysis_type(self) -> str:
        """Type of analysis being performed."""
        return self["meta"]["analysis_type"]

    @property
    def feature_names(self) -> List[str]:
        """Names of features."""
        return self["meta"]["feature_names"]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self["meta"]["n_features"]

    @property
    def algorithm(self) -> str:
        """Analysis algorithm."""
        return self["meta"]["algorithm"]

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract analysis features from the TensorDict.

        Args:
            feature_types: Types of features to extract ('analysis', 'statistical', 'clustering')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted analysis features
        """

        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add analysis-specific computed features
        if feature_types is None or "analysis" in feature_types:
            # Basic data properties
            data = self["data"]
            features["data_mean"] = torch.mean(data, dim=-1)
            features["data_std"] = torch.std(data, dim=-1)
            features["data_range"] = (
                torch.max(data, dim=-1)[0] - torch.min(data, dim=-1)[0]
            )

        if feature_types is None or "statistical" in feature_types:
            # Add statistical features if available
            if "statistics" in self:
                for stat_name, stat_value in self["statistics"].items():
                    if (
                        isinstance(stat_value, torch.Tensor)
                        and stat_value.numel() == self.n_objects
                    ):
                        features[f"stat_{stat_name}"] = stat_value

        if feature_types is None or "clustering" in feature_types:
            # Add clustering features if available
            if "labels" in self and self["meta"]["fitted"]:
                features["cluster_labels"] = self["labels"].float()

                # Distance to cluster centroid
                data = self["data"]
                labels = self["labels"]
                centroids = self["centroids"]

                centroid_distances = torch.zeros(self.n_objects)
                for k in range(self["meta"]["n_clusters"]):
                    mask = labels == k
                    if torch.any(mask):
                        cluster_data = data[mask]
                        centroid = centroids[k]
                        distances = torch.norm(cluster_data - centroid, dim=-1)
                        centroid_distances[mask] = distances

                features["distance_to_centroid"] = centroid_distances

        return features

    def extract_analysis_features(
        self,
        method: str = "basic",
        include_statistical: bool = True,
        include_geometric: bool = True,
        include_spectral: bool = False,
        **kwargs,
    ) -> AnalysisTensorDict:
        """
        Extract features from data.

        Args:
            method: Feature extraction method
            include_statistical: Include statistical features
            include_geometric: Include geometric features
            include_spectral: Include spectral features

        Returns:
            Self with extracted features
        """
        features_list = []

        if method == "basic":
            features_list.append(
                self.extract_basic_features(
                    data_key="data",
                    include_statistical=include_statistical,
                    include_geometric=include_geometric,
                )
            )

        if include_spectral:
            features_list.append(
                self.extract_spectral_features(data_key="data", **kwargs)
            )

        if features_list:
            features = torch.cat(features_list, dim=-1)
            self["features"] = features

            # Update feature names
            feature_names = []
            if include_statistical:
                feature_names.extend(
                    ["mean", "std", "median", "min", "max", "variance"]
                )
            if include_geometric and self["data"].shape[-1] >= 2:
                if self["data"].shape[-1] == 2:
                    feature_names.extend(["magnitude", "angle"])
                elif self["data"].shape[-1] == 3:
                    feature_names.extend(["magnitude", "azimuth", "elevation"])
            if include_spectral:
                feature_names.extend(
                    [
                        "mean_power",
                        "power_std",
                        "peak_power",
                        "peak_freq",
                        "total_power",
                    ]
                )

            self["meta"]["feature_names"] = feature_names
            self["meta"]["n_features"] = len(feature_names)

        self.add_history("extract_features", method=method)
        return self

    def compute_statistics(
        self,
        include_basic: bool = True,
        include_percentiles: bool = True,
        percentiles: List[float] = [25, 50, 75],
        include_correlation: bool = False,
    ) -> AnalysisTensorDict:
        """
        Compute statistical measures.

        Args:
            include_basic: Include basic statistics
            include_percentiles: Include percentiles
            percentiles: List of percentiles to compute
            include_correlation: Include correlation matrix

        Returns:
            Self with computed statistics
        """
        data = self["data"]

        if include_basic:
            stats = {
                "mean": torch.mean(data, dim=-1),
                "std": torch.std(data, dim=-1),
                "min": torch.min(data, dim=-1)[0],
                "max": torch.max(data, dim=-1)[0],
                "median": torch.median(data, dim=-1)[0],
                "variance": torch.var(data, dim=-1),
            }
            self["statistics"].update(stats)
            self["meta"]["computed_stats"].extend(list(stats.keys()))

        if include_percentiles:
            for p in percentiles:
                percentile_name = f"p{int(p)}"
                percentile_value = torch.quantile(data, p / 100, dim=-1)
                self["statistics"][percentile_name] = percentile_value
                self["meta"]["computed_stats"].append(percentile_name)

        if include_correlation:
            # Center the data
            centered = data - torch.mean(data, dim=0)
            # Calculate correlation matrix
            cov = torch.mm(centered.T, centered) / (data.shape[0] - 1)
            std = torch.sqrt(torch.diag(cov))
            corr = cov / torch.outer(std, std)
            self["statistics"]["correlation_matrix"] = corr
            self["meta"]["computed_stats"].append("correlation_matrix")

        self.add_history("compute_statistics")
        return self

    def fit_clustering(
        self,
        algorithm: str = "kmeans",
        n_clusters: Optional[int] = None,
        max_iters: int = 100,
        tolerance: float = 1e-4,
        **kwargs,
    ) -> AnalysisTensorDict:
        """
        Perform clustering analysis.

        Args:
            algorithm: Clustering algorithm
            n_clusters: Number of clusters
            max_iters: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Self with fitted clustering
        """
        if n_clusters is not None:
            self["meta"]["n_clusters"] = n_clusters
            self["centroids"] = torch.zeros(n_clusters, self["data"].shape[-1])

        if algorithm == "kmeans":
            self._fit_kmeans(max_iters, tolerance)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")

        self["meta"]["algorithm"] = algorithm
        self["meta"]["fitted"] = True
        self["meta"]["parameters"].update(
            {
                "max_iters": max_iters,
                "tolerance": tolerance,
            }
        )

        self.add_history("fit_clustering", algorithm=algorithm)
        return self

    def _fit_kmeans(self, max_iters: int, tolerance: float):
        """Fit K-means clustering."""
        data = self["data"]
        n_clusters = self["meta"]["n_clusters"]

        # Initialize centroids randomly
        centroids = data[torch.randperm(data.shape[0])[:n_clusters]]

        for iteration in range(max_iters):
            # Assign points to nearest centroids
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if torch.sum(mask) > 0:
                    new_centroids[k] = torch.mean(data[mask], dim=0)
                else:
                    new_centroids[k] = centroids[k]

            # Check convergence
            if torch.norm(new_centroids - centroids) < tolerance:
                break

            centroids = new_centroids

        self["labels"] = labels
        self["centroids"] = centroids
        self["meta"]["parameters"]["final_iteration"] = iteration

    def predict_clusters(self, new_data: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster labels for new data.

        Args:
            new_data: [M, D] New data points

        Returns:
            [M] Cluster labels
        """
        if not self["meta"]["fitted"]:
            raise ValueError("Model must be fitted first")

        centroids = self["centroids"]
        distances = torch.cdist(new_data, centroids)
        return torch.argmin(distances, dim=1)

    def compute_inertia(self) -> torch.Tensor:
        """Compute clustering inertia (within-cluster sum of squares)."""
        if not self["meta"]["fitted"]:
            raise ValueError("Model must be fitted first")

        data = self["data"]
        labels = self["labels"]
        centroids = self["centroids"]

        inertia = torch.tensor(0.0)
        for k in range(self["meta"]["n_clusters"]):
            mask = labels == k
            if torch.sum(mask) > 0:
                cluster_data = data[mask]
                centroid = centroids[k]
                inertia += torch.sum((cluster_data - centroid) ** 2)

        return inertia

    def get_feature(self, name: str) -> torch.Tensor:
        """
        Get a specific feature by name.

        Args:
            name: Feature name

        Returns:
            Feature tensor
        """
        if "features" not in self:
            raise ValueError("No features available. Run extract_features() first.")

        if name not in self.feature_names:
            raise ValueError(
                f"Feature '{name}' not found. Available: {self.feature_names}"
            )

        idx = self.feature_names.index(name)
        return self["features"][..., idx]

    def get_statistic(self, name: str) -> torch.Tensor:
        """
        Get a specific statistic by name.

        Args:
            name: Statistic name

        Returns:
            Statistic tensor
        """
        if "statistics" not in self:
            raise ValueError("No statistics available. Run compute_statistics() first.")

        if name not in self["statistics"]:
            raise ValueError(
                f"Statistic '{name}' not found. "
                f"Available: {list(self['statistics'].keys())}"
            )

        return self["statistics"][name]

    def validate(self) -> bool:
        """Validate the analysis tensor."""
        required_keys = ["data"]

        if self.analysis_type in ["statistics", "combined"]:
            required_keys.append("statistics")

        if self.analysis_type in ["clustering", "combined"]:
            required_keys.extend(["labels", "centroids"])

        return self.validate_required_keys(required_keys)
