"""
TensorDict-basierte Feature-Extraktion und Datenverarbeitung
==========================================================

Umstellung der Feature-, Statistics- und Clustering-Klassen auf TensorDict-Architektur.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict
import numpy as np

from .tensordict_astro import AstroTensorDict


class FeatureTensorDict(AstroTensorDict):
    """
    TensorDict für Feature-Extraktion und -Verarbeitung.

    Struktur:
    {
        "features": Tensor[N, F],     # Extrahierte Features
        "raw_data": TensorDict,       # Original-Daten (optional)
        "meta": {
            "feature_names": List[str],
            "n_features": int,
            "extraction_method": str,
            "normalization": str,
        }
    }
    """

    def __init__(self, features: torch.Tensor, feature_names: List[str], 
                 raw_data: Optional[AstroTensorDict] = None,
                 extraction_method: str = "manual",
                 normalization: str = "none", **kwargs):
        """
        Initialisiert FeatureTensorDict.

        Args:
            features: [N, F] Feature-Matrix
            feature_names: Namen der Features
            raw_data: Original-Daten (optional)
            extraction_method: Methode der Feature-Extraktion
            normalization: Art der Normalisierung
        """
        if len(feature_names) != features.shape[-1]:
            raise ValueError(f"Feature names length {len(feature_names)} doesn't match features shape {features.shape[-1]}")

        n_objects = features.shape[0]

        data = {
            "features": features,
            "meta": TensorDict({
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "extraction_method": extraction_method,
                "normalization": normalization,
            }, batch_size=(n_objects,))
        }

        if raw_data is not None:
            data["raw_data"] = raw_data

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def feature_names(self) -> List[str]:
        """Namen der Features."""
        return self["meta", "feature_names"]

    @property
    def n_features(self) -> int:
        """Anzahl der Features."""
        return self["meta", "n_features"]

    def get_feature(self, name: str) -> torch.Tensor:
        """
        Holt ein spezifisches Feature.

        Args:
            name: Name des Features

        Returns:
            Feature-Tensor
        """
        if name not in self.feature_names:
            raise ValueError(f"Feature '{name}' not found. Available: {self.feature_names}")

        idx = self.feature_names.index(name)
        return self["features"][..., idx]

    def normalize(self, method: str = "standard") -> FeatureTensorDict:
        """
        Normalisiert die Features.

        Args:
            method: Normalisierungsmethode ('standard', 'minmax', 'robust')

        Returns:
            Normalisierter FeatureTensorDict
        """
        features = self["features"]

        if method == "standard":
            # Z-Score Normalisierung
            mean = torch.mean(features, dim=0)
            std = torch.std(features, dim=0)
            normalized = (features - mean) / (std + 1e-8)
        elif method == "minmax":
            # Min-Max Normalisierung
            min_vals = torch.min(features, dim=0)[0]
            max_vals = torch.max(features, dim=0)[0]
            normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)
        elif method == "robust":
            # Robust Normalisierung (Median/MAD)
            median = torch.median(features, dim=0)[0]
            mad = torch.median(torch.abs(features - median), dim=0)[0]
            normalized = (features - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return FeatureTensorDict(
            features=normalized,
            feature_names=self.feature_names,
            raw_data=self.get("raw_data", None),
            extraction_method=self["meta", "extraction_method"],
            normalization=method
        )

    def select_features(self, indices: Union[List[int], List[str]]) -> FeatureTensorDict:
        """
        Wählt spezifische Features aus.

        Args:
            indices: Feature-Indizes oder -Namen

        Returns:
            FeatureTensorDict mit ausgewählten Features
        """
        if isinstance(indices[0], str):
            # Namen zu Indizes konvertieren
            indices = [self.feature_names.index(name) for name in indices]

        selected_features = self["features"][..., indices]
        selected_names = [self.feature_names[i] for i in indices]

        return FeatureTensorDict(
            features=selected_features,
            feature_names=selected_names,
            raw_data=self.get("raw_data", None),
            extraction_method=self["meta", "extraction_method"],
            normalization=self["meta", "normalization"]
        )


class StatisticsTensorDict(AstroTensorDict):
    """
    TensorDict für statistische Analysen.

    Struktur:
    {
        "data": Tensor[N, D],         # Input-Daten
        "statistics": TensorDict,     # Berechnete Statistiken
        "meta": {
            "computed_stats": List[str],
            "method": str,
        }
    }
    """

    def __init__(self, data: torch.Tensor, **kwargs):
        """
        Initialisiert StatisticsTensorDict.

        Args:
            data: [N, D] Input-Daten
        """
        n_objects = data.shape[0]

        tensor_data = {
            "data": data,
            "statistics": TensorDict({}, batch_size=(n_objects,)),
            "meta": TensorDict({
                "computed_stats": [],
                "method": "pytorch",
            }, batch_size=(n_objects,))
        }

        super().__init__(tensor_data, batch_size=(n_objects,), **kwargs)

    def compute_basic_stats(self) -> StatisticsTensorDict:
        """Berechnet grundlegende Statistiken."""
        data = self["data"]

        stats = {
            "mean": torch.mean(data, dim=-1),
            "std": torch.std(data, dim=-1),
            "min": torch.min(data, dim=-1)[0],
            "max": torch.max(data, dim=-1)[0],
            "median": torch.median(data, dim=-1)[0],
        }

        self["statistics"].update(stats)
        self["meta", "computed_stats"].extend(list(stats.keys()))

        return self

    def compute_percentiles(self, percentiles: List[float] = [25, 50, 75]) -> StatisticsTensorDict:
        """
        Berechnet Perzentile.

        Args:
            percentiles: Liste der Perzentile
        """
        data = self["data"]

        for p in percentiles:
            percentile_name = f"p{int(p)}"
            percentile_value = torch.quantile(data, p/100, dim=-1)
            self["statistics", percentile_name] = percentile_value

        self["meta", "computed_stats"].extend([f"p{int(p)}" for p in percentiles])

        return self

    def compute_correlation_matrix(self) -> torch.Tensor:
        """Berechnet Korrelationsmatrix."""
        data = self["data"]

        # Zentriere die Daten
        centered = data - torch.mean(data, dim=0)

        # Berechne Korrelationsmatrix
        cov = torch.mm(centered.T, centered) / (data.shape[0] - 1)
        std = torch.sqrt(torch.diag(cov))
        corr = cov / torch.outer(std, std)

        self["statistics", "correlation_matrix"] = corr
        self["meta", "computed_stats"].append("correlation_matrix")

        return corr


class ClusteringTensorDict(AstroTensorDict):
    """
    TensorDict für Clustering-Algorithmen.

    Struktur:
    {
        "data": Tensor[N, D],         # Input-Daten
        "labels": Tensor[N],          # Cluster-Labels
        "centroids": Tensor[K, D],    # Cluster-Zentren
        "meta": {
            "algorithm": str,
            "n_clusters": int,
            "parameters": Dict,
        }
    }
    """

    def __init__(self, data: torch.Tensor, n_clusters: int = 3, 
                 algorithm: str = "kmeans", **kwargs):
        """
        Initialisiert ClusteringTensorDict.

        Args:
            data: [N, D] Input-Daten
            n_clusters: Anzahl der Cluster
            algorithm: Clustering-Algorithmus
        """
        n_objects, n_features = data.shape

        tensor_data = {
            "data": data,
            "labels": torch.zeros(n_objects, dtype=torch.long),
            "centroids": torch.zeros(n_clusters, n_features),
            "meta": TensorDict({
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "parameters": {},
                "fitted": False,
            }, batch_size=(n_objects,))
        }

        super().__init__(tensor_data, batch_size=(n_objects,), **kwargs)

    def fit_kmeans(self, max_iters: int = 100, tolerance: float = 1e-4) -> ClusteringTensorDict:
        """
        Führt K-Means Clustering durch.

        Args:
            max_iters: Maximale Iterationen
            tolerance: Konvergenz-Toleranz
        """
        data = self["data"]
        n_clusters = self["meta", "n_clusters"]

        # Initialisiere Zentroide zufällig
        centroids = data[torch.randperm(data.shape[0])[:n_clusters]]

        for iteration in range(max_iters):
            # Weise Punkte zu nächsten Zentroiden zu
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)

            # Aktualisiere Zentroide
            new_centroids = torch.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if torch.sum(mask) > 0:
                    new_centroids[k] = torch.mean(data[mask], dim=0)
                else:
                    new_centroids[k] = centroids[k]

            # Prüfe Konvergenz
            if torch.norm(new_centroids - centroids) < tolerance:
                break

            centroids = new_centroids

        self["labels"] = labels
        self["centroids"] = centroids
        self["meta", "fitted"] = True
        self["meta", "parameters"] = {
            "max_iters": max_iters,
            "tolerance": tolerance,
            "final_iteration": iteration
        }

        return self

    def compute_inertia(self) -> torch.Tensor:
        """Berechnet die Inertia (Within-Cluster Sum of Squares)."""
        if not self["meta", "fitted"]:
            raise ValueError("Model must be fitted first")

        data = self["data"]
        labels = self["labels"]
        centroids = self["centroids"]

        inertia = 0.0
        for k in range(self["meta", "n_clusters"]):
            mask = labels == k
            if torch.sum(mask) > 0:
                cluster_data = data[mask]
                centroid = centroids[k]
                inertia += torch.sum((cluster_data - centroid) ** 2)

        return inertia

    def predict(self, new_data: torch.Tensor) -> torch.Tensor:
        """
        Vorhersage für neue Daten.

        Args:
            new_data: [M, D] Neue Datenpunkte

        Returns:
            [M] Cluster-Labels
        """
        if not self["meta", "fitted"]:
            raise ValueError("Model must be fitted first")

        centroids = self["centroids"]
        distances = torch.cdist(new_data, centroids)
        return torch.argmin(distances, dim=1)
