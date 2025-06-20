"""
Advanced data processing using AstroLab tensor system.

This module provides enhanced data processing capabilities using the tensor system,
including feature engineering, clustering, and statistical analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

from ..tensors import ClusteringTensor, FeatureTensor, StatisticsTensor, SurveyTensor

# Import SimulationTensor as well
try:
    from ..tensors import SimulationTensor

    SIMULATION_TENSOR_AVAILABLE = True
except ImportError:
    SIMULATION_TENSOR_AVAILABLE = False
    SimulationTensor = None


class SimpleProcessingConfig(BaseModel):
    """Simplified configuration for data processing."""

    # Basic options
    device: str = Field(default="cpu", description="Device to use for processing")
    batch_size: int = Field(default=1000, description="Batch size for processing")

    # Feature engineering
    enable_feature_engineering: bool = Field(
        default=True, description="Enable feature engineering"
    )

    # Clustering
    enable_clustering: bool = Field(default=True, description="Enable clustering")
    dbscan_eps: float = Field(default=0.1, description="DBSCAN epsilon")
    dbscan_min_samples: int = Field(default=5, description="DBSCAN min samples")

    # Statistics
    enable_statistics: bool = Field(default=True, description="Enable statistics")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


class SimpleAstroProcessor:
    """Simplified astronomical data processor using tensor operations."""

    def __init__(self, config: Optional[SimpleProcessingConfig] = None):
        """Initialize the processor."""
        self.config = config or SimpleProcessingConfig()

    def process(
        self, survey_tensor: Union[SurveyTensor, "SimulationTensor"]
    ) -> Dict[str, Any]:
        """Process survey tensor or simulation tensor with simplified operations."""
        results = {"input_tensor": survey_tensor}

        # Handle different tensor types
        if hasattr(survey_tensor, "simulation_name"):
            # SimulationTensor
            tensor_name = survey_tensor.simulation_name
            tensor_type = "simulation"
        elif hasattr(survey_tensor, "survey_name"):
            # SurveyTensor
            tensor_name = survey_tensor.survey_name
            tensor_type = "survey"
        else:
            # Fallback
            tensor_name = "unknown"
            tensor_type = "unknown"

        print(f"🔬 Processing {tensor_name} ({tensor_type})...")
        print(f"   📊 Input shape: {survey_tensor._data.shape}")

        try:
            # Feature Engineering
            if self.config.enable_feature_engineering:
                print("  🔧 Feature engineering...")
                feature_results = self._apply_features(survey_tensor)
                results.update(feature_results)

            # Clustering
            if self.config.enable_clustering:
                print("  🎯 Clustering...")
                cluster_results = self._apply_clustering(survey_tensor)
                results.update(cluster_results)

            # Statistics
            if self.config.enable_statistics:
                print("  📈 Statistics...")
                stats_results = self._apply_statistics(survey_tensor)
                results.update(stats_results)

        except Exception as e:
            print(f"  ⚠️ Processing failed: {e}")
            results["error"] = str(e)

        print("  ✅ Processing complete!")
        return results

    def _apply_features(self, survey_tensor: SurveyTensor) -> Dict[str, Any]:
        """Apply feature engineering."""
        try:
            # Create FeatureTensor
            feature_tensor = FeatureTensor(
                survey_tensor._data,
                feature_names=[
                    f"feature_{i}" for i in range(survey_tensor._data.shape[1])
                ],
                feature_types="mixed",
            )

            # Apply basic scaling
            scaled_data = feature_tensor.scale_features(method="standard")

            return {
                "feature_tensor": feature_tensor,
                "scaled_data": scaled_data,
                "n_features": survey_tensor._data.shape[1],
            }

        except Exception as e:
            print(f"    ⚠️ Feature engineering failed: {e}")
            return {"feature_error": str(e)}

    def _apply_clustering(self, survey_tensor: SurveyTensor) -> Dict[str, Any]:
        """Apply clustering analysis."""
        try:
            # Use first 3 columns for clustering (or all if less than 3)
            n_cols = min(3, survey_tensor._data.shape[1])
            cluster_data = survey_tensor._data[:, :n_cols]

            # Create ClusteringTensor
            cluster_tensor = ClusteringTensor(
                cluster_data,
                coordinate_system="cartesian",
                coordinate_columns=list(range(n_cols)),
            )

            # Apply DBSCAN
            cluster_labels = cluster_tensor.dbscan_clustering(
                eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples
            )

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            return {
                "cluster_tensor": cluster_tensor,
                "cluster_labels": cluster_labels,
                "n_clusters": n_clusters,
                "n_noise": sum(1 for label in cluster_labels if label == -1),
            }

        except Exception as e:
            print(f"    ⚠️ Clustering failed: {e}")
            return {"clustering_error": str(e)}

    def _apply_statistics(self, survey_tensor: SurveyTensor) -> Dict[str, Any]:
        """Apply statistical analysis."""
        try:
            # Create StatisticsTensor
            stats_tensor = StatisticsTensor(
                survey_tensor._data,
                coordinate_columns=[0, 1] if survey_tensor._data.shape[1] >= 2 else [0],
            )

            # Compute basic luminosity function
            if survey_tensor._data.shape[1] > 2:
                bin_centers, phi = stats_tensor.luminosity_function(
                    magnitude_column=2,  # Use 3rd column as magnitude
                    bins=10,
                    function_name="basic_lf",
                )

                return {
                    "stats_tensor": stats_tensor,
                    "luminosity_function": {"bin_centers": bin_centers, "phi": phi},
                    "n_functions": len(stats_tensor.list_functions()),
                }
            else:
                return {
                    "stats_tensor": stats_tensor,
                    "message": "Not enough columns for luminosity function",
                }

        except Exception as e:
            print(f"    ⚠️ Statistics failed: {e}")
            return {"statistics_error": str(e)}


# Keep the original class for backward compatibility but disable problematic features
class AdvancedAstroProcessor:
    """Advanced processor with experimental features (mostly disabled)."""

    def __init__(self, config: Optional[SimpleProcessingConfig] = None):
        """Initialize with simplified config."""
        self.config = config or SimpleProcessingConfig()
        self.simple_processor = SimpleAstroProcessor(config)

    def process(self, survey_tensor: SurveyTensor) -> Dict[str, Any]:
        """Process using simplified processor."""
        print("⚠️ Using simplified processing (advanced features disabled)")
        return self.simple_processor.process(survey_tensor)


# For backward compatibility
ProcessingConfig = SimpleProcessingConfig
