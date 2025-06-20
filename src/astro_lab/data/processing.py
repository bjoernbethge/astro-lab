"""
Advanced tensor processing for astronomical data.

Enhanced with ML feature engineering, clustering, statistics, and cross-matching.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import the new data processing tensors
from astro_lab.tensors import (
    ClusteringTensor,
    CrossMatchTensor,
    FeatureTensor,
    StatisticsTensor,
    SurveyTensor,
)


class ProcessingConfig(BaseModel):
    """Type-safe configuration for tensor processing."""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", use_enum_values=True
    )

    device: str = Field(default="auto", description="Device for tensor operations")
    batch_size: int = Field(
        default=32, ge=1, le=10000, description="Batch size for processing"
    )
    max_samples: Optional[Dict[str, int]] = Field(
        default=None, description="Maximum samples per survey"
    )
    surveys: Optional[List[str]] = Field(
        default=None, description="List of surveys to process"
    )

    # New processing options
    enable_feature_engineering: bool = Field(
        default=True, description="Enable ML feature engineering"
    )
    enable_clustering: bool = Field(
        default=False, description="Enable clustering analysis"
    )
    enable_statistics: bool = Field(
        default=False, description="Enable statistical analysis"
    )
    enable_crossmatch: bool = Field(default=False, description="Enable cross-matching")

    # Feature engineering options
    feature_scaling_method: str = Field(
        default="astronomical", description="Feature scaling method"
    )
    feature_imputation_method: str = Field(
        default="astronomical", description="Missing value imputation method"
    )
    outlier_detection_method: str = Field(
        default="astronomical", description="Outlier detection method"
    )

    # Clustering options
    clustering_method: str = Field(default="dbscan", description="Clustering algorithm")
    clustering_eps: float = Field(default=1.0, description="DBSCAN eps parameter")
    clustering_min_samples: int = Field(
        default=5, description="DBSCAN min_samples parameter"
    )

    # Statistics options
    compute_luminosity_functions: bool = Field(
        default=True, description="Compute luminosity functions"
    )
    compute_correlation_functions: bool = Field(
        default=False, description="Compute correlation functions"
    )
    bootstrap_errors: bool = Field(
        default=False, description="Compute bootstrap errors"
    )

    @field_validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate device string."""
        if v not in ["auto", "cpu", "cuda", "mps"]:
            raise ValueError(
                f"device must be one of ['auto', 'cpu', 'cuda', 'mps'], got {v}"
            )
        return v

    def __init__(self, **data: Any) -> None:
        """Initialize with default values."""
        # Set defaults for max_samples and surveys if not provided
        if "max_samples" not in data or data["max_samples"] is None:
            data["max_samples"] = {
                "gaia": 10000,
                "sdss_spectral": 1000,
                "linear_lightcurve": 500,
                "rrlyrae": 300,
                "nsa": 5000,
                "exoplanet": 2000,
                "satellite": 100,
                "tng50": 5000,
                "astrophot": 1000,
            }

        if "surveys" not in data or data["surveys"] is None:
            data["surveys"] = [
                "gaia",
                "sdss_spectral",
                "linear_lightcurve",
                "rrlyrae",
                "nsa",
                "exoplanet",
                "satellite",
                "tng50",
                "astrophot",
            ]

        super().__init__(**data)


class AdvancedAstroProcessor:
    """
    Advanced astronomical data processor with ML capabilities.

    Integrates feature engineering, clustering, statistics, and cross-matching
    for comprehensive astronomical data analysis.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the advanced processor.

        Parameters
        ----------
        config : ProcessingConfig, optional
            Processing configuration
        """
        self.config = config or ProcessingConfig()

        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        print(f"🚀 AdvancedAstroProcessor initialized on {self.device}")
        print(f"   Features: {self.config.enable_feature_engineering}")
        print(f"   Clustering: {self.config.enable_clustering}")
        print(f"   Statistics: {self.config.enable_statistics}")
        print(f"   CrossMatch: {self.config.enable_crossmatch}")

    def process_survey_data(
        self, survey_tensor: SurveyTensor, output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Process survey data with advanced tensor operations.

        Parameters
        ----------
        survey_tensor : SurveyTensor
            Input survey tensor
        output_dir : Path, optional
            Output directory for results

        Returns
        -------
        Dict[str, Any]
            Processing results
        """
        results = {
            "survey_name": survey_tensor.survey_name,
            "n_objects": len(survey_tensor),
            "processing_steps": [],
        }

        print(
            f"\n🌟 Processing {survey_tensor.survey_name} data ({len(survey_tensor)} objects)"
        )

        # Step 1: Feature Engineering
        if self.config.enable_feature_engineering:
            feature_results = self._process_features(survey_tensor)
            results["features"] = feature_results
            results["processing_steps"].append("feature_engineering")

        # Step 2: Clustering Analysis
        if self.config.enable_clustering:
            clustering_results = self._process_clustering(survey_tensor)
            results["clustering"] = clustering_results
            results["processing_steps"].append("clustering")

        # Step 3: Statistical Analysis
        if self.config.enable_statistics:
            stats_results = self._process_statistics(survey_tensor)
            results["statistics"] = stats_results
            results["processing_steps"].append("statistics")

        # Save results if output directory specified
        if output_dir:
            self._save_processing_results(results, output_dir)

        return results

    def _process_features(self, survey_tensor: SurveyTensor) -> Dict[str, Any]:
        """Process features using FeatureTensor."""
        print("  🔧 Feature Engineering...")

        try:
            # Convert survey data to feature matrix
            data_matrix = survey_tensor._data.cpu().numpy()
            feature_names = survey_tensor.column_names

            # Create FeatureTensor
            feature_tensor = FeatureTensor(
                data_matrix,
                feature_names=feature_names,
                survey_flags={"survey": survey_tensor.survey_name},
            )

            # Apply preprocessing pipeline
            processed_tensor = feature_tensor

            if self.config.feature_imputation_method:
                processed_tensor = processed_tensor.impute_missing_values(
                    method=self.config.feature_imputation_method
                )

            if self.config.feature_scaling_method:
                processed_tensor = processed_tensor.scale_features(
                    method=self.config.feature_scaling_method
                )

            # Detect outliers
            outliers = processed_tensor.detect_outliers(
                method=self.config.outlier_detection_method
            )

            # Compute colors if magnitude data available
            mag_bands = [name for name in feature_names if "mag" in name.lower()]
            if len(mag_bands) >= 2:
                try:
                    processed_tensor = processed_tensor.compute_colors(mag_bands[:3])
                except Exception as e:
                    print(f"    ⚠️ Color computation failed: {e}")

            # Get feature statistics
            stats = processed_tensor.get_feature_statistics()

            results = {
                "n_features": processed_tensor.n_features,
                "n_outliers": int(outliers.sum()),
                "outlier_fraction": float(outliers.sum()) / len(outliers),
                "feature_statistics": stats,
                "preprocessing_history": processed_tensor.get_metadata(
                    "preprocessing_history", []
                ),
                "processed_tensor": processed_tensor,
            }

            print(
                f"    ✅ Features: {processed_tensor.n_features}, Outliers: {results['n_outliers']}"
            )
            return results

        except Exception as e:
            print(f"    ❌ Feature processing failed: {e}")
            return {"error": str(e)}

    def _process_clustering(self, survey_tensor: SurveyTensor) -> Dict[str, Any]:
        """Process clustering using ClusteringTensor."""
        print("  🔍 Clustering Analysis...")

        try:
            # Get spatial coordinates
            spatial_tensor = survey_tensor.get_spatial_tensor()
            positions = spatial_tensor.cartesian.cpu().numpy()

            # Get additional features if available
            features = None
            if hasattr(survey_tensor, "_data"):
                # Use magnitude and color features
                feature_cols = []
                for i, name in enumerate(survey_tensor.column_names):
                    if any(
                        keyword in name.lower() for keyword in ["mag", "color", "mass"]
                    ):
                        feature_cols.append(i)

                if feature_cols:
                    features = survey_tensor._data[:, feature_cols].cpu().numpy()

            # Create ClusteringTensor
            clustering_tensor = ClusteringTensor(
                positions,
                features=features,
                coordinate_system="cartesian",
                astronomical_context=self._get_astronomical_context(
                    survey_tensor.survey_name
                ),
            )

            # Apply clustering algorithm
            if self.config.clustering_method == "dbscan":
                labels = clustering_tensor.dbscan_clustering(
                    eps=self.config.clustering_eps,
                    min_samples=self.config.clustering_min_samples,
                )
            elif self.config.clustering_method == "galaxy_clusters":
                labels = clustering_tensor.galaxy_cluster_detection(
                    richness_threshold=self.config.clustering_min_samples,
                    radius_mpc=self.config.clustering_eps,
                )
            elif self.config.clustering_method == "stellar_associations":
                labels = clustering_tensor.stellar_association_detection(
                    max_separation_pc=self.config.clustering_eps
                    * 1000,  # Convert to pc
                    min_members=self.config.clustering_min_samples,
                )
            else:
                raise ValueError(
                    f"Unknown clustering method: {self.config.clustering_method}"
                )

            # Get clustering statistics
            stats = clustering_tensor.get_cluster_statistics(
                self.config.clustering_method
            )

            results = {
                "method": self.config.clustering_method,
                "n_clusters": stats["n_clusters"],
                "n_noise": stats["n_noise"],
                "noise_fraction": stats["noise_fraction"],
                "cluster_labels": labels,
                "cluster_statistics": stats,
                "clustering_tensor": clustering_tensor,
            }

            print(
                f"    ✅ Clusters: {stats['n_clusters']}, Noise: {stats['noise_fraction']:.2%}"
            )
            return results

        except Exception as e:
            print(f"    ❌ Clustering failed: {e}")
            return {"error": str(e)}

    def _process_statistics(self, survey_tensor: SurveyTensor) -> Dict[str, Any]:
        """Process statistics using StatisticsTensor."""
        print("  📊 Statistical Analysis...")

        try:
            # Extract observables (magnitudes, colors, etc.)
            data_matrix = survey_tensor._data.cpu().numpy()

            # Get coordinates if available
            coordinates = None
            try:
                spatial_tensor = survey_tensor.get_spatial_tensor()
                ra_dec = spatial_tensor._data[:, :2].cpu().numpy()  # RA, Dec
                coordinates = ra_dec
            except:
                pass

            # Create StatisticsTensor
            stats_tensor = StatisticsTensor(data_matrix, coordinates=coordinates)

            results = {}

            # Compute luminosity functions
            if self.config.compute_luminosity_functions:
                # Find magnitude columns
                mag_columns = []
                for i, name in enumerate(survey_tensor.column_names):
                    if "mag" in name.lower():
                        mag_columns.append(i)

                for mag_col in mag_columns[:3]:  # Limit to first 3 magnitude bands
                    try:
                        bin_centers, phi = stats_tensor.luminosity_function(
                            magnitude_column=mag_col,
                            bins=20,
                            function_name=f"lf_{survey_tensor.column_names[mag_col]}",
                        )
                        results[f"luminosity_function_{mag_col}"] = {
                            "bin_centers": bin_centers,
                            "phi": phi,
                        }
                    except Exception as e:
                        print(f"    ⚠️ LF computation failed for column {mag_col}: {e}")

            # Compute correlation functions
            if self.config.compute_correlation_functions and coordinates is not None:
                try:
                    r_centers, xi_r = stats_tensor.two_point_correlation(
                        r_bins=10,
                        estimator="davis_peebles",  # Use simple estimator
                    )
                    results["correlation_function"] = {
                        "r_centers": r_centers,
                        "xi_r": xi_r,
                    }
                except Exception as e:
                    print(f"    ⚠️ Correlation function failed: {e}")

            # Compute bootstrap errors
            if self.config.bootstrap_errors:
                function_names = stats_tensor.list_functions()
                for func_name in function_names[:2]:  # Limit to first 2 functions
                    try:
                        errors = stats_tensor.bootstrap_errors(
                            func_name,
                            n_bootstrap=50,  # Reduced for speed
                        )
                        results[f"bootstrap_errors_{func_name}"] = {
                            "std_error": errors["std_error"],
                            "confidence_intervals": {
                                "lower": errors["lower_ci"],
                                "upper": errors["upper_ci"],
                            },
                        }
                    except Exception as e:
                        print(f"    ⚠️ Bootstrap errors failed for {func_name}: {e}")

            results["statistics_tensor"] = stats_tensor
            results["n_functions"] = len(stats_tensor.list_functions())

            print(f"    ✅ Statistics: {results['n_functions']} functions computed")
            return results

        except Exception as e:
            print(f"    ❌ Statistical analysis failed: {e}")
            return {"error": str(e)}

    def cross_match_catalogs(
        self,
        catalog_a: SurveyTensor,
        catalog_b: SurveyTensor,
        tolerance_arcsec: float = 2.0,
        method: str = "nearest_neighbor",
    ) -> Dict[str, Any]:
        """Cross-match two catalogs using CrossMatchTensor."""
        print(
            f"  🎯 Cross-matching {catalog_a.survey_name} vs {catalog_b.survey_name}..."
        )

        try:
            # Extract coordinate data
            coords_a = self._extract_coordinates(catalog_a)
            coords_b = self._extract_coordinates(catalog_b)

            # Create CrossMatchTensor
            crossmatch_tensor = CrossMatchTensor(
                coords_a,
                coords_b,
                catalog_names=(catalog_a.survey_name, catalog_b.survey_name),
                coordinate_columns={"a": [0, 1], "b": [0, 1]},
            )

            # Perform cross-matching
            results = crossmatch_tensor.sky_coordinate_matching(
                tolerance_arcsec=tolerance_arcsec, method=method
            )

            # Add Bayesian matching for comparison
            try:
                bayesian_results = crossmatch_tensor.bayesian_matching(
                    tolerance_arcsec=tolerance_arcsec * 2, prior_density=1e-6
                )
                results["bayesian_matches"] = bayesian_results
            except Exception as e:
                print(f"    ⚠️ Bayesian matching failed: {e}")

            print(
                f"    ✅ Matches: {len(results['matches'])}, Rate: {results['statistics']['match_rate_a']:.2%}"
            )
            return results

        except Exception as e:
            print(f"    ❌ Cross-matching failed: {e}")
            return {"error": str(e)}

    def _extract_coordinates(self, survey_tensor: SurveyTensor) -> np.ndarray:
        """Extract RA/Dec coordinates from survey tensor."""
        # Look for RA/Dec columns
        ra_col = dec_col = None
        for i, name in enumerate(survey_tensor.column_names):
            if name.lower() in ["ra", "right_ascension"]:
                ra_col = i
            elif name.lower() in ["dec", "declination"]:
                dec_col = i

        if ra_col is not None and dec_col is not None:
            coords = survey_tensor._data[:, [ra_col, dec_col]].cpu().numpy()
        else:
            # Fallback: use first two columns
            coords = survey_tensor._data[:, :2].cpu().numpy()

        return coords

    def _get_astronomical_context(self, survey_name: str) -> str:
        """Get astronomical context based on survey name."""
        if any(
            keyword in survey_name.lower() for keyword in ["gaia", "star", "stellar"]
        ):
            return "stars"
        elif any(
            keyword in survey_name.lower() for keyword in ["nsa", "sdss", "galaxy"]
        ):
            return "galaxies"
        elif any(
            keyword in survey_name.lower() for keyword in ["tng", "simulation", "lss"]
        ):
            return "lss"
        else:
            return "general"

    def _save_processing_results(self, results: Dict[str, Any], output_dir: Path):
        """Save processing results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_file = output_dir / "processing_summary.json"
        summary = {
            "survey_name": results["survey_name"],
            "n_objects": results["n_objects"],
            "processing_steps": results["processing_steps"],
        }

        # Add summaries from each step
        for step in results["processing_steps"]:
            if step in results and isinstance(results[step], dict):
                step_summary = {
                    k: v
                    for k, v in results[step].items()
                    if not k.endswith("_tensor") and k != "error"
                }
                summary[step] = step_summary

        import json

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"    💾 Results saved to {output_dir}")


# Legacy processor for backward compatibility
class AstroTensorProcessor(AdvancedAstroProcessor):
    """Legacy tensor processor - now redirects to AdvancedAstroProcessor."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        # Create config with feature engineering enabled by default
        if config is None:
            config = ProcessingConfig(enable_feature_engineering=True)
        super().__init__(config)

    # ... existing code ...


# Convenience functions
def create_simple_processor(device: str = "auto") -> AstroTensorProcessor:
    """Create a simple tensor processor."""
    config = ProcessingConfig(device=device)
    return AstroTensorProcessor(config=config)


def process_coordinate_dict(
    coordinates: Dict[str, torch.Tensor], device: str = "auto"
) -> Dict[str, torch.Tensor]:
    """Process a dictionary of coordinate tensors."""
    processor = create_simple_processor(device=device)
    return processor.process_coordinate_data(coordinates)


def normalize_astronomical_data(
    data: Dict[str, torch.Tensor], device: str = "auto"
) -> Dict[str, torch.Tensor]:
    """Normalize astronomical data for ML processing."""
    processor = create_simple_processor(device=device)

    normalized = {}
    for key, tensor in data.items():
        if "mag" in key.lower():
            # Normalize magnitudes
            normalized[key] = processor.normalize_magnitudes(tensor)
        elif any(coord in key.lower() for coord in ["ra", "dec", "distance"]):
            # Process coordinates
            coord_dict = {key: tensor}
            processed = processor.process_coordinate_data(coord_dict)
            normalized.update(processed)
        else:
            # Move to device without special processing
            normalized[key] = processor.to_device(tensor)

    return normalized


__all__ = [
    "ProcessingConfig",
    "AstroTensorProcessor",
    "create_simple_processor",
    "process_coordinate_dict",
    "normalize_astronomical_data",
]
