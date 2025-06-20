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

            # Get feature names from column mapping
            column_mapping = survey_tensor.column_mapping
            feature_names = (
                list(column_mapping.keys())
                if column_mapping
                else [f"col_{i}" for i in range(data_matrix.shape[1])]
            )

            # Create FeatureTensor
            feature_tensor = FeatureTensor(data_matrix, feature_names=feature_names)

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
                column_mapping = survey_tensor.column_mapping
                feature_names = list(column_mapping.keys()) if column_mapping else []

                for i, name in enumerate(feature_names):
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
                # Try to extract from survey tensor directly first
                column_mapping = survey_tensor.column_mapping
                if "ra" in column_mapping and "dec" in column_mapping:
                    ra_idx = column_mapping["ra"]
                    dec_idx = column_mapping["dec"]
                    ra_dec = survey_tensor._data[:, [ra_idx, dec_idx]].cpu().numpy()
                    coordinates = ra_dec
                else:
                    # Fallback: use spatial tensor
                    spatial_tensor = survey_tensor.get_spatial_tensor()
                    if hasattr(spatial_tensor, "cartesian"):
                        coords_3d = spatial_tensor.cartesian.cpu().numpy()
                        coordinates = (
                            coords_3d[:, :2] if coords_3d.shape[1] >= 2 else coords_3d
                        )
            except Exception as e:
                print(f"    ⚠️ Could not extract coordinates: {e}")
                coordinates = None

            # Create StatisticsTensor
            stats_tensor = StatisticsTensor(data_matrix, coordinates=coordinates)

            results = {}

            # Compute luminosity functions
            if self.config.compute_luminosity_functions:
                # Find magnitude columns
                column_mapping = survey_tensor.column_mapping
                feature_names = list(column_mapping.keys()) if column_mapping else []
                mag_columns = []

                for name in feature_names:
                    if "mag" in name.lower():
                        mag_columns.append(column_mapping[name])

                for mag_col in mag_columns[:3]:  # Limit to first 3 magnitude bands
                    try:
                        bin_centers, phi = stats_tensor.luminosity_function(
                            magnitude_column=mag_col,
                            bins=20,
                            function_name=f"lf_{feature_names[mag_col] if mag_col < len(feature_names) else mag_col}",
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

            results["statistics_tensor"] = stats_tensor
            results["n_functions"] = len(stats_tensor.list_functions())

            print(f"    ✅ Statistics: {results['n_functions']} functions computed")
            return results

        except Exception as e:
            print(f"    ❌ Statistical analysis failed: {e}")
            return {"error": str(e)}

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

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to processing device."""
        return tensor.to(self.device)

    def process_coordinate_data(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Process astronomical coordinate data.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing coordinate tensors

        Returns
        -------
        Dict[str, torch.Tensor]
            Processed coordinate data
        """
        processed = {}

        for key, tensor in data.items():
            # Move to device
            tensor = self.to_device(tensor)

            # Basic processing
            if "ra" in key.lower():
                # Ensure RA is in [0, 360] range
                processed[key] = torch.fmod(tensor, 360.0)
            elif "dec" in key.lower():
                # Ensure Dec is in [-90, 90] range
                processed[key] = torch.clamp(tensor, -90.0, 90.0)
            elif "distance" in key.lower() or "dist" in key.lower():
                # Ensure distances are positive
                processed[key] = torch.abs(tensor)
            else:
                processed[key] = tensor

        return processed

    def normalize_magnitudes(
        self, magnitudes: torch.Tensor, mag_zero: float = 0.0, mag_range: float = 30.0
    ) -> torch.Tensor:
        """
        Normalize magnitude values for ML processing.

        Parameters
        ----------
        magnitudes : torch.Tensor
            Magnitude tensor
        mag_zero : float
            Zero point for normalization
        mag_range : float
            Magnitude range for scaling

        Returns
        -------
        torch.Tensor
            Normalized magnitudes
        """
        magnitudes = self.to_device(magnitudes)

        # Handle missing values (typically 99.0 in astronomical data)
        mask = magnitudes < 90.0
        normalized = torch.zeros_like(magnitudes)

        # Normalize valid magnitudes to [0, 1]
        valid_mags = magnitudes[mask]
        if len(valid_mags) > 0:
            normalized[mask] = (valid_mags - mag_zero) / mag_range

        return torch.clamp(normalized, 0.0, 1.0)

    def create_spatial_data(
        self,
        coordinates: torch.Tensor,
        ra: Optional[torch.Tensor] = None,
        dec: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Create spatial data dictionary from coordinate data.

        Parameters
        ----------
        coordinates : torch.Tensor
            Coordinate tensor [N, 3]
        ra : torch.Tensor, optional
            Right ascension
        dec : torch.Tensor, optional
            Declination
        distance : torch.Tensor, optional
            Distance values

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with spatial data
        """
        try:
            # Move to device
            coordinates = self.to_device(coordinates)

            spatial_data = {
                "coordinates": coordinates,
                "coordinate_system": "icrs",
                "unit": "Mpc",
            }

            if ra is not None:
                spatial_data["ra"] = self.to_device(ra)
            if dec is not None:
                spatial_data["dec"] = self.to_device(dec)
            if distance is not None:
                spatial_data["distance"] = self.to_device(distance)

            print(f"✅ Created spatial data: {coordinates.shape}")
            return spatial_data

        except Exception as e:
            print(f"❌ Error creating spatial data: {e}")
            return {}

    def create_survey_data(
        self,
        features: torch.Tensor,
        survey_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create survey data dictionary from feature data.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor
        survey_name : str
            Name of the survey
        metadata : Dict[str, Any], optional
            Additional metadata

        Returns
        -------
        Dict[str, Any]
            Dictionary with survey data
        """
        try:
            # Move to device
            features = self.to_device(features)

            # Create data dictionary
            survey_data = {
                "features": features,
                "survey_name": survey_name,
                "shape": features.shape,
                "device": str(features.device),
                **(metadata or {}),
            }

            print(f"✅ Created survey data for {survey_name}: {features.shape}")
            return survey_data

        except Exception as e:
            print(f"❌ Error creating survey data for {survey_name}: {e}")
            return {}

    def compute_colors(
        self, magnitudes: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute astronomical colors from magnitudes.

        Parameters
        ----------
        magnitudes : Dict[str, torch.Tensor]
            Dictionary of magnitude tensors

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of color tensors
        """
        colors = {}

        # Common color combinations
        color_pairs = [
            ("u", "g"),
            ("g", "r"),
            ("r", "i"),
            ("i", "z"),
            ("g", "i"),
            ("r", "z"),
            ("u", "r"),
        ]

        for mag1, mag2 in color_pairs:
            if mag1 in magnitudes and mag2 in magnitudes:
                color_name = f"{mag1}_{mag2}"
                m1 = self.to_device(magnitudes[mag1])
                m2 = self.to_device(magnitudes[mag2])

                # Only compute colors for valid magnitudes
                mask = (m1 < 90.0) & (m2 < 90.0)
                color = torch.zeros_like(m1)
                color[mask] = m1[mask] - m2[mask]
                colors[color_name] = color

        return colors

    def save_tensors(
        self, tensors: Dict[str, Any], output_dir: Union[str, Path]
    ) -> Path:
        """
        Save processed tensors to disk.

        Parameters
        ----------
        tensors : Dict[str, Any]
            Tensor dictionary to save
        output_dir : Union[str, Path]
            Output directory

        Returns
        -------
        Path
            Path to saved tensors
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tensor_file = output_dir / "astro_tensors.pt"

        # Convert complex objects to regular tensors for saving
        save_dict = {}
        for name, tensor_obj in tensors.items():
            if isinstance(tensor_obj, dict) and "features" in tensor_obj:
                # Save the features tensor from survey data
                save_dict[name] = tensor_obj["features"]
            elif isinstance(tensor_obj, dict) and "coordinates" in tensor_obj:
                # Save the coordinates tensor from spatial data
                save_dict[name] = tensor_obj["coordinates"]
            elif isinstance(tensor_obj, torch.Tensor):
                save_dict[name] = tensor_obj
            else:
                # Skip non-tensor objects
                continue

        if save_dict:
            torch.save(save_dict, tensor_file)
            print(f"💾 Tensors saved to {tensor_file}")
            print(f"   - {len(save_dict)} tensor objects")
        else:
            print("⚠️  No tensors to save")

        return tensor_file

    def load_tensors(self, tensor_file: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Load processed tensors from disk.

        Parameters
        ----------
        tensor_file : Union[str, Path]
            Path to tensor file

        Returns
        -------
        Dict[str, torch.Tensor]
            Loaded tensors
        """
        tensor_file = Path(tensor_file)

        if not tensor_file.exists():
            raise FileNotFoundError(f"Tensor file not found: {tensor_file}")

        tensors = torch.load(tensor_file, map_location=self.device)

        print(f"📂 Loaded tensors from {tensor_file}")
        print(f"   - {len(tensors)} tensor objects")

        return tensors


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
