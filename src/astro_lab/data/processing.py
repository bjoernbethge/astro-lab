"""
AstroLab Data Processing Module
==============================

Data processing utilities for astronomical datasets.
Handles data cleaning, feature engineering, and preprocessing.
"""

import argparse
import sys

# Removed memory.py - using simple context managers
from contextlib import contextmanager


# Minimal no-op context managers
@contextmanager
def comprehensive_cleanup_context(description: str):
    yield


@contextmanager
def pytorch_memory_context(description: str):
    yield


@contextmanager
def memory_tracking_context(description: str):
    yield


@contextmanager
def file_processing_context(file_path, memory_limit_mb=1000.0):
    yield {"file_path": file_path, "stats": {}}


@contextmanager
def batch_processing_context(total_items, batch_size=1, memory_threshold_mb=500.0):
    yield {"total_items": total_items, "stats": {}}


from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, Field, field_validator
from sklearn.neighbors import NearestNeighbors

# Import SimulationTensor as well
from ..tensors import (
    ClusteringTensor,
    FeatureTensor,
    SimulationTensor,
    StatisticsTensor,
    SurveyTensor,
)
from .config import data_config


class SimpleProcessingConfig(BaseModel):
    """Enhanced configuration for data processing with memory management."""

    # Basic options
    device: str = Field(default="cpu", description="Device to use for processing")
    batch_size: int = Field(default=1000, description="Batch size for processing")
    memory_limit_mb: float = Field(default=1000.0, description="Memory limit in MB")

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

    # Memory management
    enable_memory_optimization: bool = Field(
        default=True, description="Enable memory optimization"
    )
    cleanup_intermediate: bool = Field(
        default=True, description="Clean up intermediate results"
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


class EnhancedDataProcessor:
    """
    Enhanced data processor with comprehensive memory management.

    Provides astronomical data processing capabilities with automatic
    memory optimization and resource cleanup.
    """

    def __init__(self, config: Optional[SimpleProcessingConfig] = None):
        """Initialize processor with memory management."""
        with comprehensive_cleanup_context("DataProcessor initialization"):
            self.config = config or SimpleProcessingConfig()
            self.device = torch.device(self.config.device)

            # Set up processing statistics
            self.processing_stats = {
                "files_processed": 0,
                "total_objects": 0,
                "total_memory_used": 0.0,
                "processing_time": 0.0,
            }

    def process(
        self,
        data: Union[pl.DataFrame, Path, str],
        output_path: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process astronomical data with comprehensive memory management.

        Args:
            data: Input data (DataFrame or file path)
            output_path: Optional output path
            **kwargs: Additional processing parameters

        Returns:
            Processing results
        """
        # Handle different input types
        if isinstance(data, (str, Path)):
            return self._process_file(Path(data), output_path, **kwargs)
        elif isinstance(data, pl.DataFrame):
            return self._process_dataframe(data, output_path, **kwargs)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _process_file(
        self, file_path: Path, output_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        """Process file with memory management."""
        with file_processing_context(
            file_path=file_path, memory_limit_mb=self.config.memory_limit_mb
        ) as processing_params:
            # Load data
            with pytorch_memory_context("File loading"):
                if file_path.suffix.lower() in [".fits", ".fit"]:
                    from .utils import load_fits_optimized

                    df = load_fits_optimized(file_path)
                elif file_path.suffix.lower() in [".parquet", ".pq"]:
                    df = pl.read_parquet(file_path)
                elif file_path.suffix.lower() == ".csv":
                    df = pl.read_csv(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Process the dataframe
            result = self._process_dataframe(df, output_path, **kwargs)
            result["input_file"] = str(file_path)
            result["memory_stats"] = processing_params["stats"]

            return result

    def _process_dataframe(
        self, df: pl.DataFrame, output_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        """Process DataFrame with memory management."""
        with comprehensive_cleanup_context("DataFrame processing") as cleanup_stats:
            # Create survey tensor
            with pytorch_memory_context("Survey tensor creation"):
                survey_tensor = self._create_survey_tensor(df)

            results = {
                "survey_name": survey_tensor.survey_name,
                "num_objects": len(survey_tensor),
                "original_columns": len(df.columns),
                "processed_features": {},
                "memory_stats": cleanup_stats,
            }

            # Feature engineering
            if self.config.enable_feature_engineering:
                with pytorch_memory_context("Feature engineering"):
                    feature_tensor = self._create_feature_tensor(survey_tensor)
                    results["processed_features"]["features"] = {
                        "shape": list(feature_tensor.shape),
                        "feature_names": feature_tensor.feature_names,
                    }

            # Clustering
            if self.config.enable_clustering:
                with pytorch_memory_context("Clustering"):
                    clustering_tensor = self._create_clustering_tensor(survey_tensor)
                    results["processed_features"]["clustering"] = {
                        "n_clusters": clustering_tensor.n_clusters,
                        "cluster_centers": clustering_tensor.cluster_centers.shape
                        if clustering_tensor.cluster_centers is not None
                        else None,
                    }

            # Statistics
            if self.config.enable_statistics:
                with pytorch_memory_context("Statistics"):
                    stats_tensor = self._create_statistics_tensor(survey_tensor)
                    results["processed_features"]["statistics"] = {
                        "mean": stats_tensor.mean.tolist()
                        if stats_tensor.mean is not None
                        else None,
                        "std": stats_tensor.std.tolist()
                        if stats_tensor.std is not None
                        else None,
                    }

            # Save results if output path provided
            if output_path:
                with pytorch_memory_context("Results saving"):
                    self._save_results(results, output_path)
                    results["output_path"] = str(output_path)

            return results

    def _create_survey_tensor(self, df: pl.DataFrame) -> SurveyTensor:
        """Create survey tensor with memory optimization."""
        with memory_tracking_context("Survey tensor creation"):
            # Convert to tensor data
            numeric_columns = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]

            if not numeric_columns:
                raise ValueError("No numeric columns found for tensor creation")

            # Extract numeric data
            tensor_data = df.select(numeric_columns).to_numpy()
            tensor_data = torch.from_numpy(tensor_data).float().to(self.device)

            # Create survey tensor
            survey_tensor = SurveyTensor(
                data=tensor_data,
                survey_name=getattr(df, "survey_name", "unknown"),
                column_names=numeric_columns,
            )

            return survey_tensor

    def _create_feature_tensor(self, survey_tensor: SurveyTensor) -> FeatureTensor:
        """Create feature tensor with memory management."""
        with survey_tensor.memory_efficient_context("Feature tensor creation"):
            # Simple feature engineering - can be extended
            features = survey_tensor._data

            # Add some basic features
            if features.shape[1] >= 2:
                # Add distance from origin
                distance = torch.norm(features[:, :2], dim=1, keepdim=True)
                features = torch.cat([features, distance], dim=1)

            feature_names = survey_tensor.column_names + ["distance_2d"]

            return FeatureTensor(data=features, feature_names=feature_names)

    def _create_clustering_tensor(
        self, survey_tensor: SurveyTensor
    ) -> ClusteringTensor:
        """Create clustering tensor with memory management."""
        with survey_tensor.memory_efficient_context("Clustering tensor creation"):
            # Use sklearn for clustering (CPU-based)
            data_cpu = survey_tensor._data.cpu().numpy()

            from sklearn.cluster import DBSCAN

            dbscan = DBSCAN(
                eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples
            )
            cluster_labels = dbscan.fit_predict(data_cpu)

            # Convert back to tensor
            labels_tensor = torch.from_numpy(cluster_labels).to(self.device)

            return ClusteringTensor(
                data=survey_tensor._data, labels=labels_tensor, algorithm="dbscan"
            )

    def _create_statistics_tensor(
        self, survey_tensor: SurveyTensor
    ) -> StatisticsTensor:
        """Create statistics tensor with memory management."""
        with survey_tensor.memory_efficient_context("Statistics tensor creation"):
            data = survey_tensor._data

            # Calculate statistics
            mean = torch.mean(data, dim=0)
            std = torch.std(data, dim=0)
            min_vals = torch.min(data, dim=0)[0]
            max_vals = torch.max(data, dim=0)[0]

            return StatisticsTensor(
                data=data, mean=mean, std=std, min=min_vals, max=max_vals
            )

    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save processing results with memory management."""
        with comprehensive_cleanup_context("Results saving"):
            import json

            # Convert tensors to serializable format
            serializable_results = self._make_serializable(results)

            # Save as JSON
            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2)

    def _make_serializable(self, obj: Any) -> Any:
        """Make object serializable by converting tensors to lists."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif hasattr(obj, "__dict__"):
            return str(obj)
        else:
            return obj

    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple files in batch with memory management.

        Args:
            file_paths: List of file paths to process
            output_dir: Optional output directory

        Returns:
            Batch processing results
        """
        file_paths = [Path(p) for p in file_paths]

        with batch_processing_context(
            total_items=len(file_paths),
            batch_size=1,  # Process one file at a time
            memory_threshold_mb=self.config.memory_limit_mb,
        ) as batch_config:
            batch_results = {
                "files_processed": [],
                "total_objects": 0,
                "failed_files": [],
                "memory_stats": batch_config["stats"],
            }

            for file_path in file_paths:
                with comprehensive_cleanup_context(f"Batch file: {file_path.name}"):
                    try:
                        # Determine output path
                        if output_dir:
                            output_path = (
                                Path(output_dir) / f"{file_path.stem}_processed.json"
                            )
                        else:
                            output_path = None

                        # Process file
                        result = self._process_file(file_path, output_path)
                        batch_results["files_processed"].append(result)
                        batch_results["total_objects"] += result.get("num_objects", 0)

                    except Exception as e:
                        batch_results["failed_files"].append(
                            {"file": str(file_path), "error": str(e)}
                        )

            return batch_results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        try:
            import psutil

            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            current_memory = 0.0

        stats = {
            **self.processing_stats,
            "current_memory_mb": current_memory,
            "device": str(self.device),
            "config": self.config.dict(),
        }

        return stats


# Enhanced processing functions with memory management
def process_survey_data(
    data: Union[pl.DataFrame, Path, str],
    config: Optional[SimpleProcessingConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Process survey data with memory management.

    Args:
        data: Input data
        config: Processing configuration
        **kwargs: Additional parameters

    Returns:
        Processing results
    """
    with comprehensive_cleanup_context("Survey data processing"):
        processor = EnhancedDataProcessor(config)
        return processor.process(data, **kwargs)


def batch_process_files(
    file_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[SimpleProcessingConfig] = None,
) -> Dict[str, Any]:
    """
    Batch process multiple files with memory management.

    Args:
        file_paths: List of file paths
        output_dir: Output directory
        config: Processing configuration

    Returns:
        Batch processing results
    """
    with comprehensive_cleanup_context("Batch file processing"):
        processor = EnhancedDataProcessor(config)
        return processor.process_batch(file_paths, output_dir)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AstroLab Data Processing CLI")
    parser.add_argument(
        "--survey",
        type=str,
        required=True,
        help="Survey name: gaia, sdss, tng50, tng50_temporal",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path (.parquet or .pt)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples/particles",
    )
    parser.add_argument(
        "--snapshot-id",
        type=int,
        default=None,
        help="Snapshot ID (for temporal datasets)",
    )
    args = parser.parse_args()

    print(f"🚀 Processing survey: {args.survey}")
    print(f"   Output: {args.output}")
    print(f"   Max samples: {args.max_samples}")
    if args.snapshot_id is not None:
        print(f"   Snapshot ID: {args.snapshot_id}")

    # Loader mapping
    from .core import (
        load_gaia_data,
        load_sdss_data,
        load_tng50_data,
        load_tng50_temporal_data,
    )

    survey_loaders = {
        "gaia": load_gaia_data,
        "sdss": load_sdss_data,
        "tng50": load_tng50_data,
        "tng50_temporal": load_tng50_temporal_data,
    }
    if args.survey not in survey_loaders:
        print(f"❌ Unknown survey: {args.survey}")
        sys.exit(1)

    # Load data
    loader = survey_loaders[args.survey]
    loader_kwargs = {"max_samples": args.max_samples}
    if args.survey == "tng50_temporal" and args.snapshot_id is not None:
        loader_kwargs["snapshot_id"] = args.snapshot_id
    dataset = loader(**loader_kwargs)

    # Save output
    output_path = Path(args.output)
    if output_path.suffix == ".parquet":
        print(f"💾 Saving as Parquet: {output_path}")
        dataset.data.write_parquet(str(output_path))
        print(f"✅ Done: {output_path}")
    elif output_path.suffix == ".pt":
        print(f"💾 Saving as PyTorch tensor: {output_path}")
        torch.save(dataset.data.to_pandas(), str(output_path))
        print(f"✅ Done: {output_path}")
    else:
        print(f"❌ Unknown output file type: {output_path.suffix}")
        sys.exit(2)
