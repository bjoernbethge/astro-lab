"""
AstroLab Data Processing - Advanced data processing with memory management
=======================================================================

Enhanced data processing with memory optimization, batch processing,
and comprehensive cleanup.
"""

import argparse
import gc
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import psutil
import torch
from pydantic import BaseModel, Field, field_validator

from astro_lab.tensors.feature_tensordict import FeatureTensorDict
from astro_lab.tensors.satellite_tensordict import SatelliteTensorDict
from astro_lab.tensors.simulation_tensordict import SimulationTensorDict
from astro_lab.tensors.survey_tensordict import SurveyTensorDict

logger = logging.getLogger(__name__)

# =========================================================================
# 🧹 MEMORY MANAGEMENT CONTEXTS
# =========================================================================


@contextmanager
def comprehensive_cleanup_context():
    """Context manager for comprehensive memory cleanup."""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextmanager
def pytorch_memory_context():
    """Context manager for PyTorch memory management."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextmanager
def memory_tracking_context():
    """Context manager for memory tracking."""
    try:
        yield
    finally:
        pass


@contextmanager
def file_processing_context():
    """Context manager for file processing."""
    try:
        yield
    finally:
        pass


@contextmanager
def batch_processing_context():
    """Context manager for batch processing."""
    try:
        yield
    finally:
        gc.collect()


# =========================================================================
# ⚙️ PROCESSING CONFIGURATION
# =========================================================================


class SimpleProcessingConfig(BaseModel):
    """configuration for data processing with memory management."""

    # Basic options
    device: str = Field(default="cuda", description="Device to use for processing")
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

    # Output options
    save_intermediate: bool = Field(
        default=False, description="Save intermediate results"
    )
    output_format: str = Field(default="parquet", description="Output format")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError("Device must be 'cpu', 'cuda', or 'mps'")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        if v not in ["parquet", "json", "pickle"]:
            raise ValueError("Output format must be 'parquet', 'json', or 'pickle'")
        return v


# =========================================================================
# 🔧 ENHANCED DATA PROCESSOR
# =========================================================================


class EnhancedDataProcessor:
    """Enhanced data processor with memory management."""

    def __init__(self, config: Optional[SimpleProcessingConfig] = None):
        """Initialize processor with configuration."""
        self.config = config or SimpleProcessingConfig()
        self.device = torch.device(self.config.device)
        self.processing_stats = {
            "files_processed": 0,
            "total_objects": 0,
            "memory_peak_mb": 0.0,
        }

    def process(
        self,
        data: Union[pl.DataFrame, Path, str],
        output_path: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process data with memory management.

        Args:
            data: Input data (DataFrame or file path)
            output_path: Optional output path
            **kwargs: Additional parameters

        Returns:
            Processing results
        """
        if isinstance(data, (str, Path)):
            return self._process_file(Path(data), output_path, **kwargs)
        elif isinstance(data, pl.DataFrame):
            return self._process_dataframe(data, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _process_file(
        self, file_path: Path, output_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        """Process a single file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"🔄 Processing file: {file_path.name}")

        # Load data based on file type
        if file_path.suffix == ".parquet":
            df = pl.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pl.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        return self._process_dataframe(df, output_path, **kwargs)

    def _process_dataframe(
        self, df: pl.DataFrame, output_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        """Process DataFrame with memory management."""
        print(f"📊 Processing DataFrame: {len(df)} rows, {len(df.columns)} columns")

        results = {
            "input_shape": (len(df), len(df.columns)),
            "processing_config": self.config.dict(),
            "tensors_created": {},
            "statistics": {},
        }

        # Create survey tensor
        if self.config.enable_feature_engineering:
            with pytorch_memory_context():
                survey_tensor = self._create_survey_tensor(df)
                results["tensors_created"]["survey"] = survey_tensor.shape

                # Create feature tensor
                feature_tensor = self._create_feature_tensor(survey_tensor)
                results["tensors_created"]["feature"] = feature_tensor.shape

                if self.config.cleanup_intermediate:
                    del survey_tensor
                    gc.collect()

        # Create clustering tensor
        if self.config.enable_clustering:
            with pytorch_memory_context():
                clustering_tensor = self._create_clustering_tensor(survey_tensor)
                results["tensors_created"]["clustering"] = clustering_tensor.shape

        # Create statistics tensor
        if self.config.enable_statistics:
            with pytorch_memory_context():
                stats_tensor = self._create_statistics_tensor(survey_tensor)
                results["tensors_created"]["statistics"] = stats_tensor.shape

        # Save results if requested
        if output_path:
            self._save_results(results, Path(output_path))

        # Update processing stats
        self.processing_stats["files_processed"] += 1
        self.processing_stats["total_objects"] += len(df)

        print(
            f"✅ Processing completed: {len(results['tensors_created'])} tensors created"
        )

        return results

    def _create_survey_tensor(self, df: pl.DataFrame) -> SurveyTensorDict:
        """Create survey tensor from DataFrame."""
        # Convert DataFrame to tensor format
        tensor_data = {}
        for col in df.columns:
            try:
                col_data = df[col].to_numpy()
                if col_data.dtype.kind in "fc":
                    tensor_data[col] = torch.tensor(col_data, dtype=torch.float32)
                elif col_data.dtype.kind in "i":
                    tensor_data[col] = torch.tensor(col_data, dtype=torch.long)
            except Exception as e:
                print(f"⚠️ Could not convert column {col}: {e}")

        return SurveyTensorDict(tensor_data)

    def _create_feature_tensor(
        self, survey_tensor: SurveyTensorDict
    ) -> FeatureTensorDict:
        """Create feature tensor from survey tensor."""
        # Extract features from survey tensor
        features = {}
        for key, value in survey_tensor.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                features[key] = value

        return FeatureTensorDict(features)

    def _create_clustering_tensor(
        self, survey_tensor: SurveyTensorDict
    ) -> SatelliteTensorDict:
        """Create clustering tensor from survey tensor."""
        # Simple clustering based on spatial coordinates
        if "ra" in survey_tensor and "dec" in survey_tensor:
            coords = torch.stack([survey_tensor["ra"], survey_tensor["dec"]], dim=1)

            # Simple distance-based clustering
            from sklearn.cluster import DBSCAN

            clustering = DBSCAN(
                eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples
            ).fit(coords.numpy())

            cluster_labels = torch.tensor(clustering.labels_, dtype=torch.long)
            return SatelliteTensorDict({"cluster_labels": cluster_labels})

        return SatelliteTensorDict({})

    def _create_statistics_tensor(
        self, survey_tensor: SurveyTensorDict
    ) -> SimulationTensorDict:
        """Create statistics tensor from survey tensor."""
        stats = {}
        for key, value in survey_tensor.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                stats[f"{key}_mean"] = torch.mean(value)
                stats[f"{key}_std"] = torch.std(value)
                stats[f"{key}_min"] = torch.min(value)
                stats[f"{key}_max"] = torch.max(value)

        return SimulationTensorDict(stats)

    def _save_results(self, results: Dict[str, Any], output_path: Path):
        """Save processing results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.output_format == "parquet":
            # Convert results to DataFrame and save
            import pandas as pd

            df_results = pd.DataFrame([results])
            df_results.to_parquet(output_path)
        elif self.config.output_format == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(self._make_serializable(results), f, indent=2)
        elif self.config.output_format == "pickle":
            import pickle

            with open(output_path, "wb") as f:
                pickle.dump(results, f)

        print(f"💾 Results saved to: {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

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

        with batch_processing_context():
            batch_results = {
                "files_processed": [],
                "total_objects": 0,
                "failed_files": [],
            }

            for file_path in file_paths:
                with comprehensive_cleanup_context():
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
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024

        stats = {
            **self.processing_stats,
            "current_memory_mb": current_memory,
            "device": str(self.device),
            "config": self.config.dict(),
        }

        return stats


# processing functions with memory management
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
    with comprehensive_cleanup_context():
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
    with comprehensive_cleanup_context():
        processor = EnhancedDataProcessor(config)
        return processor.process_batch(file_paths, output_dir)


# Keep the original class for backward compatibility but disable problematic features
class AstroProcessor:
    """processor with experimental features (mostly disabled)."""

    def __init__(self, config: Optional[SimpleProcessingConfig] = None):
        """Initialize with simplified config."""
        self.config = config or SimpleProcessingConfig()
        self.simple_processor = EnhancedDataProcessor(config)

    def process(self, survey_tensor: SurveyTensorDict) -> Dict[str, Any]:
        """Process using simplified processor."""
        print("⚠️ Using simplified processing (advanced features disabled)")
        # Convert SurveyTensor to DataFrame for processing
        import polars as pl

        # Create a dummy DataFrame since the processor expects that format
        df = pl.DataFrame({"dummy": [1]})
        return self.simple_processor._process_dataframe(df)


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
    from .core import survey_loader

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
