"""
AstroLab Data Configuration
==========================

Centralized configuration for data directory structure and paths.
This replaces hardcoded paths throughout the codebase.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union


class DataConfig:
    """Centralized data configuration for AstroLab."""

    def __init__(self, base_dir: Union[str, Path] = "data"):
        self.base_dir = Path(base_dir)

    @property
    def raw_dir(self) -> Path:
        """Raw data directory."""
        return self.base_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Processed data directory."""
        return self.base_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        return self.base_dir / "cache"

    @property
    def experiments_dir(self) -> Path:
        """Experiments directory for MLflow and checkpoints."""
        return self.base_dir / "experiments"

    @property
    def mlruns_dir(self) -> Path:
        """MLflow tracking directory."""
        return self.experiments_dir / "mlruns"

    @property
    def checkpoints_dir(self) -> Path:
        """Lightning checkpoints directory."""
        return self.experiments_dir / "checkpoints"

    @property
    def configs_dir(self) -> Path:
        """Configuration files directory."""
        return self.base_dir / "configs"

    def get_survey_raw_dir(self, survey: str) -> Path:
        """Get raw directory for specific survey."""
        return self.raw_dir / survey

    def get_survey_processed_dir(self, survey: str) -> Path:
        """Get processed directory for specific survey."""
        return self.processed_dir / survey

    def get_catalog_path(self, survey: str, processed: bool = True) -> Path:
        """Get standard catalog path for survey."""
        if processed:
            return self.get_survey_processed_dir(survey) / "catalog.parquet"
        else:
            # Raw catalog naming depends on survey
            raw_dir = self.get_survey_raw_dir(survey)
            return raw_dir / f"{survey}_catalog.parquet"

    def get_graph_path(self, survey: str, k_neighbors: int = 8) -> Path:
        """Get graph data path for survey."""
        return self.get_survey_processed_dir(survey) / f"graphs_k{k_neighbors}.pt"

    def get_tensor_path(self, survey: str) -> Path:
        """Get tensor data path for survey."""
        return self.get_survey_processed_dir(survey) / "tensors.pt"

    def setup_directories(self):
        """Create only core data directory structure (no survey templates)."""
        # Only create core directories
        core_dirs = [
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.experiments_dir,
            self.configs_dir,
        ]

        # Create only core directories
        for dir_path in core_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"📁 Core data structure created in: {self.base_dir}")

    def ensure_survey_directories(self, survey: str):
        """Create directories for a specific survey only when needed."""
        raw_dir = self.get_survey_raw_dir(survey)
        processed_dir = self.get_survey_processed_dir(survey)

        # Create only if they don't exist
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Created directories for {survey} survey")

    def ensure_experiment_directories(self, experiment_name: str):
        """Create experiment directories only when needed."""
        mlruns_dir = self.mlruns_dir
        checkpoint_dir = self.checkpoints_dir / experiment_name

        # Create only if they don't exist
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"🧪 Created experiment directories for {experiment_name}")

    def get_experiment_paths(self, experiment_name: str) -> Dict[str, Path]:
        """Get all paths for an experiment."""
        return {
            "mlruns": self.mlruns_dir,
            "checkpoints": self.checkpoints_dir / experiment_name,
            "config": self.configs_dir / f"{experiment_name}.yaml",
        }

    def migrate_old_structure(self):
        """Migrate from old chaotic structure to new clean structure."""
        print("🔄 Migrating old data structure...")

        # Map old paths to new paths
        migrations = [
            # Raw data migrations
            (self.raw_dir / "fits", self.get_survey_raw_dir("sdss")),
            (self.raw_dir / "hdf5", self.get_survey_raw_dir("tng50") / "hdf5"),
            # Processed data migrations
            (self.processed_dir / "catalogs", self.processed_dir / "temp_catalogs"),
            (self.processed_dir / "ml_ready", self.processed_dir / "temp_ml_ready"),
            (self.processed_dir / "features", self.processed_dir / "temp_features"),
        ]

        for old_path, new_path in migrations:
            if old_path.exists() and old_path.is_dir():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  📦 {old_path} -> {new_path}")
                # Note: Actual file moving would be done manually or with additional logic

        print("✅ Migration plan created. Manual file moving required.")


# Global configuration instance
data_config = DataConfig()

# Environment variable support
if "ASTROLAB_DATA_DIR" in os.environ:
    data_config = DataConfig(os.environ["ASTROLAB_DATA_DIR"])


# Convenience functions for backward compatibility
def get_data_dir() -> Path:
    """Get the configured data directory."""
    return data_config.base_dir


def get_raw_dir() -> Path:
    """Get the raw data directory."""
    return data_config.raw_dir


def get_processed_dir() -> Path:
    """Get the processed data directory."""
    return data_config.processed_dir


def get_survey_paths(survey: str) -> Dict[str, Path]:
    """Get all standard paths for a survey."""
    return {
        "raw_dir": data_config.get_survey_raw_dir(survey),
        "processed_dir": data_config.get_survey_processed_dir(survey),
        "catalog": data_config.get_catalog_path(survey),
        "graphs": data_config.get_graph_path(survey),
        "tensors": data_config.get_tensor_path(survey),
    }


def get_experiment_paths(experiment_name: str) -> Dict[str, Path]:
    """Get all paths for an experiment."""
    return data_config.get_experiment_paths(experiment_name)
