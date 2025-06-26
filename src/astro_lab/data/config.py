"""
AstroLab Data Configuration
==========================

Centralized configuration for data directory structure and paths.
This replaces hardcoded paths throughout the codebase.

Directory Creation Policy:
- Core directories (raw, processed, cache, etc.) are only created when explicitly requested
- Survey-specific directories are only created when actually working with that survey
- No automatic directory creation on import to avoid cluttering the filesystem
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configure AstroPhot logging before any imports
os.environ.setdefault("ASTROPHOT_LOG_LEVEL", "ERROR")
os.environ.setdefault("ASTROPHOT_LOG_FILE", "")

# Configure logging - reduce level to avoid duplicates
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DataConfig:
    """
    Centralized data configuration for AstroLab.

    Manages all data paths, directory structures, and configuration
    for astronomical data processing and analysis.
    """

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
    def results_dir(self) -> Path:
        """Results directory for organized model outputs (in project root, not data/)."""
        return Path("results")

    @property
    def logs_dir(self) -> Path:
        """Logs directory for training logs."""
        return self.experiments_dir / "logs"

    @property
    def configs_dir(self) -> Path:
        """Configuration files directory."""
        return self.base_dir / "configs"

    @property
    def artifacts_dir(self) -> Path:
        """MLflow artifacts directory."""
        return self.experiments_dir / "artifacts"

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
        # Only create core directories under data/
        core_dirs = [
            self.raw_dir,
            self.processed_dir,
            self.experiments_dir,
        ]

        # Create only core directories
        for dir_path in core_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Core data structure created in: {self.base_dir}")

        # Results directory is separate in project root
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 Results directory created: {self.results_dir}")

    def ensure_survey_directories(self, survey: str):
        """Create directories for a specific survey only when needed."""
        raw_dir = self.get_survey_raw_dir(survey)
        processed_dir = self.get_survey_processed_dir(survey)

        # Create only if they don't exist
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Created directories for {survey} survey")

    def ensure_experiment_directories(self, experiment_name: str):
        """Create experiment directories only when needed."""
        mlruns_dir = self.mlruns_dir
        checkpoint_dir = self.checkpoints_dir / experiment_name
        logs_dir = self.logs_dir / experiment_name
        artifacts_dir = self.artifacts_dir

        # Create only if they don't exist
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🧪 Created experiment directories for {experiment_name}")

    def get_results_structure(self, survey: str, model_name: str) -> Dict[str, Path]:
        """Get organized results directory structure for survey/model."""
        base_results = self.results_dir / survey / model_name

        return {
            "base": base_results,
            "models": base_results / "models",
            "plots": base_results / "plots",
            "optuna_plots": base_results / "plots" / "optuna",
        }

    def ensure_results_directories(self, survey: str, model_name: str):
        """Ensure directories for model results exist."""
        results_dir = self.results_dir / survey / model_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different result types
        subdirs = ["checkpoints", "logs", "plots", "predictions"]
        for subdir in subdirs:
            (results_dir / subdir).mkdir(exist_ok=True)

        logger.info(f"✅ Results directories ready: {results_dir}")

        return {
            "base": results_dir,
            "checkpoints": results_dir / "checkpoints",
            "logs": results_dir / "logs",
            "plots": results_dir / "plots",
            "predictions": results_dir / "predictions",
        }

    def get_experiment_paths(self, experiment_name: str) -> Dict[str, Path]:
        """Get all paths for an experiment."""
        return {
            "mlruns": self.mlruns_dir,
            "checkpoints": self.checkpoints_dir / experiment_name,
            "artifacts": self.artifacts_dir,
            "logs": self.logs_dir / experiment_name,
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataConfig":
        """Load DataConfig from YAML file."""
        import yaml

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract data config if nested
        if "data" in config_dict:
            config_dict = config_dict["data"]

        return cls(**config_dict)


# Global configuration instance
data_config = DataConfig()

# Environment variable support
if "ASTROLAB_DATA_DIR" in os.environ:
    data_config = DataConfig(os.environ["ASTROLAB_DATA_DIR"])


# Convenience functions
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
