"""
Data Configuration for AstroLab
==============================

Central data configuration for all data-related paths and settings.
"""

from pathlib import Path
from typing import Dict

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class DataConfig:
    """Central data configuration for AstroLab."""

    def __init__(self):
        # All paths relative to project root
        self.project_root = PROJECT_ROOT
        self.base_dir = self.project_root / "data"
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.logs_dir = self.base_dir / "logs"
        self.mlruns_dir = self.base_dir / "mlruns"
        self.configs_dir = self.project_root / "configs"
        self.results_dir = self.project_root / "results"

        # Ensure all directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create all necessary directories."""
        directories = [
            self.base_dir,
            self.raw_dir,
            self.processed_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.mlruns_dir,
            self.configs_dir,
            self.results_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_survey_raw_dir(self, survey: str) -> Path:
        """Get raw data directory for survey."""
        return self.raw_dir / survey

    def get_survey_processed_dir(self, survey: str) -> Path:
        """Get processed data directory for survey."""
        return self.processed_dir / survey

    def get_survey_catalog_path(self, survey: str) -> Path:
        """Get catalog file path for survey."""
        return self.get_survey_raw_dir(survey) / f"{survey}.parquet"

    def get_survey_processed_path(self, survey: str) -> Path:
        """Get processed file path for survey."""
        return self.get_survey_processed_dir(survey) / f"{survey}_processed.parquet"

    def get_experiment_paths(self, experiment_name: str) -> Dict[str, Path]:
        """Get all paths for an experiment."""
        return {
            "checkpoints": self.checkpoints_dir / experiment_name,
            "logs": self.logs_dir / experiment_name,
            "mlruns": self.mlruns_dir / experiment_name,
            "config": self.configs_dir / f"{experiment_name}.yaml",
            "results": self.results_dir / experiment_name,
        }

    def ensure_experiment_directories(self, experiment_name: str):
        """Ensure all experiment directories exist."""
        paths = self.get_experiment_paths(experiment_name)
        for path in paths.values():
            if path.suffix:  # File
                path.parent.mkdir(parents=True, exist_ok=True)
            else:  # Directory
                path.mkdir(parents=True, exist_ok=True)




# Global instance
data_config = DataConfig()


def get_data_config() -> DataConfig:
    """Get the global data configuration instance."""
    return data_config


def setup_data_directories():
    """Setup all data directories."""
    data_config._ensure_directories()
    print("✅ Data directories created successfully")


def list_survey_paths(survey: str) -> Dict[str, Path]:
    """List all paths for a survey."""
    return {
        "raw_dir": data_config.get_survey_raw_dir(survey),
        "processed_dir": data_config.get_survey_processed_dir(survey),
        "catalog_path": data_config.get_survey_catalog_path(survey),
        "processed_path": data_config.get_survey_processed_path(survey),
    }
