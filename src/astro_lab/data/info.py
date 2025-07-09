"""
Data Information Module
======================

Unified interface to get information about available surveys and their data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from astro_lab.config import get_data_paths

logger = logging.getLogger(__name__)


class SurveyInfo:
    """Get information about available surveys and their data."""

    def __init__(self):
        self.data_paths = get_data_paths()
        self._survey_cache = {}

    def list_available_surveys(self) -> List[str]:
        """List all available survey preprocessors."""
        # Return the actual available surveys from the CLI
        return [
            "gaia",
            "sdss",
            "nsa",
            "tng50",
            "exoplanet",
            "twomass",
            "wise",
            "panstarrs",
            "des",
            "euclid",
            "linear",
            "rrlyrae",
        ]

    def get_survey_status(self, survey: str) -> Dict[str, Any]:
        """Get status of a specific survey's data."""
        if survey in self._survey_cache:
            return self._survey_cache[survey]

        result = {
            "survey": survey,
            "has_preprocessor": survey in self.list_available_surveys(),
            "raw_data": None,
            "processed_data": None,
            "status": "not_found",
        }

        if not result["has_preprocessor"]:
            result["status"] = "no_preprocessor"
            return result

        # Check for raw data
        raw_dir = Path(self.data_paths["raw_dir"]) / survey
        if raw_dir.exists():
            raw_files = list(raw_dir.glob("*.parquet")) + list(raw_dir.glob("*.fits"))
            if raw_files:
                result["raw_data"] = {
                    "path": str(raw_files[0]),
                    "size_mb": raw_files[0].stat().st_size / (1024 * 1024),
                    "exists": True,
                }

        # Check for processed data
        processed_dir = Path(self.data_paths["processed_dir"]) / survey
        processed_file = processed_dir / f"{survey}.parquet"
        if processed_file.exists():
            result["processed_data"] = {
                "path": str(processed_file),
                "size_mb": processed_file.stat().st_size / (1024 * 1024),
                "exists": True,
            }

        # Determine status
        if result["processed_data"]:
            result["status"] = "ready"
        elif result["raw_data"]:
            result["status"] = "needs_processing"
        else:
            result["status"] = "no_data"

        self._survey_cache[survey] = result
        return result

    def get_all_surveys_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all available surveys."""
        surveys = self.list_available_surveys()
        return {survey: self.get_survey_status(survey) for survey in surveys}

    def inspect_survey_data(self, survey: str, sample_size: int = 5) -> Dict[str, Any]:
        """Inspect the structure of survey data."""
        status = self.get_survey_status(survey)

        if status["status"] not in ["ready", "needs_processing"]:
            return {"error": f"Survey {survey} has no data available", "status": status}

        try:
            # Get data path - look for survey files with various patterns
            raw_dir = Path(self.data_paths["raw_dir"]) / survey
            if raw_dir.exists():
                # Look for parquet files in the survey directory
                parquet_files = list(raw_dir.glob("*.parquet"))
                if parquet_files:
                    data_path = parquet_files[0]  # Use the first parquet file found
                else:
                    data_path = raw_dir / f"{survey}.parquet"  # Fallback
            else:
                data_path = raw_dir / f"{survey}.parquet"  # Fallback

            if not data_path.exists():
                return {"error": f"Data file not found: {data_path}"}

            # Load sample
            df = pl.read_parquet(data_path)

            # Get info
            info = {
                "survey": survey,
                "data_path": str(data_path),
                "shape": df.shape,
                "n_objects": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns,
                "coordinate_columns": ["ra", "dec"],
                "magnitude_columns": ["mag"],
                "memory_usage_mb": df.estimated_size() / (1024 * 1024),
                "sample": None,
                "column_info": {},
            }

            # Column details
            for col in df.columns:
                col_data = df[col]
                info["column_info"][col] = {
                    "dtype": str(col_data.dtype),
                    "null_count": col_data.null_count(),
                    "null_percentage": 100 * col_data.null_count() / len(df),
                    "unique_count": col_data.n_unique()
                    if col_data.dtype in [pl.Utf8, pl.Categorical]
                    else None,
                }

                # Statistics for numeric columns
                if col_data.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    non_null = col_data.drop_nulls()
                    if len(non_null) > 0:
                        info["column_info"][col].update(
                            {
                                "min": float(non_null.min()),
                                "max": float(non_null.max()),
                                "mean": float(non_null.mean()),
                                "std": float(non_null.std())
                                if len(non_null) > 1
                                else 0.0,
                                "median": float(non_null.median()),
                            }
                        )

            # Sample data
            if sample_size > 0:
                sample_df = df.head(sample_size)
                info["sample"] = sample_df.to_dicts()

            # Quality validation
            is_valid, issues = self.validate_data(df)
            info["validation"] = {"is_valid": is_valid, "issues": issues}

            # Statistics
            info["statistics"] = self.get_survey_statistics(df)

            return info

        except Exception as e:
            return {
                "error": f"Failed to inspect survey {survey}: {str(e)}",
                "status": status,
            }

    def print_survey_summary(self, survey: str):
        """Print a nice summary of survey data."""
        info = self.inspect_survey_data(survey, sample_size=0)

        if "error" in info:
            print(f"\n❌ Error: {info['error']}")
            return

        print(f"\n{'=' * 60}")
        print(f"Survey: {survey.upper()}")
        print(f"{'=' * 60}")

        print("\n📊 Data Overview:")
        print(f"  Path: {info['data_path']}")
        print(f"  Objects: {info['n_objects']:,}")
        print(f"  Columns: {info['n_columns']}")
        print(f"  Memory: {info['memory_usage_mb']:.1f} MB")

        print("\n🌐 Coordinate System:")
        print(f"  Columns: {', '.join(info['coordinate_columns'])}")

        if info["magnitude_columns"]:
            print("\n🌟 Photometry:")
            print(f"  Bands: {', '.join(info['magnitude_columns'])}")

        print("\n📋 Column Summary:")
        # Group columns by type
        numeric_cols = []
        other_cols = []

        for col, col_info in info["column_info"].items():
            if "mean" in col_info:
                numeric_cols.append((col, col_info))
            else:
                other_cols.append((col, col_info))

        if numeric_cols:
            print(f"\n  Numeric Columns ({len(numeric_cols)}):")
            for col, col_info in sorted(numeric_cols)[:10]:  # Show first 10
                print(
                    f"    {col:30} [{col_info['min']:.2f}, {col_info['max']:.2f}] "
                    f"μ={col_info['mean']:.2f} σ={col_info['std']:.2f}"
                )
            if len(numeric_cols) > 10:
                print(f"    ... and {len(numeric_cols) - 10} more")

        if other_cols:
            print(f"\n  Other Columns ({len(other_cols)}):")
            for col, col_info in sorted(other_cols)[:5]:
                print(f"    {col:30} {col_info['dtype']}")
            if len(other_cols) > 5:
                print(f"    ... and {len(other_cols) - 5} more")

        # Validation
        if info["validation"]["is_valid"]:
            print("\n✅ Data validation: PASSED")
        else:
            print("\n⚠️  Data validation: ISSUES FOUND")
            for issue in info["validation"]["issues"]:
                print(f"    - {issue}")

    def print_all_surveys_status(self):
        """Print status of all available surveys."""
        status = self.get_all_surveys_status()

        print(f"\n{'=' * 80}")
        print(
            f"{'Survey':<15} {'Status':<20} {'Raw Data':<15} {'Processed':<15} {'Action':<20}"
        )
        print(f"{'=' * 80}")

        for survey, info in sorted(status.items()):
            raw = "✓" if info.get("raw_data") else "✗"
            proc = "✓" if info.get("processed_data") else "✗"

            if info["status"] == "ready":
                action = "Ready to use"
            elif info["status"] == "needs_processing":
                action = "Run preprocessing"
            elif info["status"] == "no_data":
                action = "Download data"
            else:
                action = "Check config"

            print(
                f"{survey:<15} {info['status']:<20} {raw:<15} {proc:<15} {action:<20}"
            )

        print("\n💡 To preprocess a survey: astro-lab preprocess <survey_name>")
        print(
            "💡 To inspect a survey: from astro_lab.data.info import SurveyInfo; "
            "SurveyInfo().print_survey_summary('<survey_name>')"
        )

    def compare_raw_and_processed(self, survey: str) -> Dict[str, Any]:
        """Compare raw and processed data fields for a survey."""
        # Find raw file
        raw_dir = Path(self.data_paths["raw_dir"]) / survey
        raw_files = list(raw_dir.glob("*.parquet"))
        if not raw_files:
            return {"error": f"No raw data found for {survey}"}
        raw_path = raw_files[0]
        raw_df = pl.read_parquet(raw_path)
        raw_cols = set(raw_df.columns)

        # Find processed file
        processed_dir = Path(self.data_paths["processed_dir"]) / survey
        processed_file = processed_dir / f"{survey}.parquet"
        if not processed_file.exists():
            processed_file = processed_dir / f"{survey}.parquet"
        if not processed_file.exists():
            return {"error": f"No processed data found for {survey}"}
        processed_df = pl.read_parquet(processed_file)
        processed_cols = set(processed_df.columns)

        # Compare
        only_raw = sorted(list(raw_cols - processed_cols))
        only_processed = sorted(list(processed_cols - raw_cols))
        common = sorted(list(raw_cols & processed_cols))

        return {
            "raw_path": str(raw_path),
            "processed_path": str(processed_file),
            "raw_columns": sorted(list(raw_cols)),
            "processed_columns": sorted(list(processed_cols)),
            "only_in_raw": only_raw,
            "only_in_processed": only_processed,
            "common": common,
            "raw_shape": raw_df.shape,
            "processed_shape": processed_df.shape,
        }

    def print_raw_vs_processed_summary(self, survey: str):
        """Print a side-by-side comparison of raw and processed data fields."""
        cmp = self.compare_raw_and_processed(survey)
        if "error" in cmp:
            print(f"\n❌ Error: {cmp['error']}")
            return
        print(f"\n{'=' * 60}")
        print(f"Survey: {survey.upper()} (Raw vs. Processed)")
        print(f"{'=' * 60}")
        print(f"\nRaw file:        {cmp['raw_path']}")
        print(f"Processed file:  {cmp['processed_path']}")
        print(f"\nRaw shape:       {cmp['raw_shape']}")
        print(f"Processed shape: {cmp['processed_shape']}")
        print(f"\n{'Column':<30} | {'Raw':<5} | {'Processed':<9}")
        print(f"{'-' * 30}-+{'-' * 7}-+{'-' * 11}")
        all_cols = sorted(set(cmp["raw_columns"]) | set(cmp["processed_columns"]))
        for col in all_cols:
            in_raw = "✓" if col in cmp["raw_columns"] else ""
            in_proc = "✓" if col in cmp["processed_columns"] else ""
            print(f"{col:<30} | {in_raw:<5} | {in_proc:<9}")
        if cmp["only_in_raw"]:
            print(f"\n🟡 Only in raw:        {', '.join(cmp['only_in_raw'])}")
        if cmp["only_in_processed"]:
            print(f"\n🟢 Only in processed:  {', '.join(cmp['only_in_processed'])}")

    def validate_data(self, df: pl.DataFrame) -> tuple[bool, list[str]]:
        """Validate data quality and return (is_valid, issues)."""
        issues = []

        # Check for null values in key columns
        if "ra" in df.columns and df["ra"].null_count() > 0:
            issues.append(f"RA column has {df['ra'].null_count()} null values")
        if "dec" in df.columns and df["dec"].null_count() > 0:
            issues.append(f"DEC column has {df['dec'].null_count()} null values")

        # Check coordinate ranges
        if "ra" in df.columns:
            ra_min, ra_max = df["ra"].min(), df["ra"].max()
            if ra_min is not None and ra_max is not None:
                if float(ra_min) < 0 or float(ra_max) > 360:
                    issues.append(
                        f"RA values out of range [0, 360]: [{ra_min}, {ra_max}]"
                    )

        if "dec" in df.columns:
            dec_min, dec_max = df["dec"].min(), df["dec"].max()
            if dec_min is not None and dec_max is not None:
                if float(dec_min) < -90 or float(dec_max) > 90:
                    issues.append(
                        f"DEC values out of range [-90, 90]: [{dec_min}, {dec_max}]"
                    )

        # Check for reasonable data size
        if len(df) < 10:
            issues.append(f"Very small dataset: {len(df)} rows")
        if len(df.columns) < 3:
            issues.append(f"Very few columns: {len(df.columns)}")

        return len(issues) == 0, issues

    def get_survey_statistics(self, df: pl.DataFrame) -> dict[str, any]:
        """Get basic statistics about the survey data."""
        stats = {
            "total_objects": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.estimated_size() / (1024 * 1024),
            "coordinate_columns": [],
            "magnitude_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
        }

        # Categorize columns
        for col in df.columns:
            col_data = df[col]
            if col_data.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                stats["numeric_columns"].append(col)
            elif col_data.dtype in [pl.Utf8, pl.Categorical]:
                stats["categorical_columns"].append(col)

        # Identify coordinate and magnitude columns
        coord_keywords = ["ra", "dec", "x", "y", "z", "l", "b"]
        mag_keywords = ["mag", "flux", "brightness", "phot"]

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in coord_keywords):
                stats["coordinate_columns"].append(col)
            elif any(keyword in col_lower for keyword in mag_keywords):
                stats["magnitude_columns"].append(col)

        return stats


# Convenience functions
def list_surveys() -> List[str]:
    """List all available surveys."""
    return SurveyInfo().list_available_surveys()


def survey_status(survey: Optional[str] = None):
    """Print survey status."""
    info = SurveyInfo()
    if survey:
        info.print_survey_summary(survey)
    else:
        info.print_all_surveys_status()


def inspect_data(survey: str) -> Dict[str, Any]:
    """Inspect survey data structure."""
    return SurveyInfo().inspect_survey_data(survey)
