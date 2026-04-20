"""Data download, preprocess, and dataset status tools."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import FunctionToolset

from astro_lab.config import find_project_root, get_data_paths

from .._surveys import normalize_survey, survey_keys

data_pipeline_toolset = FunctionToolset()


def _require_survey(survey: str) -> tuple[str | None, str | None]:
    canon = normalize_survey(survey)
    if canon is None:
        return None, f"Unknown survey '{survey}'. Known: {', '.join(survey_keys())}"
    return canon, None


@data_pipeline_toolset.tool
def suggest_download_cmd(
    survey: str,
    force: bool = False,
    magnitude_limit: float | None = None,
) -> str:
    """Suggest astro-lab download command for a survey."""
    canon, err = _require_survey(survey)
    if err:
        return err
    parts = ["astro-lab", "download", canon]
    if force:
        parts.append("--force")
    if magnitude_limit is not None:
        parts.extend(["--magnitude-limit", str(magnitude_limit)])
    return " ".join(parts)


@data_pipeline_toolset.tool
def suggest_preprocess_cmd(
    survey: str,
    force: bool = False,
    max_samples: int | None = None,
    sampling_strategy: str = "knn",
) -> str:
    """Suggest astro-lab preprocess command."""
    canon, err = _require_survey(survey)
    if err:
        return err
    parts = ["astro-lab", "preprocess", canon, "--sampling-strategy", sampling_strategy]
    if force:
        parts.append("--force")
    if max_samples is not None:
        parts.extend(["--max-samples", str(max_samples)])
    return " ".join(parts)


@data_pipeline_toolset.tool
def suggest_build_dataset_cmd(survey: str, dataset_type: str = "spatial") -> str:
    """Suggest astro-lab build-dataset command."""
    canon, err = _require_survey(survey)
    if err:
        return err
    return f"astro-lab build-dataset {canon} --type {dataset_type}"


@data_pipeline_toolset.tool
def dataset_status(survey: str) -> str:
    """Summarize expected processed parquet / .pt paths for a survey."""
    canon, err = _require_survey(survey)
    if err:
        return err
    root = find_project_root()
    dp = get_data_paths()
    processed = root / dp["processed_dir"] / canon
    raw = root / dp["raw_dir"] / canon
    lines = [
        f"survey: {canon}",
        f"raw_dir: {raw} (exists={raw.is_dir()})",
        f"processed_dir: {processed} (exists={processed.is_dir()})",
    ]
    if processed.is_dir():
        for pattern in ("*.parquet", "*.pt"):
            files = list(processed.glob(pattern))
            lines.append(f"  {pattern}: {len(files)} file(s)")
            for f in sorted(files)[:5]:
                lines.append(f"    - {f.name}")
    return "\n".join(lines)
