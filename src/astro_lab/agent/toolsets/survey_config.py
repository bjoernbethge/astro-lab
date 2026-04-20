"""Survey and configuration tools."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai import FunctionToolset

from astro_lab.config import find_project_root, get_config, get_data_paths, get_survey_config

from .._surveys import is_known_survey, normalize_survey, survey_keys

survey_config_toolset = FunctionToolset()


@survey_config_toolset.tool
def list_surveys() -> str:
    """Return all configured survey keys and short display names."""
    surveys = get_config().get("surveys", {})
    lines = []
    for key in sorted(surveys.keys()):
        name = surveys[key].get("name", "")
        lines.append(f"- {key}: {name}")
    return "\n".join(lines) if lines else "No surveys found in configuration."


@survey_config_toolset.tool
def show_survey_config(survey: str) -> str:
    """Return a JSON summary of one survey's YAML config (coordinates, mags, recommended model)."""
    canon = normalize_survey(survey)
    if canon is None:
        return f"Unknown survey '{survey}'. Known: {', '.join(survey_keys())}"
    cfg = get_survey_config(canon)
    summary = {
        "survey": canon,
        "name": cfg.get("name"),
        "data_release": cfg.get("data_release"),
        "coord_cols": cfg.get("coord_cols", []),
        "mag_cols": cfg.get("mag_cols", []),
        "extra_cols": cfg.get("extra_cols", []),
        "recommended_model": cfg.get("recommended_model", {}),
        "batch_size": cfg.get("batch_size"),
        "k_neighbors": cfg.get("k_neighbors"),
        "experiment_name": cfg.get("experiment_name"),
    }
    return json.dumps(summary, indent=2)


@survey_config_toolset.tool
def validate_config_paths() -> str:
    """Check project configs directory and resolved data paths (existence only)."""
    root = find_project_root()
    configs_dir = root / "configs"
    out = [f"project_root: {root}"]
    out.append(f"configs_dir exists: {configs_dir.is_dir()}")
    paths = get_data_paths()
    for key, rel in paths.items():
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        out.append(f"{key}: {p} (exists={p.exists()})")
    return "\n".join(out)
