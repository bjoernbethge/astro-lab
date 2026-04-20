"""Cross-cutting widget helpers (Parquet viz schema, coordinates)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai import FunctionToolset

from astro_lab.config import find_project_root, get_data_paths

from ..._surveys import normalize_survey, survey_keys
from .._paths import resolve_project_parquet, safe_raw_survey_dir

widget_common_toolset = FunctionToolset()

_PLEIADES_MARIMO = "examples/marimo/pleiades_gaia_cosmograph.py"


def _root() -> Path:
    return find_project_root()


def _spherical_to_cartesian_deg(
    ra_deg: float, dec_deg: float, distance_pc: float
) -> tuple[float, float, float]:
    import numpy as np

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    d = distance_pc
    x = float(d * np.cos(dec_rad) * np.cos(ra_rad))
    y = float(d * np.cos(dec_rad) * np.sin(ra_rad))
    z = float(d * np.sin(dec_rad))
    return x, y, z


def _widgets_parquet_schema_for_viz_impl(
    relative_parquet: str, row_estimate: bool = True
) -> str:
    path = resolve_project_parquet(relative_parquet)
    if path is None:
        return json.dumps(
            {"error": "not_found_or_not_under_project", "path": relative_parquet},
            indent=2,
        )
    import polars as pl

    try:
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()
    except Exception as e:
        return json.dumps({"error": "parquet_schema_failed", "path": str(path), "detail": str(e)}, indent=2)
    names = schema.names()
    dtypes = {n: str(schema[n]) for n in names}
    out: dict[str, Any] = {
        "path": str(path),
        "n_columns": len(names),
        "columns": names,
        "dtypes": dtypes,
        "marimo_notebook_cosmograph": _PLEIADES_MARIMO,
    }
    if row_estimate:
        try:
            out["n_rows"] = lf.select(pl.len()).collect().item()
        except Exception as e:
            out["n_rows_error"] = str(e)
    return json.dumps(out, indent=2)


@widget_common_toolset.tool
def widgets_convert_radec_distance_to_cartesian(
    ra_deg: float,
    dec_deg: float,
    distance_pc: float,
) -> str:
    """Convert RA/Dec (degrees) + distance (pc) to x,y,z (pc Cartesian)."""
    x, y, z = _spherical_to_cartesian_deg(ra_deg, dec_deg, distance_pc)
    return json.dumps(
        {"ra_deg": ra_deg, "dec_deg": dec_deg, "distance_pc": distance_pc, "x": x, "y": y, "z": z},
        indent=2,
    )


@widget_common_toolset.tool
def widgets_parquet_schema_for_viz(relative_parquet: str, row_estimate: bool = True) -> str:
    """Polars schema (+ optional row count) for a project Parquet file."""
    return _widgets_parquet_schema_for_viz_impl(relative_parquet, row_estimate)


@widget_common_toolset.tool
def widgets_first_survey_parquet_schema(survey: str) -> str:
    """Schema for the first ``*.parquet`` under ``data/raw/<survey>/``."""
    canon = normalize_survey(survey)
    if canon is None:
        return json.dumps(
            {"error": "unknown_survey", "survey": survey, "known": survey_keys()},
            indent=2,
        )
    rel_raw = get_data_paths()["raw_dir"]
    raw = Path(rel_raw)
    if not raw.is_absolute():
        raw = _root() / raw
    folder = safe_raw_survey_dir(raw, canon)
    if folder is None or not folder.is_dir():
        return json.dumps(
            {"error": "raw_survey_dir_missing", "dir": str(raw / canon)},
            indent=2,
        )
    parquets = sorted(folder.glob("*.parquet"))
    if not parquets:
        return json.dumps(
            {"error": "no_parquet_in_dir", "dir": str(folder)},
            indent=2,
        )
    rel = parquets[0].relative_to(_root().resolve())
    return _widgets_parquet_schema_for_viz_impl(
        str(rel).replace("\\", "/"), row_estimate=True
    )
