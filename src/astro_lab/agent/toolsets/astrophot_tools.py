"""AstroPhot — real FITS under ``data/raw/<survey>/``; same fit pipeline as ``ImageProcessor.use_astrophot_models`` without importing ``astro_lab.widgets``."""

from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from astropy.io import fits
from pydantic_ai import FunctionToolset

from astro_lab.config import find_project_root, get_data_paths

from .._surveys import normalize_survey, survey_keys
from ._paths import safe_raw_survey_dir, safe_raw_survey_file

astrophot_toolset = FunctionToolset()

_DOCS_URL = "https://astrophot.readthedocs.io/"


def _raw_dir() -> Path:
    root = find_project_root()
    rel = get_data_paths()["raw_dir"]
    p = Path(rel)
    return p if p.is_absolute() else root / p


def _first_2d_numeric_image_from_fits(path: Path) -> tuple[np.ndarray | None, str | None]:
    """Return the first 2D numeric image HDU as float64, or (None, error_code).

    Catalog / bintable FITS (structured dtypes) are skipped so we never cast
    tables to float64 (that raises the numpy "unsafe" cast error).
    """
    with fits.open(path, memmap=False) as hdul:
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(0.0)
            if data.dtype.names is not None:
                continue
            arr = np.asarray(data)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            if arr.ndim == 2 and arr.size > 0:
                return np.asarray(arr, dtype=np.float64), None
    return None, "no_2d_image_hdu"


def _fit_astrophot_models(
    image: Any, model_type: str, zeropoint: float
) -> dict[str, object]:
    import astrophot as ap

    gaussian_model = cast(Callable[..., Any], ap.models.gaussian_model)
    sersic_model = cast(Callable[..., Any], ap.models.sersic_model)
    point_source = cast(Callable[..., Any], ap.models.point_source)

    target = ap.image.Target_Image(
        data=image.astype(np.float64),
        zeropoint=zeropoint,
        variance="auto",
    )
    if model_type == "gaussian":
        model = gaussian_model(name="gaussian_source", target=target)
    elif model_type == "sersic":
        model = sersic_model(name="galaxy", target=target)
    elif model_type == "point_source":
        model = point_source(name="point_source", target=target)
    else:
        model = gaussian_model(name="gaussian_source", target=target)
    result = ap.fit.LM(model, verbose=0).fit()
    return {"model": model, "result": result, "target": target}


@astrophot_toolset.tool
def astrophot_version_info() -> str:
    """Installed AstroPhot version (distribution metadata)."""
    try:
        ver = importlib.metadata.version("astrophot")
    except importlib.metadata.PackageNotFoundError:
        return json.dumps({"installed": False})
    return json.dumps({"installed": True, "version": ver, "docs": _DOCS_URL}, indent=2)


@astrophot_toolset.tool
def astrophot_list_raw_fits_candidates(survey: str) -> str:
    """List ``.fits`` / ``.fit`` under ``data/raw/<survey>/``."""
    canon = normalize_survey(survey)
    if canon is None:
        return json.dumps(
            {"error": "unknown_survey", "survey": survey, "known": survey_keys()},
            indent=2,
        )
    raw_base = _raw_dir()
    folder = safe_raw_survey_dir(raw_base, canon)
    if folder is None or not folder.is_dir():
        return json.dumps(
            {"survey": canon, "dir": str(raw_base / canon), "files": []},
            indent=2,
        )
    files = sorted(folder.glob("*.fits")) + sorted(folder.glob("*.fit"))
    return json.dumps(
        {
            "survey": canon,
            "dir": str(folder),
            "n_files": len(files),
            "paths": [str(f) for f in files],
        },
        indent=2,
    )


@astrophot_toolset.tool
def astrophot_fit_from_raw_fits(
    survey: str,
    fits_filename: str | None = None,
    model_type: str = "gaussian",
    zeropoint: float = 25.0,
) -> str:
    """Load FITS from ``data/raw/<survey>/``, run AstroPhot LM fit (Gaussian / sersic / point_source)."""
    allowed_models = frozenset({"gaussian", "sersic", "point_source"})
    mt = model_type.strip().lower()
    if mt not in allowed_models:
        mt = "gaussian"
    try:
        zp = float(zeropoint)
        if not math.isfinite(zp):
            zp = 25.0
    except (TypeError, ValueError):
        zp = 25.0
    canon = normalize_survey(survey)
    if canon is None:
        return json.dumps(
            {"error": "unknown_survey", "survey": survey, "known": survey_keys()},
            indent=2,
        )
    raw_base = _raw_dir()
    folder = safe_raw_survey_dir(raw_base, canon)
    if folder is None or not folder.is_dir():
        return json.dumps(
            {"error": "raw_dir_missing", "path": str(raw_base / canon)},
            indent=2,
        )
    if fits_filename:
        path = safe_raw_survey_file(raw_base, canon, fits_filename)
        if path is None:
            return json.dumps(
                {"error": "fits_not_found_or_invalid_name", "path": str(folder / fits_filename)},
                indent=2,
            )
    else:
        candidates = sorted(folder.glob("*.fits")) + sorted(folder.glob("*.fit"))
        if not candidates:
            return json.dumps(
                {"error": "no_fits_in_dir", "path": str(folder)},
                indent=2,
            )
        path = candidates[0]

    try:
        arr, skip_reason = _first_2d_numeric_image_from_fits(path)
    except Exception as e:
        return json.dumps({"error": "fits_read_failed", "path": str(path), "detail": str(e)}, indent=2)

    if arr is None:
        return json.dumps(
            {
                "error": "fits_not_imaging_data",
                "path": str(path),
                "detail": skip_reason,
                "hint": "This file looks like a table/catalog FITS (e.g. NSA). "
                "Use Parquet/DuckDB tools to list stars, not astrophot_fit_from_raw_fits.",
            },
            indent=2,
        )

    try:
        out = _fit_astrophot_models(arr, model_type=mt, zeropoint=zp)
    except Exception as e:
        return json.dumps(
            {
                "error": "astrophot_fit_failed",
                "path": str(path),
                "model_type": mt,
                "detail": str(e),
            },
            indent=2,
        )

    result = out["result"]
    model = out["model"]
    summary = {
        "path": str(path),
        "shape": list(arr.shape),
        "model_type": mt,
        "zeropoint": zp,
        "result_repr": repr(result)[:2000],
        "model_repr": repr(model)[:500],
    }
    return json.dumps(summary, indent=2, default=str)
