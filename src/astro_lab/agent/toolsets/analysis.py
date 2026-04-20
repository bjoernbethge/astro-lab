"""Survey analysis via :class:`astro_lab.data.info.SurveyInfo` and cosmic-web CLI hints."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai import FunctionToolset

from astro_lab.config import find_project_root, get_data_paths

from .._surveys import normalize_survey, survey_keys
from ._paths import resolve_project_parquet, safe_raw_survey_dir

analysis_toolset = FunctionToolset()

_COSMIC_SURVEYS = frozenset({"gaia", "nsa", "exoplanet", "sdss", "tng50"})


def _survey_info():
    from astro_lab.data.info import SurveyInfo

    return SurveyInfo()


def _require_survey(survey: str) -> tuple[str | None, str | None]:
    canon = normalize_survey(survey)
    if canon is None:
        return None, f"Unknown survey '{survey}'. Known: {', '.join(survey_keys())}"
    return canon, None


@analysis_toolset.tool
def survey_disk_status_json(survey: str) -> str:
    """On-disk raw/processed presence for one survey (:meth:`SurveyInfo.get_survey_status`)."""
    canon, err = _require_survey(survey)
    if err:
        return json.dumps({"error": err})
    info = _survey_info()
    status = info.get_survey_status(canon)
    return json.dumps(status, indent=2, default=str)


@analysis_toolset.tool
def survey_all_disk_status_json() -> str:
    """Status map for all preprocessors (:meth:`SurveyInfo.get_all_surveys_status`)."""
    info = _survey_info()
    return json.dumps(info.get_all_surveys_status(), indent=2, default=str)


@analysis_toolset.tool
def survey_parquet_inspection_json(survey: str, sample_rows: int = 3) -> str:
    """Column stats + small row sample from first raw Parquet (:meth:`SurveyInfo.inspect_survey_data`)."""
    canon, err = _require_survey(survey)
    if err:
        return json.dumps({"error": err})
    info = _survey_info()
    try:
        cap = max(0, min(int(sample_rows), 20))
    except (TypeError, ValueError):
        cap = 3
    data = info.inspect_survey_data(canon, sample_size=cap)
    if "sample" in data and data["sample"] is not None and cap == 0:
        data = {k: v for k, v in data.items() if k != "sample"}
    return json.dumps(data, indent=2, default=str)


def _cosmic_web_cmd_line(
    canon: str,
    *,
    visualize: bool = False,
    max_samples: int | None = None,
) -> str | None:
    if canon not in _COSMIC_SURVEYS:
        return None
    parts = ["astro-lab", "cosmic-web", canon]
    if visualize:
        parts.append("--visualize")
    if max_samples is not None:
        try:
            ms = int(max_samples)
        except (TypeError, ValueError):
            ms = None
        if ms is not None:
            parts.extend(["--max-samples", str(ms)])
    return " ".join(parts)


@analysis_toolset.tool
def suggest_cosmic_web_cmd(
    survey: str,
    visualize: bool = False,
    max_samples: int | None = None,
) -> str:
    """Suggest ``astro-lab cosmic-web <survey>`` (positional survey). Optional ``--visualize``, ``--max-samples``. No other flags are implied."""
    canon, err = _require_survey(survey)
    if err:
        return err
    line = _cosmic_web_cmd_line(canon, visualize=visualize, max_samples=max_samples)
    if line is None:
        return (
            f"cosmic-web CLI only supports {sorted(_COSMIC_SURVEYS)}; "
            f"'{canon}' is not listed."
        )
    return line


@analysis_toolset.tool
def star_catalog_quicklook_json(
    survey: str,
    preview_limit: int = 12,
) -> str:
    """Factual snapshot for stellar catalogs: disk status, first raw Parquet path, DuckDB row preview. Use for 'show stars' / Sterne."""
    canon, err = _require_survey(survey)
    if err:
        return json.dumps({"error": err})
    try:
        lim = max(1, min(int(preview_limit), 100))
    except (TypeError, ValueError):
        lim = 12

    info = _survey_info()
    disk = info.get_survey_status(canon)

    root = find_project_root().resolve()
    rel_raw = get_data_paths()["raw_dir"]
    raw = Path(rel_raw)
    if not raw.is_absolute():
        raw = root / raw
    folder = safe_raw_survey_dir(raw, canon)

    preview_rows: list[dict[str, object]] | None = None
    columns: list[str] | None = None
    parquet_rel: str | None = None
    preview_error: str | None = None

    if folder is not None and folder.is_dir():
        parquets = sorted(folder.glob("*.parquet"))
        if parquets:
            rel = parquets[0].relative_to(root)
            parquet_rel = str(rel).replace("\\", "/")
            path = resolve_project_parquet(parquet_rel)
            if path is not None:
                import duckdb

                con = duckdb.connect(database=":memory:")
                try:
                    df = con.execute(
                        f"SELECT * FROM read_parquet(?) LIMIT {lim}",
                        [str(path)],
                    ).pl()
                    columns = list(df.columns)
                    preview_rows = df.to_dicts()
                except Exception as e:
                    preview_error = str(e)

    cosmic_line = _cosmic_web_cmd_line(
        canon, visualize=True, max_samples=min(50_000, max(1000, lim * 500))
    )
    out: dict[str, object] = {
        "survey": canon,
        "disk_status": disk,
        "first_raw_parquet_relative": parquet_rel,
        "preview_row_limit": lim,
        "preview_columns": columns,
        "preview_rows": preview_rows,
        "preview_error": preview_error,
        "cosmic_web_command_example": cosmic_line,
        "cosmic_web_readme": (
            "CLI syntax is only: astro-lab cosmic-web <survey> [--visualize] [--max-samples N] "
            "plus optional clustering/catalog flags from --help. "
            "There is no --filters or --survey flag."
        ),
        "cosmic_web_pipeline_note": (
            "The cosmic-web Python entrypoint may still be incomplete in this repo "
            "(check astro_lab.cli.cosmic_web). Prefer Parquet/DuckDB preview above for reliable exploration."
        ),
    }
    nxt: list[str] = ["survey_parquet_inspection_json"]
    if parquet_rel:
        nxt.insert(0, f"duckdb_parquet_preview_rows(relative_parquet={parquet_rel!r})")
    out["suggested_next_tools"] = nxt
    return json.dumps(out, indent=2, default=str)


@analysis_toolset.tool
def suggest_info_cmd(survey: str = "all") -> str:
    """Suggest ``astro-lab info`` for one survey or all."""
    if survey == "all":
        return "astro-lab info all"
    canon, err = _require_survey(survey)
    if err:
        return err
    return f"astro-lab info {canon}"
