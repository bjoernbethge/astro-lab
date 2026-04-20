"""DuckDB + project Parquet (Marimo brings DuckDB); optional astro-extension SQL hints."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai import FunctionToolset

from astro_lab.config import find_project_root

from ._paths import resolve_parquet_sql_path, resolve_project_parquet

astro_duck_toolset = FunctionToolset()

_ASTRO_DUCK_REPO = "https://github.com/synapticore-io/astro-duck"
_MARIMO_EXAMPLE = "examples/marimo/gaia_astro_duck.py"

_DEFAULT_GAIA_PARQUET = "data/raw/gaia/gaia_dr3_bright_all_sky_mag12.0.parquet"
_DEFAULT_TWOMASS_PARQUET = "data/raw/twomass/twomass_psc_mag4.0.parquet"


def _root() -> Path:
    return find_project_root()


@astro_duck_toolset.tool
def astro_duck_extension_and_marimo_pointers() -> str:
    """One screen: INSTALL/LOAD ``astro`` community extension + Marimo example path."""
    return json.dumps(
        {
            "install_load_sql": [
                "INSTALL astro FROM community;",
                "LOAD astro;",
            ],
            "upstream": _ASTRO_DUCK_REPO,
            "marimo_example": _MARIMO_EXAMPLE,
            "functions_cheatsheet": [
                "astro_absolute_mag(apparent_mag, distance_pc)",
                "astro_angular_separation(ra1, dec1, ra2, dec2)  -- degrees",
                "astro_radec_to_xyz(ra, dec, distance_pc)",
                "astro_jd_from_timestamp(ts)",
                "astro_lmst(jd, longitude_deg)",
                "astro_altaz_from_radec(ra, dec, lmst_h, latitude_deg)",
            ],
        },
        indent=2,
    )


@astro_duck_toolset.tool
def gaia_astro_duck_example_paths() -> str:
    """Resolved Gaia / 2MASS Parquet paths and ready-made ``read_parquet`` view SQL."""
    root = _root()
    gaia = root / _DEFAULT_GAIA_PARQUET
    twomass = root / _DEFAULT_TWOMASS_PARQUET

    def _line(label: str, p: Path) -> str:
        return f"{label}: {p} (exists={p.is_file()})"

    lines = [
        _line("gaia_parquet", gaia),
        _line("twomass_parquet", twomass),
        "",
        "Example views:",
        f"CREATE OR REPLACE VIEW gaia AS SELECT * FROM read_parquet('{gaia.as_posix()}');",
        f"CREATE OR REPLACE VIEW twomass AS SELECT * FROM read_parquet('{twomass.as_posix()}');",
    ]
    return "\n".join(lines)


@astro_duck_toolset.tool
def suggest_astro_duck_parquet_views(
    gaia_parquet: str | None = None, twomass_parquet: str | None = None
) -> str:
    """Emit ``CREATE VIEW`` for Gaia and 2MASS Parquet using project defaults or given paths."""
    root = _root()
    g = resolve_parquet_sql_path(
        gaia_parquet, default_under_root=Path(_DEFAULT_GAIA_PARQUET)
    )
    t = resolve_parquet_sql_path(
        twomass_parquet, default_under_root=Path(_DEFAULT_TWOMASS_PARQUET)
    )
    if g is None:
        return json.dumps(
            {
                "error": "invalid_or_outside_project_gaia_path",
                "input": gaia_parquet,
            }
        )
    if t is None:
        return json.dumps(
            {
                "error": "invalid_or_outside_project_twomass_path",
                "input": twomass_parquet,
            }
        )
    return "\n".join(
        [
            f"CREATE OR REPLACE VIEW gaia AS SELECT * FROM read_parquet('{g.as_posix()}');",
            f"CREATE OR REPLACE VIEW twomass AS SELECT * FROM read_parquet('{t.as_posix()}');",
            f"-- gaia exists={g.is_file()} | twomass exists={t.is_file()}",
        ]
    )


@astro_duck_toolset.tool
def duckdb_parquet_preview_rows(relative_parquet: str, limit: int = 25) -> str:
    """Run DuckDB ``SELECT *`` on a project Parquet file (parameterized path; capped ``limit``)."""
    path = resolve_project_parquet(relative_parquet)
    if path is None:
        return json.dumps({"error": "invalid_or_missing_parquet", "path": relative_parquet})
    import duckdb

    try:
        lim = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        lim = 25
    con = duckdb.connect(database=":memory:")
    q = f"SELECT * FROM read_parquet(?) LIMIT {lim}"
    try:
        df = con.execute(q, [str(path)]).pl()
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(path)})
    return json.dumps(
        {
            "path": str(path),
            "limit": lim,
            "n_returned": len(df),
            "columns": df.columns,
            "rows": df.to_dicts(),
        },
        default=str,
    )


@astro_duck_toolset.tool
def duckdb_parquet_rowcount(relative_parquet: str) -> str:
    """``COUNT(*)`` for a project Parquet file via DuckDB."""
    path = resolve_project_parquet(relative_parquet)
    if path is None:
        return json.dumps({"error": "invalid_or_missing_parquet", "path": relative_parquet})
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(path)})
    return json.dumps({"path": str(path), "count": int(n)})


@astro_duck_toolset.tool
def duckdb_parquet_summarize(relative_parquet: str) -> str:
    """DuckDB ``SUMMARIZE`` over Parquet (column stats; scans file — avoid on huge tables in tight loops)."""
    path = resolve_project_parquet(relative_parquet)
    if path is None:
        return json.dumps({"error": "invalid_or_missing_parquet", "path": relative_parquet})
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        df = con.execute(
            "SUMMARIZE SELECT * FROM read_parquet(?)",
            [str(path)],
        ).df()
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(path)})
    return df.to_json(orient="split")
