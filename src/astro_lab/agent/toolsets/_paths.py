"""Project-root jail and safe path joins for agent tools."""

from __future__ import annotations

from pathlib import Path

from astro_lab.config import find_project_root


def project_root() -> Path:
    return find_project_root().resolve()


def resolve_project_parquet(rel_or_abs: str) -> Path | None:
    """Parquet file path that exists, resolved, and lies under the project root."""
    root = project_root()
    p = Path(rel_or_abs).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    if not p.is_file() or p.suffix.lower() != ".parquet":
        return None
    return p


def resolve_parquet_sql_path(rel_or_abs: str | None, *, default_under_root: Path) -> Path | None:
    """Resolved .parquet path under project (file may be missing). Used for SQL snippets only."""
    root = project_root()
    if rel_or_abs is None:
        p = default_under_root if default_under_root.is_absolute() else (root / default_under_root).resolve()
    else:
        p = Path(rel_or_abs).expanduser()
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    if p.suffix.lower() != ".parquet":
        return None
    return p


def safe_raw_survey_dir(raw_base: Path, survey_key: str) -> Path | None:
    """``raw_base / survey_key`` resolved; None if ``survey_key`` tries to leave ``raw_base``."""
    base = raw_base.resolve()
    if not survey_key or any(sep in survey_key for sep in ("/", "\\")):
        return None
    folder = (base / survey_key).resolve()
    try:
        folder.relative_to(base)
    except ValueError:
        return None
    return folder


def safe_raw_survey_file(raw_base: Path, survey_key: str, filename: str) -> Path | None:
    """``raw_base / survey_key / filename`` with no traversal; filename must be a single path segment."""
    folder = safe_raw_survey_dir(raw_base, survey_key)
    if folder is None or not folder.is_dir():
        return None
    if not filename or filename in (".", ".."):
        return None
    name = Path(filename)
    if name.name != filename:
        return None
    if any(p == ".." for p in name.parts):
        return None
    path = (folder / filename).resolve()
    try:
        path.relative_to(folder)
    except ValueError:
        return None
    return path if path.is_file() else None
