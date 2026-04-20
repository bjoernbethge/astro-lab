"""Survey name helpers for the AstroLab agent (aligned with configs/surveys.yaml)."""

from __future__ import annotations

from astro_lab.config import get_config


def survey_keys() -> list[str]:
    surveys = get_config().get("surveys", {})
    return sorted(surveys.keys())


def is_known_survey(name: str) -> bool:
    return name.lower() in {k.lower() for k in survey_keys()}


def normalize_survey(name: str) -> str | None:
    lower = {k.lower(): k for k in survey_keys()}
    return lower.get(name.lower())
