"""Safety and CLI validation helpers."""

from __future__ import annotations

import re

from pydantic_ai import FunctionToolset

from .._surveys import is_known_survey, survey_keys

safety_toolset = FunctionToolset()

_ALLOWED_SUBCOMMANDS = frozenset(
    {
        "download",
        "preprocess",
        "train",
        "info",
        "cosmic-web",
        "build-dataset",
        "optimize",
        "hpo",
        "config",
        "process",
    }
)


@safety_toolset.tool
def explain_cli_safety() -> str:
    """Describe AstroLab agent safety rules for CLI suggestions."""
    return (
        "Only use 'astro-lab' with allow-listed subcommands. "
        f"Allowed: {', '.join(sorted(_ALLOWED_SUBCOMMANDS))}. "
        "Survey names must match configured surveys. "
        "Do not run rm, curl to unknown hosts, or arbitrary scripts."
    )


@safety_toolset.tool
def validate_cli_plan(subcommand: str, survey: str | None = None) -> str:
    """Check subcommand (and optional survey) against allow-lists; return OK or error message."""
    sub = subcommand.strip().lower()
    if sub not in _ALLOWED_SUBCOMMANDS:
        return (
            f"REJECTED: subcommand '{subcommand}' not allowed. "
            f"Allowed: {', '.join(sorted(_ALLOWED_SUBCOMMANDS))}"
        )
    if survey is None:
        return f"OK: subcommand '{sub}' is allowed (no survey checked)."
    if not is_known_survey(survey):
        return (
            f"REJECTED: unknown survey '{survey}'. Known: {', '.join(survey_keys())}"
        )
    return f"OK: astro-lab {sub} {survey} passes static validation."


@safety_toolset.tool
def sanitize_one_liner(command: str) -> str:
    """Reject if the string looks like chained shell or disallowed tokens; otherwise echo trimmed line."""
    trimmed = command.strip()
    if not trimmed:
        return "REJECTED: empty command"
    if ".." in trimmed:
        return "REJECTED: parent-directory segments (..) are not allowed"
    lower = trimmed.lower()
    if any(
        bad in lower
        for bad in (
            ";",
            "&&",
            "||",
            "| ",
            "`",
            "$(",  # noqa: S105 - pattern not a password
            "rm ",
            "format ",
            "del ",
            "shutdown",
        )
    ):
        return "REJECTED: command contains disallowed shell patterns or risky tokens"
    if not re.match(r"^astro-lab(\s+[\w\-./]+)*$", trimmed):
        return "REJECTED: must be a simple astro-lab single-line invocation (no shell metacharacters)"
    return trimmed
