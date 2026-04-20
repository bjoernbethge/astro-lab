"""Discover Ollama models from the local ``ollama`` CLI."""

from __future__ import annotations

import os
import subprocess
from typing import Final

DEFAULT_OLLAMA_BASE_URL: Final[str] = "http://127.0.0.1:11434/v1"


def is_cloud_model(model_name: str) -> bool:
    """True for Ollama cloud-tagged models (name contains ``:cloud`` or ``-cloud``)."""
    return ":cloud" in model_name or "-cloud" in model_name


def last_cloud_model_from_ollama_list() -> str | None:
    """Run ``ollama list`` and return the last cloud model in display order, or None."""
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.strip().splitlines()
    if len(lines) < 2:
        return None
    cloud: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        if is_cloud_model(name):
            cloud.append(name)
    return cloud[-1] if cloud else None


def ensure_ollama_base_url() -> None:
    """Set ``OLLAMA_BASE_URL`` if unset (required by Pydantic AI's OllamaProvider)."""
    os.environ.setdefault("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
