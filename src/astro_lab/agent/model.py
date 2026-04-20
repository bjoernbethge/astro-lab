"""Provider / model resolution for AstroLab Pydantic AI agents."""

from __future__ import annotations

import os
from typing import Literal

from astro_lab.agent.ollama_discovery import (
    ensure_ollama_base_url,
    last_cloud_model_from_ollama_list,
)


def _resolve_provider(
    provider: Literal["ollama", "openai"] | None,
) -> Literal["ollama", "openai"]:
    env = (provider or os.environ.get("ASTROLAB_AGENT_PROVIDER") or "ollama").lower()
    if env in ("openai", "ollama"):
        return env  # type: ignore[return-value]
    return "ollama"


def resolve_model_id(
    model: str | None = None,
    *,
    provider: Literal["ollama", "openai"] | None = None,
) -> str:
    """Return a ``provider:model`` string for :class:`pydantic_ai.Agent` (Ollama default)."""
    prov = _resolve_provider(provider)

    if prov == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is required when using the OpenAI provider "
                "(set ASTROLAB_AGENT_PROVIDER=openai or pass provider='openai')."
            )
        name = model or os.environ.get("ASTROLAB_AGENT_MODEL", "gpt-4o-mini")
        if name.startswith("openai:"):
            return name
        return f"openai:{name}"

    ensure_ollama_base_url()
    if model:
        raw = model
    else:
        raw = os.environ.get("ASTROLAB_AGENT_MODEL")
    if not raw:
        raw = last_cloud_model_from_ollama_list() or "llama3.2"
    if raw.startswith("ollama:"):
        return raw
    return f"ollama:{raw}"
