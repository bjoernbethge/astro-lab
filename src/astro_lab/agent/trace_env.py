"""Temporary ``ASTROLAB_AGENT_TRACE`` env for nested specialist runs."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

T = TypeVar("T")


@contextmanager
def agent_trace_env(enabled: bool) -> Generator[None, None, None]:
    key = "ASTROLAB_AGENT_TRACE"
    prior = os.environ.get(key)
    try:
        os.environ[key] = "1" if enabled else "0"
        yield
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


def specialist_trace_enabled() -> bool:
    return os.environ.get("ASTROLAB_AGENT_TRACE", "0").strip() == "1"


def run_with_optional_trace(
    fn: Callable[[], T],
    *,
    trace: bool,
    label: str,
    file: Any | None = None,
) -> T:
    """Run ``fn`` (usually ``agent.run_sync``); print trace and set env for nested specialists."""
    import sys

    from astro_lab.agent.repl_trace import print_agent_run_trace

    out = file or sys.stderr
    with agent_trace_env(trace):
        result = fn()
    if trace:
        print_agent_run_trace(result, label=label, file=out)
    return result
