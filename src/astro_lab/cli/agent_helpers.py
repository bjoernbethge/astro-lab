"""Trace defaults for ``astro-lab agent``."""

from __future__ import annotations

import os
import sys
from typing import Any


def effective_agent_trace(args: Any) -> bool:
    """Whether to print tool/step traces to stderr."""
    if getattr(args, "no_trace", False):
        return False
    if getattr(args, "trace", False):
        return True
    env = os.environ.get("ASTROLAB_AGENT_TRACE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    interactive = getattr(args, "message", None) is None and sys.stdin.isatty()
    return interactive
