#!/usr/bin/env python3

"""CLI entry for the AstroLab Pydantic AI agent (mastermind team orchestrator)."""

from __future__ import annotations

import os
import sys

from astro_lab.agent import create_astro_team
from astro_lab.agent.trace_env import run_with_optional_trace
from astro_lab.cli.agent_helpers import effective_agent_trace
from astro_lab.cli.agent_repl import run_interactive_repl


def main(args) -> int:
    use_openai = getattr(args, "openai", False)
    provider = "openai" if use_openai else "ollama"

    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Use Ollama by default, or set the key for --openai.",
            file=sys.stderr,
        )
        return 1

    model = getattr(args, "model", None)
    include_mcp = not getattr(args, "no_mcp", False)
    mcp_config = getattr(args, "mcp_config", None)
    trace = effective_agent_trace(args)
    try:
        agent = create_astro_team(
            model=model,
            provider=provider,
            mcp_config=mcp_config,
            include_mcp=include_mcp,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    message = getattr(args, "message", None)
    if message:
        result = run_with_optional_trace(
            lambda: agent.run_sync(message),
            trace=trace,
            label="astro-lab agent (one-shot)",
            file=sys.stderr,
        )
        output = getattr(result, "output", None)
        if output is not None:
            print(output)
        else:
            print(result)
        return 0

    return run_interactive_repl(agent, trace=trace, provider=provider)
