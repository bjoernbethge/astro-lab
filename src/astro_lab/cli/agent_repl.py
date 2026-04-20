"""Interactive REPL for ``astro-lab agent`` (Rich + prompt_toolkit)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from astro_lab.agent.trace_env import run_with_optional_trace


def _history_file() -> Path:
    path = Path.home() / ".astro-lab" / "agent_repl_history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _print_welcome(console: Console, *, trace: bool, provider: str) -> None:
    title = "AstroLab mastermind"
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column()
    table.add_row("Mode", escape("Team orchestrator"))
    table.add_row("Provider", escape(provider))
    table.add_row("Exit", escape(":q · exit · quit · Ctrl+D"))
    table.add_row(
        "Training",
        escape("Background jobs ask y/N unless ASTROLAB_TRAINING_AUTO_APPROVE=1"),
    )
    trace_note = "off (--no-trace or ASTROLAB_AGENT_TRACE=0)"
    if trace:
        trace_note = (
            "on → stderr: user / thinking preview / tools "
            "(final reply only on stdout; ASTROLAB_AGENT_TRACE_MODEL_TEXT=1 to mirror it in trace)"
        )
    table.add_row("Tool trace", escape(trace_note))

    panel = Panel(
        table,
        title=f"[bold bright_blue]{title}[/]",
        subtitle="[dim]↑↓ history · multiline paste supported[/dim]",
        border_style="bright_blue",
        expand=False,
    )
    console.print(panel)
    console.print()


def run_interactive_repl(agent: Any, *, trace: bool, provider: str) -> int:
    """TTY loop with persistent history and styled prompt."""
    console = Console(file=sys.stdout, highlight=False)
    _print_welcome(console, trace=trace, provider=provider)

    session = PromptSession(
        history=FileHistory(str(_history_file())),
        style=Style.from_dict(
            {
                "": "#c8c8c8",
            }
        ),
    )
    prompt = HTML(
        "<ansigreen><b>astro-lab</b></ansigreen> "
        "<ansibrightblack>agent</ansibrightblack> <ansicyan>&gt;</ansicyan> "
    )
    label = "astro-lab agent (orchestrator)"

    while True:
        try:
            line = session.prompt(prompt).strip()
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print()
            return 0

        if line.lower() in {":q", "exit", "quit"}:
            return 0
        if not line:
            continue

        result = run_with_optional_trace(
            lambda: agent.run_sync(line),
            trace=trace,
            label=label,
            file=sys.stderr,
        )
        output = getattr(result, "output", None)
        console.print(output if output is not None else result)
