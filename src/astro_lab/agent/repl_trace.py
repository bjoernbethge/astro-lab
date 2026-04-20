"""Format Pydantic AI run messages for terminal trace output (stderr)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, TextIO

from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)

def _clip(s: str, max_len: int) -> str:
    s = s.rstrip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"\n… ({len(s) - max_len} more characters)"


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _format_args(args: Any) -> str:
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, indent=2, default=str)
    except TypeError:
        return str(args)


def format_new_messages_text(
    messages: list[Any],
    *,
    max_tool_chars: int = 6000,
    prefix: str = "",
) -> str:
    """Human-readable trace of *new* messages from one agent run."""
    lines: list[str] = []
    pre = prefix or ""
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f"{pre}[user] {part.content!s}")
                elif isinstance(part, BaseToolReturnPart):
                    body = _clip(str(part.content), max_tool_chars)
                    lines.append(
                        f"{pre}[tool return] {part.tool_name} id={part.tool_call_id}\n{body}"
                    )
                else:
                    lines.append(f"{pre}[request part] {type(part).__name__}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, BaseToolCallPart):
                    lines.append(
                        f"{pre}[tool call] {part.tool_name} id={part.tool_call_id} "
                        f"{_format_args(part.args)}"
                    )
                elif isinstance(part, TextPart):
                    # Same text is printed on stdout as the run result; repeating here doubles output.
                    if part.content and _env_truthy("ASTROLAB_AGENT_TRACE_MODEL_TEXT"):
                        lines.append(f"{pre}[model text] {_clip(part.content, max_tool_chars)}")
                elif isinstance(part, ThinkingPart):
                    body = part.content or ""
                    if _env_truthy("ASTROLAB_AGENT_TRACE_THINKING"):
                        lines.append(f"{pre}[thinking] {_clip(body, max_tool_chars)}")
                    elif not body:
                        lines.append(f"{pre}[thinking] (empty)")
                    else:
                        lines.append(f"{pre}[thinking] {_clip(body, 360)}")
                else:
                    raw = repr(part)
                    lines.append(
                        f"{pre}[response part] {type(part).__name__} {_clip(raw, 400)}"
                    )
    return "\n".join(lines)


def print_agent_run_trace(
    result: Any,
    *,
    label: str = "agent",
    file: TextIO | None = None,
    max_tool_chars: int = 6000,
) -> None:
    """Print :attr:`new_messages` from an :class:`~pydantic_ai.run.AgentRunResult`."""
    out = file or sys.stderr
    msgs = getattr(result, "new_messages", lambda: [])()
    if not msgs:
        return
    banner = f"── {label} ({len(msgs)} message(s)) ──"
    print(banner, file=out)
    text = format_new_messages_text(
        msgs, max_tool_chars=max_tool_chars, prefix="  "
    )
    if text:
        print(text, file=out)
    print(file=out)
