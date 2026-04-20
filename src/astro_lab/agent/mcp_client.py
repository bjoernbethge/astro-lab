"""MCP client helpers: Pydantic AI server toolsets and MCP Apps (interactive UI) metadata.

`MCP Apps`_ standardize interactive tool UIs over MCP: tools may advertise
``_meta.ui.resourceUri`` so hosts can load HTML UI in sandboxed iframes (see also
`mcpui.dev`_).

.. _MCP Apps: https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1865
.. _mcpui.dev: https://mcpui.dev/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic_ai.mcp import (
    MCPServerSSE,
    MCPServerStdio,
    MCPServerStreamableHTTP,
    load_mcp_servers,
)

MCP_APPS_LANDING_URL = "https://mcpui.dev/"
MCP_APPS_STANDARD_PR = (
    "https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1865"
)


def ui_resource_uri_from_tool_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Return the MCP Apps UI resource URI from Pydantic AI / MCP tool metadata, if any.

    Hosts resolve this URI (often ``ui://...``) to HTML or other UI resources per the
    MCP Apps spec. Metadata may use ``_meta`` (wire format) or ``meta`` (some clients).
    """
    if not metadata:
        return None
    meta = metadata.get("_meta")
    if meta is None:
        meta = metadata.get("meta")
    if not isinstance(meta, dict):
        return None
    ui = meta.get("ui")
    if not isinstance(ui, dict):
        return None
    uri = ui.get("resourceUri") or ui.get("resource_uri")
    if uri is None:
        return None
    return str(uri)


def tool_has_mcp_apps_ui(metadata: dict[str, Any] | None) -> bool:
    """Whether tool metadata declares an MCP Apps UI resource."""
    return ui_resource_uri_from_tool_metadata(metadata) is not None


def default_mcp_config_path(project_root: Path | None = None) -> Path | None:
    """Resolve MCP JSON config path: ``ASTROLAB_MCP_CONFIG``, then ``configs/mcp.json``, ``mcp.json``."""
    env = (os.environ.get("ASTROLAB_MCP_CONFIG") or "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None

    root = project_root or _project_root()
    for candidate in (root / "configs" / "mcp.json", root / "mcp.json"):
        if candidate.is_file():
            return candidate
    return None


def _project_root() -> Path:
    from astro_lab.config import find_project_root

    return find_project_root()


def load_mcp_toolsets(
    config_path: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE]:
    """Load MCP servers as Pydantic AI toolsets (``MCPServerStdio`` / HTTP / SSE).

    Uses :func:`pydantic_ai.mcp.load_mcp_servers` on a JSON file with top-level
    ``mcpServers`` (Cursor / Claude Desktop style). Environment expansion
    ``${VAR}`` / ``${VAR:-default}`` is supported by Pydantic AI.

    Returns an empty list if no config path resolves or the file is missing.
    """
    if config_path is not None:
        path = Path(config_path).expanduser()
        if not path.is_file():
            return []
    else:
        path = default_mcp_config_path(project_root)
        if path is None:
            return []
    return load_mcp_servers(path)


def mcp_apps_agent_hint() -> str:
    """One paragraph for system instructions: MCP Apps + where AstroLab loads remote tools."""
    return (
        "Optional **MCP** remote tools: when `configs/mcp.json` or `ASTROLAB_MCP_CONFIG` "
        "is set, this agent also exposes those MCP servers' tools (Streamable HTTP, SSE, or stdio per config). "
        "Under **MCP Apps**, tools may include `_meta.ui.resourceUri` for interactive UI in supporting hosts; "
        f"see {MCP_APPS_LANDING_URL}"
    )
