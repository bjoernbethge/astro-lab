"""AstroLab Pydantic AI agent (Ollama by default, optional OpenAI + FunctionToolsets)."""

from .mcp_client import (
    MCP_APPS_LANDING_URL,
    MCP_APPS_STANDARD_PR,
    default_mcp_config_path,
    load_mcp_toolsets,
    mcp_apps_agent_hint,
    tool_has_mcp_apps_ui,
    ui_resource_uri_from_tool_metadata,
)
from .model import resolve_model_id
from .ollama_discovery import (
    DEFAULT_OLLAMA_BASE_URL,
    ensure_ollama_base_url,
    is_cloud_model,
    last_cloud_model_from_ollama_list,
)
from .team import ORCHESTRATOR_INSTRUCTIONS, create_astro_team

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "MCP_APPS_LANDING_URL",
    "MCP_APPS_STANDARD_PR",
    "ORCHESTRATOR_INSTRUCTIONS",
    "create_astro_team",
    "default_mcp_config_path",
    "ensure_ollama_base_url",
    "is_cloud_model",
    "last_cloud_model_from_ollama_list",
    "load_mcp_toolsets",
    "mcp_apps_agent_hint",
    "resolve_model_id",
    "tool_has_mcp_apps_ui",
    "ui_resource_uri_from_tool_metadata",
]
