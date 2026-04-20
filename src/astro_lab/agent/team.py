"""Multi-agent AstroLab team: orchestrator delegates to focused specialist agents (Pydantic AI)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic_ai import Agent, CombinedToolset, RunContext

from astro_lab.agent.mcp_client import load_mcp_toolsets, mcp_apps_agent_hint
from astro_lab.agent.model import resolve_model_id
from astro_lab.agent.ollama_discovery import ensure_ollama_base_url
from astro_lab.agent.repl_trace import print_agent_run_trace
from astro_lab.agent.toolsets import (
    alcg_widgets_toolset,
    albpy_widgets_toolset,
    alo3d_widgets_toolset,
    alpv_widgets_toolset,
    analysis_toolset,
    astro_duck_toolset,
    astrophot_toolset,
    data_pipeline_toolset,
    enhanced_widgets_toolset,
    plotly_widgets_toolset,
    safety_toolset,
    survey_config_toolset,
    training_toolset,
    widget_common_toolset,
)
from astro_lab.agent.trace_env import specialist_trace_enabled

ORCHESTRATOR_INSTRUCTIONS = """You are the AstroLab team lead. Route work to the **narrowest** specialist whose tools fit the question.

**Configuration & data acquisition**
- consult_survey_expert — surveys.yaml, list_surveys, show_survey_config, validate_config_paths
- consult_data_expert — download, preprocess, build-dataset, dataset_status

**Training**
- consult_training_expert — train CLI, HPO, background jobs (prepare → execute + job status/log tools)

**Analysis (tabular / catalogs / CLI hints — not rendering)**
- consult_catalog_analysis_expert — SurveyInfo JSON, star_catalog_quicklook_json, survey_parquet_inspection_json, suggest_cosmic_web_cmd, suggest_info_cmd
- consult_duckdb_parquet_expert — DuckDB on project Parquet (preview, rowcount, summarize), astro-duck SQL hints

**Visualization & backends (not raw catalog science)**
- consult_visualization_interactive_expert — Plotly + Cosmograph (plotly package, plotly_bridge, alcg exports), Parquet schema for plots, RA/Dec→xyz (widget_common)
- consult_visualization_3d_pipeline_expert — Blender AlbPy, PyVista (alpv), Open3D (alo3d), Enhanced tensor/ImageProcessor introspection

**Imaging (FITS on disk)**
- consult_imaging_fits_expert — AstroPhot on raw FITS (list candidates, fit); not for Parquet star lists

Optional MCP tools attach to this orchestrator only and are usable alongside the ``consult_*`` delegations.

Combine multiple specialists when the user spans domains. Do not invent survey names or ``astro-lab`` flags; use specialist tool outputs for CLI lines.
"""

_SURVEY_INSTRUCTIONS = """You are the AstroLab survey and configuration specialist.
Use your tools for facts. Answer concisely."""

_DATA_INSTRUCTIONS = """You are the AstroLab data pipeline specialist (download, preprocess, datasets).
Use your tools; suggest astro-lab CLI lines, do not claim you ran shell commands."""

_TRAINING_INSTRUCTIONS = """You are the AstroLab training specialist.
Workflow: train_prepare_background (stages plan) → train_execute_background after user confirms (REPL prompts y/N; non-TTY needs ASTROLAB_TRAINING_AUTO_APPROVE=1). Then train_job_status / train_job_log_tail / train_jobs_list. suggest_train_cmd when the user runs the CLI themselves."""

_CATALOG_ANALYSIS_INSTRUCTIONS = """You are the **catalog / survey analysis** specialist (on-disk status, tabular inspection, CLI hints).
Tools: survey_disk_status_json, survey_all_disk_status_json, survey_parquet_inspection_json, star_catalog_quicklook_json, suggest_cosmic_web_cmd, suggest_info_cmd.
For "show stars" / Sterne: call star_catalog_quicklook_json first; cite JSON (disk paths, preview rows, cosmic_web_command_example). Do not invent flags; cosmic-web is ``astro-lab cosmic-web <survey>`` plus optional --visualize / --max-samples only.
You do **not** have DuckDB or widget/plotting tools — delegate those domains to the team lead if needed."""

_DUCKDB_INSTRUCTIONS = """You are the **DuckDB / Parquet SQL** specialist for paths under the project root.
Tools: astro_duck_extension_and_marimo_pointers, gaia_astro_duck_example_paths, suggest_astro_duck_parquet_views, duckdb_parquet_preview_rows, duckdb_parquet_rowcount, duckdb_parquet_summarize.
For survey disk layout or YAML, the catalog specialist handles that — focus on SQL/Parquet here."""

_VIZ_INTERACTIVE_INSTRUCTIONS = """You are the **interactive web visualization** specialist (Plotly, Cosmograph, plot-oriented Parquet/coords).
Tools: widgets_list_plotly_bridge_functions, widgets_list_plotly_package_exports, widgets_list_alcg_export_names, widgets_parquet_schema_for_viz, widgets_first_survey_parquet_schema, widgets_convert_radec_distance_to_cartesian.
Do not use SurveyInfo or DuckDB execute tools — ask the team lead to route catalog or SQL questions elsewhere."""

_VIZ_3D_INSTRUCTIONS = """You are the **3D / engine pipeline** specialist (Blender AlbPy, PyVista, Open3D, Enhanced tensor/image helpers).
Tools: widgets_list_albpy_export_names, widgets_list_alpv_export_names, widgets_list_alo3d_export_names, widgets_image_processor_method_summary.
You do not run Blender or heavy imports; only AST/introspection. For FITS photometry fitting, that is the imaging specialist."""

_IMAGING_FITS_INSTRUCTIONS = """You are the **FITS imaging / AstroPhot** specialist (2D images under data/raw).
Tools: astrophot_version_info, astrophot_list_raw_fits_candidates, astrophot_fit_from_raw_fits.
Catalog/table FITS → catalog specialist + DuckDB, not AstroPhot."""


def _survey_toolsets() -> CombinedToolset:
    return CombinedToolset([safety_toolset, survey_config_toolset])


def _data_toolsets() -> CombinedToolset:
    return CombinedToolset([safety_toolset, data_pipeline_toolset])


def _training_toolsets() -> CombinedToolset:
    return CombinedToolset([safety_toolset, training_toolset])


def _catalog_analysis_toolsets() -> CombinedToolset:
    return CombinedToolset([safety_toolset, analysis_toolset])


def _duckdb_toolsets() -> CombinedToolset:
    return CombinedToolset([safety_toolset, astro_duck_toolset])


def _viz_interactive_toolsets() -> CombinedToolset:
    return CombinedToolset(
        [
            safety_toolset,
            widget_common_toolset,
            plotly_widgets_toolset,
            alcg_widgets_toolset,
        ]
    )


def _viz_3d_toolsets() -> CombinedToolset:
    return CombinedToolset(
        [
            safety_toolset,
            albpy_widgets_toolset,
            alpv_widgets_toolset,
            alo3d_widgets_toolset,
            enhanced_widgets_toolset,
        ]
    )


def _imaging_fits_toolsets() -> CombinedToolset:
    return CombinedToolset([safety_toolset, astrophot_toolset])


def _run_specialist(
    agent: Agent,
    question: str,
    *,
    usage: object,
    trace_label: str,
) -> str:
    r = agent.run_sync(question, usage=usage)
    if specialist_trace_enabled():
        print_agent_run_trace(r, label=trace_label, file=sys.stderr)
    return str(r.output)


def create_astro_team(
    model: str | None = None,
    *,
    provider: Literal["ollama", "openai"] | None = None,
    mcp_config: str | Path | None = None,
    include_mcp: bool = True,
) -> Agent:
    """Create an orchestrator :class:`~pydantic_ai.Agent` with eight focused specialist delegates.

    Each specialist shares the same model as the orchestrator (from :func:`resolve_model_id`).
    Child runs forward ``usage=ctx.usage`` for unified accounting.

    MCP servers (if any) attach to the **orchestrator** only.
    """
    ensure_ollama_base_url()
    model_id = resolve_model_id(model, provider=provider)
    mcp_toolsets = load_mcp_toolsets(mcp_config) if include_mcp else []

    survey_agent = Agent(
        model_id,
        instructions=_SURVEY_INSTRUCTIONS,
        toolsets=[_survey_toolsets()],
    )
    data_agent = Agent(
        model_id,
        instructions=_DATA_INSTRUCTIONS,
        toolsets=[_data_toolsets()],
    )
    training_agent = Agent(
        model_id,
        instructions=_TRAINING_INSTRUCTIONS,
        toolsets=[_training_toolsets()],
    )
    catalog_analysis_agent = Agent(
        model_id,
        instructions=_CATALOG_ANALYSIS_INSTRUCTIONS,
        toolsets=[_catalog_analysis_toolsets()],
    )
    duckdb_agent = Agent(
        model_id,
        instructions=_DUCKDB_INSTRUCTIONS,
        toolsets=[_duckdb_toolsets()],
    )
    viz_interactive_agent = Agent(
        model_id,
        instructions=_VIZ_INTERACTIVE_INSTRUCTIONS,
        toolsets=[_viz_interactive_toolsets()],
    )
    viz_3d_agent = Agent(
        model_id,
        instructions=_VIZ_3D_INSTRUCTIONS,
        toolsets=[_viz_3d_toolsets()],
    )
    imaging_fits_agent = Agent(
        model_id,
        instructions=_IMAGING_FITS_INSTRUCTIONS,
        toolsets=[_imaging_fits_toolsets()],
    )

    orch_instructions = ORCHESTRATOR_INSTRUCTIONS
    if mcp_toolsets:
        orch_instructions = (
            f"{ORCHESTRATOR_INSTRUCTIONS.rstrip()}\n\n{mcp_apps_agent_hint()}"
        )

    if mcp_toolsets:
        orchestrator = Agent(
            model_id,
            instructions=orch_instructions,
            toolsets=mcp_toolsets,
        )
    else:
        orchestrator = Agent(model_id, instructions=orch_instructions)

    @orchestrator.tool
    def consult_survey_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the survey/config specialist (YAML, paths, list_surveys)."""
        return _run_specialist(
            survey_agent, question, usage=ctx.usage, trace_label="specialist: survey"
        )

    @orchestrator.tool
    def consult_data_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the data pipeline specialist (download, preprocess, datasets)."""
        return _run_specialist(
            data_agent, question, usage=ctx.usage, trace_label="specialist: data"
        )

    @orchestrator.tool
    def consult_training_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the training specialist."""
        return _run_specialist(
            training_agent, question, usage=ctx.usage, trace_label="specialist: training"
        )

    @orchestrator.tool
    def consult_catalog_analysis_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the catalog / SurveyInfo / tabular analysis specialist."""
        return _run_specialist(
            catalog_analysis_agent,
            question,
            usage=ctx.usage,
            trace_label="specialist: catalog_analysis",
        )

    @orchestrator.tool
    def consult_duckdb_parquet_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the DuckDB / Parquet SQL specialist."""
        return _run_specialist(
            duckdb_agent, question, usage=ctx.usage, trace_label="specialist: duckdb"
        )

    @orchestrator.tool
    def consult_visualization_interactive_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the Plotly / Cosmograph / plot-schema specialist."""
        return _run_specialist(
            viz_interactive_agent,
            question,
            usage=ctx.usage,
            trace_label="specialist: viz_interactive",
        )

    @orchestrator.tool
    def consult_visualization_3d_pipeline_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the AlbPy / PyVista / Open3D / Enhanced pipeline specialist."""
        return _run_specialist(
            viz_3d_agent, question, usage=ctx.usage, trace_label="specialist: viz_3d"
        )

    @orchestrator.tool
    def consult_imaging_fits_expert(ctx: RunContext[None], question: str) -> str:
        """Ask the FITS / AstroPhot imaging specialist."""
        return _run_specialist(
            imaging_fits_agent, question, usage=ctx.usage, trace_label="specialist: imaging_fits"
        )

    return orchestrator
