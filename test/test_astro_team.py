"""Smoke tests for multi-agent AstroLab team (no API keys)."""

from unittest.mock import patch

from astro_lab.agent.team import create_astro_team


@patch("astro_lab.agent.team.resolve_model_id", return_value="ollama:llama3.2")
def test_astro_team_orchestrator_registers_delegation_tools(_mock_resolve: object) -> None:
    orch = create_astro_team()
    fs = orch._function_toolset
    names = {t.name for t in fs.tools.values()}
    assert names == {
        "consult_survey_expert",
        "consult_data_expert",
        "consult_training_expert",
        "consult_catalog_analysis_expert",
        "consult_duckdb_parquet_expert",
        "consult_visualization_interactive_expert",
        "consult_visualization_3d_pipeline_expert",
        "consult_imaging_fits_expert",
    }
