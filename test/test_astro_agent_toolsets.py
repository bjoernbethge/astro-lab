"""Smoke tests for AstroLab Pydantic AI toolsets (no API keys)."""

import json

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from astro_lab.agent import (
    tool_has_mcp_apps_ui,
    ui_resource_uri_from_tool_metadata,
)
from astro_lab.agent.toolsets import (
    alcg_widgets_toolset,
    astro_duck_toolset,
    astrophot_toolset,
    survey_config_toolset,
)
from astro_lab.agent.toolsets.astro_duck import astro_duck_extension_and_marimo_pointers
from astro_lab.agent.toolsets.astrophot_tools import astrophot_version_info
from astro_lab.agent.toolsets.widget_tools import widgets_convert_radec_distance_to_cartesian
from astro_lab.agent.toolsets.survey_config import list_surveys
from astro_lab.agent.toolsets.training import train_jobs_list, train_prepare_background


def test_mcp_apps_ui_resource_uri_from_metadata():
    assert (
        ui_resource_uri_from_tool_metadata(
            {"_meta": {"ui": {"resourceUri": "ui://app/widget"}}}
        )
        == "ui://app/widget"
    )
    assert (
        ui_resource_uri_from_tool_metadata(
            {"meta": {"ui": {"resource_uri": "ui://legacy/x"}}}
        )
        == "ui://legacy/x"
    )
    assert tool_has_mcp_apps_ui({"_meta": {"ui": {"resourceUri": "ui://a"}}})
    assert not tool_has_mcp_apps_ui({})
    assert ui_resource_uri_from_tool_metadata(None) is None


def test_list_surveys_tool_plain_returns_text():
    out = list_surveys()
    assert isinstance(out, str)
    assert len(out) > 0


def test_astro_duck_extension_pointers_json():
    out = astro_duck_extension_and_marimo_pointers()
    data = json.loads(out)
    assert "install_load_sql" in data
    assert any("INSTALL astro" in s for s in data["install_load_sql"])


def test_astrophot_version_info_json():
    out = astrophot_version_info()
    data = json.loads(out)
    assert "installed" in data


def test_widgets_spherical_cartesian_json():
    out = widgets_convert_radec_distance_to_cartesian(0.0, 0.0, 100.0)
    data = json.loads(out)
    assert data["distance_pc"] == 100.0
    assert "x" in data and "y" in data and "z" in data


def test_survey_config_toolset_registers_with_test_model():
    test_model = TestModel()
    agent = Agent(test_model, toolsets=[survey_config_toolset])
    result = agent.run_sync("Call list_surveys and summarize in one sentence.")
    assert result is not None
    names = [t.name for t in test_model.last_model_request_parameters.function_tools]
    assert "list_surveys" in names


def test_astro_duck_toolset_registers():
    test_model = TestModel()
    agent = Agent(test_model, toolsets=[astro_duck_toolset])
    agent.run_sync("Call gaia_astro_duck_example_paths only.")
    names = [t.name for t in test_model.last_model_request_parameters.function_tools]
    assert "gaia_astro_duck_example_paths" in names


def test_astrophot_toolset_registers():
    test_model = TestModel()
    agent = Agent(test_model, toolsets=[astrophot_toolset])
    agent.run_sync("Call astrophot_list_raw_fits_candidates with survey gaia only.")
    names = [t.name for t in test_model.last_model_request_parameters.function_tools]
    assert "astrophot_list_raw_fits_candidates" in names


def test_train_jobs_list_json():
    out = json.loads(train_jobs_list(5))
    assert "jobs" in out and isinstance(out["jobs"], list)


def test_train_prepare_background_rejects_unknown_survey():
    data = json.loads(train_prepare_background("___not_a_real_survey___"))
    assert "error" in data


def test_alcg_widgets_toolset_registers():
    test_model = TestModel()
    agent = Agent(test_model, toolsets=[alcg_widgets_toolset])
    agent.run_sync("Call widgets_list_alcg_export_names only.")
    names = [t.name for t in test_model.last_model_request_parameters.function_tools]
    assert "widgets_list_alcg_export_names" in names
