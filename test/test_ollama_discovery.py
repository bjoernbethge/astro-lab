"""Tests for Ollama model discovery (no real ollama daemon required)."""

from unittest.mock import patch

from astro_lab.agent.ollama_discovery import (
    is_cloud_model,
    last_cloud_model_from_ollama_list,
)


def test_is_cloud_model():
    assert is_cloud_model("glm-5.1:cloud")
    assert is_cloud_model("gemma4:31b-cloud")
    assert is_cloud_model("qwen3-coder:480b-cloud")
    assert not is_cloud_model("teuken-7b:latest")
    assert not is_cloud_model("llama3.2")


@patch("astro_lab.agent.ollama_discovery.subprocess.run")
def test_last_cloud_model_takes_last_in_list_order(mock_run):
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = """
NAME                               ID              SIZE      MODIFIED
glm-5.1:cloud                      59472abf9d0a    -         5 days ago
teuken-7b:latest                   32205a0c7eda    5.0 GB    6 days ago
nemotron-3-super:cloud             be3943c5a818    -         2 weeks ago
"""
    assert last_cloud_model_from_ollama_list() == "nemotron-3-super:cloud"
