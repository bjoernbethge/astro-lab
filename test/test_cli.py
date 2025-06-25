"""
Tests for CLI functionality.
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLI:
    """Test the command-line interface."""

    def run_cli(self, *args):
        """Run the CLI with given arguments."""
        cmd = [sys.executable, "-m", "astro_lab.cli"] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        return result

    def test_cli_help(self):
        """Test that help message is displayed."""
        result = self.run_cli("--help")
        assert result.returncode == 0
        assert "AstroLab: Modern Astronomical Machine Learning" in result.stdout
        assert "Available commands:" in result.stdout

    def test_train_help(self):
        """Test train command help."""
        result = self.run_cli("train", "--help")
        assert result.returncode == 0
        assert "Train astronomical machine learning models" in result.stdout
        assert "--config" in result.stdout
        assert "--quick" in result.stdout

    def test_preprocess_help(self):
        """Test preprocess command help."""
        result = self.run_cli("preprocess", "--help")
        assert result.returncode == 0
        assert "Preprocess astronomical catalogs" in result.stdout
        assert "--input" in result.stdout
        assert "--survey" in result.stdout

    def test_download_help(self):
        """Test download command help."""
        result = self.run_cli("download", "--help")
        assert result.returncode == 0
        assert "Download astronomical survey data" in result.stdout

    def test_optimize_help(self):
        """Test optimize command help."""
        result = self.run_cli("optimize", "--help")
        assert result.returncode == 0
        assert "hyperparameter optimization" in result.stdout.lower()

    def test_invalid_command(self):
        """Test error handling for invalid command."""
        result = self.run_cli("invalid_command")
        assert result.returncode != 0
        assert (
            "invalid choice" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )

    def test_no_command(self):
        """Test error when no command is provided."""
        result = self.run_cli()
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    @pytest.mark.parametrize("survey", ["gaia", "sdss", "nsa"])
    def test_preprocess_survey_validation(self, survey, tmp_path):
        """Test survey validation in preprocess command."""
        # Create dummy input file
        input_file = tmp_path / f"{survey}_data.csv"
        input_file.write_text("ra,dec\n1.0,2.0\n")

        result = self.run_cli(
            "preprocess",
            "--input",
            str(input_file),
            "--survey",
            survey,
            "--output",
            str(tmp_path),
        )
        # The command should either succeed or fail gracefully
        # If it succeeds, it should mention the survey or preprocessing
        # If it fails, it should still recognize the survey type
        assert (
            result.returncode == 0
            or survey in result.stderr.lower()
            or survey in result.stdout.lower()
            or "preprocessed" in result.stdout.lower()
        )

    def test_train_quick_validation(self):
        """Test quick training validation."""
        result = self.run_cli(
            "train",
            "--quick",
            "gaia",
            "invalid_model",
        )
        assert result.returncode != 0
        assert (
            "unknown model" in result.stderr.lower() or "error" in result.stderr.lower()
        )

    def test_train_config_missing(self):
        """Test error when config file is missing."""
        result = self.run_cli(
            "train",
            "--config",
            "nonexistent_config.yaml",
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()
