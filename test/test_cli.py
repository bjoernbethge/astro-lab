"""
Tests for CLI functionality

Tests the command-line interface of AstroLab including download,
preprocessing, and training commands.
"""

import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import io


class TestCLI:
    """Test CLI functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        # Use subprocess to avoid import side effects
        result = subprocess.run(
            [sys.executable, "-m", "astro_lab.cli", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "AstroLab - Astronomical Machine Learning Laboratory" in result.stdout

    def test_cli_version(self):
        """Test CLI version output."""
        result = subprocess.run(
            [sys.executable, "-m", "astro_lab.cli", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "AstroLab" in result.stdout

    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "astro_lab.cli"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Available Commands:" in result.stdout

    def test_download_help(self):
        """Test download command help."""
        result = subprocess.run(
            [sys.executable, "-m", "astro_lab.cli", "download", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "download" in result.stdout.lower()

    def test_download_gaia_command(self):
        """Test Gaia download command with mocking."""
        with patch('sys.argv', ['astro-lab', 'download', 'gaia', '--magnitude-limit', '10.0']):
            with patch('astro_lab.data.download_bright_all_sky') as mock_download:
                mock_download.return_value = "Download completed"
                with patch('astro_lab.cli.logger') as mock_logger:
                    from astro_lab.cli import main
                    # Wrap in try-except to handle any sys.exit calls
                    try:
                        main()
                    except SystemExit:
                        pass
                    mock_download.assert_called_once_with(magnitude_limit=10.0)

    def test_download_list_command(self):
        """Test list catalogs command."""
        with patch('sys.argv', ['astro-lab', 'download', 'list']):
            with patch('astro_lab.data.list_catalogs') as mock_list:
                mock_list.return_value = ["gaia", "sdss", "nsa"]
                with patch('astro_lab.cli.logger.info') as mock_logger:
                    from astro_lab.cli import main
                    try:
                        main()
                    except SystemExit:
                        pass
                    mock_list.assert_called_once()

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        with patch('sys.argv', ['astro-lab', 'download', 'gaia']):
            with patch('astro_lab.data.download_bright_all_sky') as mock_download:
                mock_download.side_effect = Exception("Download failed")
                with patch('sys.exit') as mock_exit:
                    from astro_lab.cli import main
                    main()
                    mock_exit.assert_called_with(1)

    def test_cli_invalid_command(self):
        """Test CLI with invalid command using subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "astro_lab.cli", "invalid-command"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 2

    def test_preprocess_help(self):
        """Test preprocess command help."""
        result = subprocess.run(
            [sys.executable, "-m", "astro_lab.cli", "preprocess", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "preprocess" in result.stdout.lower()

    def test_train_help(self):
        """Test train command help."""
        result = subprocess.run(
            [sys.executable, "-m", "astro_lab.cli", "train", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "train" in result.stdout.lower()

    def test_config_command(self):
        """Test config command functionality.""" 
        with patch('sys.argv', ['astro-lab', 'config', 'surveys']):
            # Mock at the module level where it's imported
            with patch('astro_lab.cli.handle_config') as mock_handle_config:
                # Import before patching
                from astro_lab.cli import main
                
                # Make the mock return successfully
                mock_handle_config.return_value = None
                
                try:
                    main()
                except SystemExit:
                    pass
                    
                # Check that handle_config was called with the correct args
                mock_handle_config.assert_called_once()
                args = mock_handle_config.call_args[0][0]
                assert hasattr(args, 'config_action')
                assert args.config_action == 'surveys' 