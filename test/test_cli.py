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
from unittest.mock import patch, MagicMock


class TestCLI:
    """Test CLI functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        with patch('sys.argv', ['astro-lab', '--help']):
            with patch('sys.exit') as mock_exit:
                # Import here to avoid import errors
                from astro_lab.cli import main as cli_main
                cli_main()
                # CLI may call sys.exit multiple times, which is normal
                mock_exit.assert_called_with(0)

    def test_cli_version(self):
        """Test CLI version output."""
        with patch('sys.argv', ['astro-lab', '--version']):
            with patch('sys.exit') as mock_exit:
                from astro_lab.cli import main as cli_main
                cli_main()
                mock_exit.assert_called_with(0)

    def test_cli_welcome_message(self):
        """Test CLI welcome message."""
        with patch('sys.argv', ['astro-lab']):
            with patch('builtins.print') as mock_print:
                with patch('sys.exit') as mock_exit:
                    from astro_lab.cli import main as cli_main
                    cli_main()
                    # Should print welcome message
                    mock_print.assert_called()
                    # CLI should show help and exit with 0 when no command is provided
                    mock_exit.assert_called_with(0)

    def test_download_help(self):
        """Test download command help."""
        with patch('sys.argv', ['astro-lab', 'download', '--help']):
            with patch('sys.exit') as mock_exit:
                from astro_lab.cli import main as cli_main
                cli_main()
                mock_exit.assert_called_with(0)

    def test_download_gaia_command(self):
        """Test Gaia download command."""
        with patch('sys.argv', ['astro-lab', 'download', 'gaia', '--magnitude-limit', '10.0']):
            with patch('astro_lab.data.download_bright_all_sky') as mock_download:
                mock_download.return_value = "Download completed"
                with patch('sys.exit') as mock_exit:
                    from astro_lab.cli import main as cli_main
                    cli_main()
                    mock_download.assert_called_once_with(magnitude_limit=10.0)
                    mock_exit.assert_not_called()

    def test_download_list_command(self):
        """Test list catalogs command."""
        with patch('sys.argv', ['astro-lab', 'download', 'list']):
            with patch('astro_lab.data.list_catalogs') as mock_list:
                mock_list.return_value = ["gaia", "sdss", "nsa"]
                with patch('sys.exit') as mock_exit:
                    from astro_lab.cli import main as cli_main
                    cli_main()
                    mock_list.assert_called_once()
                    mock_exit.assert_not_called()

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        with patch('sys.argv', ['astro-lab', 'download', 'gaia']):
            with patch('astro_lab.data.download_bright_all_sky') as mock_download:
                mock_download.side_effect = Exception("Download failed")
                with patch('sys.exit') as mock_exit:
                    from astro_lab.cli import main as cli_main
                    cli_main()
                    mock_exit.assert_called_once_with(1)

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        with patch('sys.argv', ['astro-lab', 'invalid-command']):
            with patch('sys.exit') as mock_exit:
                from astro_lab.cli import main as cli_main
                cli_main()
                # CLI may call sys.exit multiple times, check that it was called with error code 2
                mock_exit.assert_any_call(2)
                assert mock_exit.call_count >= 1

    def test_cli_argument_parsing(self):
        """Test CLI argument parsing."""
        # Test magnitude limit argument
        with patch('sys.argv', ['astro-lab', 'download', 'gaia', '--magnitude-limit', '15.5']):
            with patch('astro_lab.data.download_bright_all_sky') as mock_download:
                mock_download.return_value = "Download completed"
                with patch('sys.exit') as mock_exit:
                    from astro_lab.cli import main as cli_main
                    cli_main()
                    mock_download.assert_called_once_with(magnitude_limit=15.5)
                    mock_exit.assert_not_called()

    def test_cli_output_formatting(self):
        """Test CLI output formatting."""
        with patch('sys.argv', ['astro-lab', 'download', 'list']):
            with patch('astro_lab.data.list_catalogs') as mock_list:
                mock_list.return_value = ["gaia", "sdss", "nsa"]
                with patch('builtins.print') as mock_print:
                    with patch('sys.exit') as mock_exit:
                        from astro_lab.cli import main as cli_main
                        cli_main()
                        # Should print available datasets
                        mock_print.assert_called()
                        mock_exit.assert_not_called() 