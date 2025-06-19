"""
Tests for utility imports and basic utility functions.

Tests basic imports and utility information functions.
"""

import pytest
from astro_lab.utils import (
    GRAPH_AVAILABLE,
    get_utils_info,
)

# Test configuration
BLENDER_UTILS_AVAILABLE = False  # Deprecated - using tensor-based architecture


class TestUtilsImports:
    """Test basic utility imports."""

    def test_import_graph_utils(self):
        """Test graph utility imports."""
        if GRAPH_AVAILABLE:
            from astro_lab.utils import create_spatial_graph, calculate_graph_metrics
            assert callable(create_spatial_graph)
            assert callable(calculate_graph_metrics)

    def test_import_utils_info(self):
        """Test utility info function."""
        info = get_utils_info()
        assert isinstance(info, dict)
        assert "graph_available" in info 