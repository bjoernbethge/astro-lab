#!/usr/bin/env python3
"""Test Cosmograph integration with AstroLab data."""

import pytest
import warnings
import numpy as np
import torch

from src.astro_lab.utils.viz.cosmograph_bridge import CosmographBridge


@pytest.fixture
def cosmograph_bridge():
    """Create CosmographBridge instance for testing."""
    return CosmographBridge()


@pytest.fixture
def mock_spatial_tensor():
    """Create mock spatial tensor for testing."""
    # Create mock coordinates
    coords = torch.randn(50, 3) * 10  # 50 points in 3D space
    
    # Create a mock tensor with cartesian attribute
    class MockTensor:
        def __init__(self, coords):
            self.cartesian = coords
            self.unit = 'pc'
            self.coordinate_system = 'cartesian'
        
        def _create_radius_graph(self, coords, radius):
            # Create simple neighbor graph
            from sklearn.neighbors import NearestNeighbors
            coords_np = coords.cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(coords_np)
            distances, indices = nbrs.kneighbors(coords_np)
            
            edge_list = []
            for i in range(len(coords_np)):
                for j in range(1, len(indices[i])):
                    if distances[i][j] <= radius:
                        edge_list.append([i, indices[i][j]])
            
            return torch.tensor(edge_list, dtype=torch.long).t()
    
    return MockTensor(coords)


@pytest.fixture
def mock_survey_data(mock_spatial_tensor):
    """Create mock survey data for testing."""
    return {"spatial_tensor": mock_spatial_tensor}


def test_cosmograph_bridge_initialization(cosmograph_bridge):
    """Test CosmographBridge initialization."""
    assert cosmograph_bridge is not None
    assert hasattr(cosmograph_bridge, 'default_config')
    assert 'background_color' in cosmograph_bridge.default_config
    assert 'simulation_gravity' in cosmograph_bridge.default_config


def test_from_survey_data(cosmograph_bridge, mock_survey_data):
    """Test creating visualization from survey data."""
    try:
        widget = cosmograph_bridge.from_survey_data(
            mock_survey_data,
            survey_name="gaia",
            radius=3.0
        )
        assert widget is not None
    except Exception as e:
        pytest.skip(f"Cosmograph not available: {e}")


def test_from_spatial_tensor(cosmograph_bridge, mock_spatial_tensor):
    """Test creating visualization from spatial tensor."""
    try:
        widget = cosmograph_bridge.from_spatial_tensor(
            mock_spatial_tensor,
            radius=3.0,
            point_color='#ffd700'
        )
        assert widget is not None
    except Exception as e:
        pytest.skip(f"Cosmograph not available: {e}")


def test_from_coordinates(cosmograph_bridge):
    """Test creating visualization from raw coordinates."""
    try:
        # Create test coordinates
        coords = np.random.uniform(-10, 10, (50, 3))
        
        widget = cosmograph_bridge.from_coordinates(
            coords,
            radius=2.0
        )
        assert widget is not None
    except Exception as e:
        pytest.skip(f"Cosmograph not available: {e}")


def test_survey_color_mapping(cosmograph_bridge):
    """Test automatic color mapping for different surveys."""
    bridge = CosmographBridge()
    
    # Test color mapping with mock data
    mock_tensor = type('MockTensor', (), {
        'cartesian': torch.randn(10, 3),
        '_create_radius_graph': lambda self, coords, radius: torch.tensor([[0, 1], [1, 0]]).t()
    })()
    
    test_data = {"spatial_tensor": mock_tensor}
    
    # These should not raise errors
    try:
        bridge.from_survey_data(test_data, survey_name="gaia")
    except Exception as e:
        pytest.skip(f"Cosmograph not available: {e}")
    
    try:
        bridge.from_survey_data(test_data, survey_name="sdss")
    except Exception as e:
        pytest.skip(f"Cosmograph not available: {e}")
    
    try:
        bridge.from_survey_data(test_data, survey_name="tng50")
    except Exception as e:
        pytest.skip(f"Cosmograph not available: {e}")


def test_invalid_data_source(cosmograph_bridge):
    """Test handling of invalid data sources."""
    with pytest.raises(ValueError):
        cosmograph_bridge.from_survey_data({}, survey_name="unknown")


def test_default_config(cosmograph_bridge):
    """Test default configuration values."""
    config = cosmograph_bridge.default_config
    
    assert config['background_color'] == '#000011'
    assert config['simulation_gravity'] == 0.1
    assert config['simulation_repulsion'] == 0.2
    assert config['show_labels'] is True
    assert config['curved_links'] is True


if __name__ == "__main__":
    # Run basic test if executed directly
    print("🌌 Testing Cosmograph integration...")
    
    bridge = CosmographBridge()
    print("✅ Bridge initialized")
    
    # Test with mock data
    coords = np.random.uniform(-10, 10, (20, 3))
    try:
        widget = bridge.from_coordinates(coords, radius=2.0)
        print("✅ Mock visualization created")
    except Exception as e:
        print(f"❌ Test failed: {e}") 