"""
Test configuration and fixtures for AstroLab.

Uses real data from the data module for comprehensive testing.
"""

from pathlib import Path

import pytest
import torch

from astro_lab.data import AstroDataModule
from astro_lab.data.datasets import SurveyGraphDataset
from astro_lab.data.graphs import create_knn_graph
from astro_lab.models.lightning import create_lightning_model, list_lightning_models
from astro_lab.tensors import PhotometricTensorDict, SpatialTensorDict, SurveyTensorDict


@pytest.fixture(scope="session")
def data_dir():
    """Test data directory."""
    return Path("data")


@pytest.fixture(scope="session")
def processed_dir():
    """Processed data directory."""
    return Path("data/processed")


@pytest.fixture(scope="session")
def survey_configs():
    """Available survey configurations."""
    return ["gaia", "nsa", "exoplanet"]


@pytest.fixture(scope="session")
def small_survey():
    """Use GAIA as default survey for all tests."""
    return "gaia"


@pytest.fixture
def astro_datamodule(small_survey):
    """Create AstroDataModule with small dataset."""
    return AstroDataModule(
        survey=small_survey,
        max_samples=100,
        batch_size=1,
        num_workers=0,
        use_subgraph_sampling=True,
        max_nodes_per_graph=500,
    )


@pytest.fixture
def survey_graph_dataset(small_survey, processed_dir):
    """Create SurveyGraphDataset with real data."""
    return SurveyGraphDataset(
        root=str(processed_dir),
        survey=small_survey,
        graph_method="knn",
        k_neighbors=8,
        max_samples=100,
    )


@pytest.fixture
def sample_survey_tensor():
    """Create a sample SurveyTensorDict for testing."""
    # Real spatial data - create coordinates tensor properly
    n_objects = 50

    # Create spatial tensor with proper 3D coordinates [N, 3]
    coordinates = torch.randn(n_objects, 3, dtype=torch.float32)
    spatial_tensor = SpatialTensorDict(
        coordinates=coordinates, coordinate_system="icrs", unit="parsec", epoch=2000.0
    )

    # Real photometric data - create magnitudes tensor [N, B]
    magnitudes = torch.randn(n_objects, 3, dtype=torch.float32)  # 3 bands
    errors = torch.randn(n_objects, 3, dtype=torch.float32)
    bands = ["g", "r", "i"]

    photometric_tensor = PhotometricTensorDict(
        magnitudes=magnitudes,
        bands=bands,
        errors=errors,
        filter_system="AB",
        is_magnitude=True,
    )

    # Create survey tensor with proper parameters
    return SurveyTensorDict(
        spatial=spatial_tensor,
        photometric=photometric_tensor,
        survey_name="test_survey",
        data_release="test",
    )


@pytest.fixture
def lightning_model():
    """Create a simple Lightning model for testing."""
    models = list_lightning_models()
    if "survey_gnn" in models:
        return create_lightning_model("survey_gnn", hidden_dim=32, num_gnn_layers=2)
    else:
        pytest.skip("No survey_gnn model available")


@pytest.fixture
def graph_builders():
    """Available graph builders."""
    return {
        "knn": create_knn_graph,
    }


@pytest.fixture(scope="session")
def available_surveys():
    """Check which surveys have data available."""
    surveys = []
    for survey in ["gaia", "nsa", "exoplanet"]:
        # Check if processed data exists
        processed_path = Path(f"data/processed/{survey}")
        if processed_path.exists():
            surveys.append(survey)
    return surveys


@pytest.fixture
def real_graph_data(survey_graph_dataset):
    """Get real graph data from dataset."""
    if len(survey_graph_dataset) > 0:
        return survey_graph_dataset[0]
    else:
        pytest.skip("No real graph data available")


@pytest.fixture
def real_survey_tensor(survey_graph_dataset):
    """Get real SurveyTensorDict from dataset."""
    try:
        return survey_graph_dataset.get_survey_tensor()
    except FileNotFoundError:
        pytest.skip("No real SurveyTensorDict available")
