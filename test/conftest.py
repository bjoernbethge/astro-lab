"""
Test configuration and fixtures for AstroLab.

Uses real data from the data module for comprehensive testing.
"""

from pathlib import Path

import polars as pl
import pytest
import torch
from torch_geometric.data import Data

from astro_lab.config import get_data_config
from astro_lab.data import (
    AstroDataModule,
    SurveyDataModule,
    create_graph_from_survey,
    get_preprocessor,
)
from astro_lab.data.dataset import SurveyGraphDataset
from astro_lab.models import AstroModel

# Get central data configuration
data_config = get_data_config()


@pytest.fixture(scope="session")
def data_dir():
    """Test data directory."""
    return Path(data_config["base_dir"])


@pytest.fixture(scope="session")
def processed_dir():
    """Processed data directory."""
    return Path(data_config["processed_dir"])


@pytest.fixture(scope="session")
def survey_configs():
    """Available survey configurations."""
    return ["gaia", "nsa", "exoplanet", "sdss", "twomass"]


@pytest.fixture(scope="session")
def small_survey():
    """Use GAIA as default survey for all tests."""
    return "gaia"


@pytest.fixture
def astro_datamodule(small_survey):
    """Create SurveyDataModule with small dataset."""
    return SurveyDataModule(
        survey=small_survey,
        task="node_classification",
        max_samples=100,
        batch_size=1,
        num_workers=0,
    )


@pytest.fixture
def survey_graph_dataset(small_survey, processed_dir):
    """Create SurveyGraphDataset with real data."""
    return SurveyGraphDataset(
        root=str(processed_dir),
        survey=small_survey,
        task="node_classification",
        k_neighbors=8,
        max_samples=100,
    )


@pytest.fixture
def sample_graph_data():
    """Create a sample PyG Data object for testing."""
    n_nodes = 50
    n_features = 10
    n_edges = 200

    # Create node features
    x = torch.randn(n_nodes, n_features, dtype=torch.float32)

    # Create 3D positions
    pos = torch.randn(n_nodes, 3, dtype=torch.float32)

    # Create random edges
    edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)

    # Create labels for classification
    y = torch.randint(0, 3, (n_nodes,), dtype=torch.long)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, pos=pos, y=y)

    # Add metadata
    data.survey = "test_survey"
    data.feature_names = [f"feat_{i}" for i in range(n_features)]
    data.num_objects = n_nodes

    return data


@pytest.fixture
def sample_survey_data():
    """Create sample survey data as Polars DataFrame."""
    n_objects = 100

    return pl.DataFrame(
        {
            "source_id": range(n_objects),
            "ra": torch.rand(n_objects).numpy() * 360,
            "dec": (torch.rand(n_objects).numpy() - 0.5) * 180,
            "parallax": torch.rand(n_objects).numpy() * 10 + 0.1,
            "parallax_error": torch.rand(n_objects).numpy() * 0.1,
            "pmra": torch.randn(n_objects).numpy() * 5,
            "pmdec": torch.randn(n_objects).numpy() * 5,
            "phot_g_mean_mag": torch.rand(n_objects).numpy() * 10 + 10,
            "phot_bp_mean_mag": torch.rand(n_objects).numpy() * 10 + 10.5,
            "phot_rp_mean_mag": torch.rand(n_objects).numpy() * 10 + 9.5,
            "ruwe": torch.rand(n_objects).numpy() * 0.5 + 0.8,
        }
    )


@pytest.fixture
def preprocessor(small_survey):
    """Get preprocessor for the default survey."""
    return get_preprocessor(small_survey)


@pytest.fixture
def lightning_model():
    """Create a simple Lightning model for testing."""
    # Create a simple AstroModel for testing
    model = AstroModel(
        num_features=10,
        num_classes=3,
        hidden_dim=32,
        num_layers=2,
        task="node_classification",
    )
    return model


@pytest.fixture(scope="session")
def available_surveys():
    """Check which surveys have data available."""
    surveys = []
    for survey in ["gaia", "nsa", "exoplanet", "sdss", "twomass"]:
        # Check if processed data exists using central config
        processed_path = (
            Path(data_config["processed_dir"]) / survey / f"{survey}.parquet"
        )
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
