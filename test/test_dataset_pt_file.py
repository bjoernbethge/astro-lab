"""Test that the dataset creates a .pt file for CI validation."""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
import torch

from astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset


def test_dataset_creates_pt_file():
    """Test that process() creates a .pt file as required by CI validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a mock parquet file
        survey_dir = tmpdir / "gaia"
        survey_dir.mkdir()

        # Create minimal parquet data
        df = pl.DataFrame(
            {
                "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
                "dec": [5.0, 15.0, 25.0, 35.0, 45.0],
                "parallax": [1.0, 2.0, 3.0, 4.0, 5.0],
                "bp_rp": [0.5, 1.0, 1.5, 2.0, 2.5],
                "mg_abs": [2.0, 3.0, 4.0, 5.0, 6.0],
                "distance_pc": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        parquet_path = survey_dir / "gaia.parquet"
        df.write_parquet(parquet_path)

        # Create dataset
        dataset = AstroLabInMemoryDataset(
            root=str(survey_dir),
            survey_name="gaia",
            task="node_classification",
            sampling_strategy="knn",
            force_reload=True,
        )

        # Check that the .pt file was created
        pt_file = Path(dataset.processed_paths[0])
        assert pt_file.exists(), f"Expected .pt file at {pt_file} but it doesn't exist"

        # Check that it can be loaded
        data, slices = torch.load(pt_file, weights_only=False)
        assert data is not None, "Data should not be None"
        assert slices is not None, "Slices should not be None"

        # Verify it's a valid PyG Data object
        assert hasattr(data, "x"), "Data should have x attribute"
        assert hasattr(data, "edge_index"), "Data should have edge_index attribute"

        # Check metadata file
        metadata_file = Path(dataset.processed_paths[1])
        assert metadata_file.exists(), f"Expected metadata file at {metadata_file}"

        with open(metadata_file) as f:
            metadata = json.load(f)
        assert metadata["streaming"] is True, "Should be marked as streaming"
        assert metadata["survey_name"] == "gaia"
        assert metadata["task"] == "node_classification"


def test_dataset_streaming_works():
    """Test that the streaming dataset can still load data on-demand."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a mock parquet file with more data
        survey_dir = tmpdir / "gaia"
        survey_dir.mkdir()

        # Create data for multiple graphs
        n_points = 100
        df = pl.DataFrame(
            {
                "ra": [float(i) for i in range(n_points)],
                "dec": [float(i * 2) for i in range(n_points)],
                "parallax": [1.0 + i * 0.1 for i in range(n_points)],
                "bp_rp": [0.5 + i * 0.01 for i in range(n_points)],
                "mg_abs": [2.0 + i * 0.05 for i in range(n_points)],
                "distance_pc": [100.0 + i * 10 for i in range(n_points)],
            }
        )
        parquet_path = survey_dir / "gaia.parquet"
        df.write_parquet(parquet_path)

        # Create dataset
        dataset = AstroLabInMemoryDataset(
            root=str(survey_dir),
            survey_name="gaia",
            task="node_classification",
            sampling_strategy="knn",
            force_reload=True,
        )

        # Verify that we can get data from the streaming dataset
        assert len(dataset) > 0, "Dataset should have at least one sample"

        # Get first graph
        graph = dataset.get(0)
        assert graph is not None, "Should be able to get graph from streaming dataset"
        assert hasattr(graph, "x"), "Graph should have features"
        assert hasattr(graph, "edge_index"), "Graph should have edges"
        assert graph.x.shape[0] > 0, "Graph should have nodes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
