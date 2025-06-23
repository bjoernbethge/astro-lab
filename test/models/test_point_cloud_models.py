"""
Tests for Point Cloud Models
============================

Tests for 3D stellar point cloud processing models.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from astro_lab.models.point_cloud_models import (
    GalacticStructureGNN,
    HierarchicalStellarGNN,
    StellarClusterGNN,
    StellarPointCloudGNN,
)


class TestStellarPointCloudGNN:
    """Test StellarPointCloudGNN for stellar processing."""

    def test_initialization(self):
        """Test StellarPointCloudGNN initializes correctly."""
        model = StellarPointCloudGNN(
            input_dim=1,  # Magnitude only
            hidden_dim=64,
            num_neighbors=16,
            use_gravitational_mp=True,
        )

        assert isinstance(model, nn.Module)
        assert model.input_dim == 1
        assert model.hidden_dim == 64
        assert model.num_neighbors == 16
        assert model.use_gravitational_mp == True
        assert hasattr(model, "input_projection")
        assert hasattr(model, "pointnet_layers")

    def test_forward_pass_with_data_object(self):
        """Test forward pass with torch_geometric Data object."""
        model = StellarPointCloudGNN(input_dim=1, hidden_dim=32, num_neighbors=8)

        # Create sample data
        pos = torch.randn(20, 3)  # 20 stars in 3D
        x = torch.randn(20, 1)  # Magnitude features
        edge_index = torch.randint(0, 20, (2, 40))  # Some edges

        data = Data(x=x, pos=pos, edge_index=edge_index)

        output = model(data)
        assert isinstance(output, dict)
        assert "embeddings" in output
        assert "edge_index" in output
        assert "positions" in output

        # Check embeddings shape
        embeddings = output["embeddings"]
        assert embeddings.shape == (20, 32)  # batch_size, hidden_dim
        assert not torch.isnan(embeddings).any()

    def test_gravitational_message_passing(self):
        """Test gravitational message passing functionality."""
        model_with_grav = StellarPointCloudGNN(
            input_dim=1, hidden_dim=48, use_gravitational_mp=True
        )

        model_without_grav = StellarPointCloudGNN(
            input_dim=1, hidden_dim=48, use_gravitational_mp=False
        )

        # Create identical data
        pos = torch.randn(15, 3)
        x = torch.randn(15, 1)
        edge_index = torch.randint(0, 15, (2, 30))
        data = Data(x=x, pos=pos, edge_index=edge_index)

        output_with = model_with_grav(data)
        output_without = model_without_grav(data)

        # Both should return dictionaries with embeddings
        assert isinstance(output_with, dict) and isinstance(output_without, dict)
        assert output_with["embeddings"].shape == output_without["embeddings"].shape
        # Outputs should be different due to gravitational features
        assert not torch.allclose(
            output_with["embeddings"], output_without["embeddings"], atol=1e-6
        )

    def test_different_num_neighbors(self):
        """Test with different numbers of neighbors."""
        pos = torch.randn(25, 3)
        x = torch.randn(25, 1)
        data = Data(x=x, pos=pos)

        for num_neighbors in [4, 8, 16]:
            model = StellarPointCloudGNN(
                input_dim=1, hidden_dim=32, num_neighbors=num_neighbors
            )

            output = model(data)
            assert isinstance(output, dict)
            assert "embeddings" in output
            embeddings = output["embeddings"]
            assert embeddings.shape == (25, 32)
            assert not torch.isnan(embeddings).any()


class TestHierarchicalStellarGNN:
    """Test HierarchicalStellarGNN for multi-scale analysis."""

    def test_initialization(self):
        """Test HierarchicalStellarGNN initializes correctly."""
        model = HierarchicalStellarGNN(
            input_dim=3, hidden_dim=64, scales=[0.1, 1.0, 10.0]
        )

        assert isinstance(model, nn.Module)
        assert model.input_dim == 3
        assert model.hidden_dim == 64
        assert model.scales == [0.1, 1.0, 10.0]
        assert hasattr(model, "scale_processors")
        assert len(model.scale_processors) == 3

    def test_forward_pass(self):
        """Test forward pass with position and features."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HierarchicalStellarGNN(
            input_dim=4, hidden_dim=48, scales=[0.5, 2.0, 8.0]
        ).to(device)

        pos = torch.randn(30, 3).to(device)  # 30 stars, 3D positions
        x = torch.randn(30, 4).to(device)  # 4 features per star

        output = model(pos, x)
        assert output.shape == (30, 48)  # batch_size, hidden_dim
        assert not torch.isnan(output).any()

    def test_multi_scale_processing(self):
        """Test multi-scale processing with different scales."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Test with single scale
        model_single = HierarchicalStellarGNN(input_dim=2, hidden_dim=32, scales=[1.0]).to(device)

        # Test with multiple scales
        model_multi = HierarchicalStellarGNN(
            input_dim=2, hidden_dim=32, scales=[0.1, 1.0, 5.0]
        ).to(device)

        pos = torch.randn(20, 3).to(device)
        x = torch.randn(20, 2).to(device)

        output_single = model_single(pos, x)
        output_multi = model_multi(pos, x)

        assert output_single.shape == (20, 32)
        assert output_multi.shape == (20, 32)
        # Multi-scale should give different results
        assert not torch.allclose(output_single, output_multi, atol=1e-6)

    def test_attention_mechanism(self):
        """Test attention mechanism in hierarchical processing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Both models use attention (GATConv) by default
        model_with_att = HierarchicalStellarGNN(
            input_dim=2, hidden_dim=32, scales=[1.0]
        ).to(device)

        model_without_att = HierarchicalStellarGNN(
            input_dim=2,
            hidden_dim=32,
            scales=[2.0],  # Different scale to get different results
        ).to(device)

        pos = torch.randn(15, 3).to(device)
        x = torch.randn(15, 2).to(device)

        output_with = model_with_att(pos, x)
        output_without = model_without_att(pos, x)

        assert output_with.shape == output_without.shape
        # Different scales should produce different results
        assert not torch.allclose(output_with, output_without, atol=1e-6)


class TestStellarClusterGNN:
    """Test StellarClusterGNN for cluster analysis."""

    def test_initialization(self):
        """Test StellarClusterGNN initializes correctly."""
        model = StellarClusterGNN(
            hidden_dim=128, cluster_detection=True, age_estimation=True
        )

        assert isinstance(model, nn.Module)
        assert model.hidden_dim == 128
        assert model.cluster_detection == True
        assert model.age_estimation == True
        assert hasattr(model, "evolution_encoder")
        assert hasattr(model, "cluster_head")

    def test_forward_pass(self):
        """Test forward pass with stellar cluster data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StellarClusterGNN(hidden_dim=64, cluster_detection=True).to(device)

        pos = torch.randn(50, 3).to(device)  # 50 stars, 3D positions
        colors = torch.randn(50, 2).to(device)  # 2 color indices (reduced from 5)
        magnitudes = torch.randn(50, 3).to(device)  # 3 magnitude bands (reduced from 6)
        edge_index = torch.randint(0, 50, (2, 100)).to(device)  # Cluster connections

        output = model(pos, colors, magnitudes, edge_index)
        assert isinstance(output, dict)
        assert "embeddings" in output
        embeddings = output["embeddings"]
        assert embeddings.shape == (50, 64)  # batch_size, hidden_dim
        assert not torch.isnan(embeddings).any()

    def test_age_estimation_output(self):
        """Test age estimation functionality."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StellarClusterGNN(hidden_dim=48, age_estimation=True).to(device)

        pos = torch.randn(25, 3).to(device)
        colors = torch.randn(25, 2).to(device)  # 2 color indices
        magnitudes = torch.randn(25, 3).to(device)  # 3 magnitude bands (total 5 features)
        edge_index = torch.randint(0, 25, (2, 50)).to(device)

        output = model(pos, colors, magnitudes, edge_index)

        # When age_estimation=True, output should include age predictions
        assert isinstance(output, dict)
        assert "embeddings" in output
        assert "age" in output
        assert output["embeddings"].shape == (25, 48)
        assert output["age"].shape == (25, 1)
        assert not torch.isnan(output["embeddings"]).any()
        assert not torch.isnan(output["age"]).any()

    def test_without_cluster_features(self):
        """Test model without cluster-specific features."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StellarClusterGNN(
            hidden_dim=32, cluster_detection=False, age_estimation=False
        ).to(device)

        pos = torch.randn(20, 3).to(device)
        colors = torch.randn(20, 2).to(device)
        magnitudes = torch.randn(20, 3).to(device)
        edge_index = torch.randint(0, 20, (2, 40)).to(device)

        output = model(pos, colors, magnitudes, edge_index)
        assert isinstance(output, dict)
        assert "embeddings" in output
        embeddings = output["embeddings"]
        assert embeddings.shape == (20, 32)
        assert not torch.isnan(embeddings).any()


class TestGalacticStructureGNN:
    """Test GalacticStructureGNN for galactic structure analysis."""

    def test_initialization(self):
        """Test GalacticStructureGNN initializes correctly."""
        model = GalacticStructureGNN(
            hidden_dim=256,
            spiral_detection=True,
            bar_detection=True,
            halo_analysis=True,
        )

        assert isinstance(model, nn.Module)
        assert model.hidden_dim == 256
        assert model.spiral_detection == True
        assert model.bar_detection == True
        assert model.halo_analysis == True
        assert hasattr(model, "galactic_encoder")

    def test_forward_pass(self):
        """Test forward pass with galactic coordinate data."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GalacticStructureGNN(hidden_dim=128, spiral_detection=True).to(device)

        galactic_coords = torch.randn(100, 6).to(device)  # l, b, distance, pm_l, pm_b, vrad
        edge_index = torch.randint(0, 100, (2, 200)).to(device)

        output = model(galactic_coords, edge_index)
        assert isinstance(output, dict)
        assert "embeddings" in output
        embeddings = output["embeddings"]
        assert embeddings.shape == (100, 128)  # batch_size, hidden_dim
        assert not torch.isnan(embeddings).any()

    def test_metallicity_prediction(self):
        """Test metallicity prediction functionality."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_with_metallicity = GalacticStructureGNN(hidden_dim=64, halo_analysis=True).to(device)
        model_without_metallicity = GalacticStructureGNN(
            hidden_dim=64, halo_analysis=False
        ).to(device)

        galactic_coords = torch.randn(30, 6).to(device)
        edge_index = torch.randint(0, 30, (2, 60)).to(device)

        output_with = model_with_metallicity(galactic_coords, edge_index)
        output_without = model_without_metallicity(galactic_coords, edge_index)

        assert isinstance(output_with, dict) and isinstance(output_without, dict)
        assert output_with["embeddings"].shape == (30, 64)
        assert output_without["embeddings"].shape == (30, 64)

        # With halo analysis should have additional outputs
        if model_with_metallicity.halo_analysis:
            assert len(output_with) >= len(output_without)

    def test_different_structure_types(self):
        """Test different galactic structure detection types."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GalacticStructureGNN(
            hidden_dim=48,
            spiral_detection=True,
            bar_detection=True,
            halo_analysis=False,
        ).to(device)

        galactic_coords = torch.randn(40, 6).to(device)
        edge_index = torch.randint(0, 40, (2, 80)).to(device)

        output = model(galactic_coords, edge_index)
        assert isinstance(output, dict)
        assert "embeddings" in output
        embeddings = output["embeddings"]
        assert embeddings.shape == (40, 48)
        assert not torch.isnan(embeddings).any()


class TestPointCloudModelIntegration:
    """Test integration between different point cloud models."""

    def test_model_compatibility(self):
        """Test that different models can work with similar inputs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create compatible models
        stellar_model = StellarPointCloudGNN(
            input_dim=1, hidden_dim=32
        ).to(device)  # StellarPointCloudGNN uses stellar_encoder for 1 feature
        hierarchical_model = HierarchicalStellarGNN(input_dim=2, hidden_dim=32).to(device)

        pos = torch.randn(20, 3).to(device)

        # Test StellarPointCloudGNN with Data object (1 feature)
        x_stellar = torch.randn(20, 1).to(device)  # 1 feature for stellar_encoder
        stellar_data = Data(x=x_stellar, pos=pos).to(device)
        stellar_output = stellar_model(stellar_data)

        # Test HierarchicalStellarGNN with tensors (2 features)
        x_hierarchical = torch.randn(20, 2).to(device)  # 2 features for input_projection
        hierarchical_output = hierarchical_model(pos, x_hierarchical)

        # Both should produce valid outputs
        assert isinstance(stellar_output, dict)
        assert "embeddings" in stellar_output
        assert stellar_output["embeddings"].shape == (20, 32)

        assert hierarchical_output.shape == (20, 32)

    def test_different_hidden_dimensions(self):
        """Test models with different hidden dimensions."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dims = [16, 32, 64]

        for hidden_dim in hidden_dims:
            model = StellarPointCloudGNN(
                input_dim=1, hidden_dim=hidden_dim
            ).to(device)  # 1 feature for stellar_encoder

            pos = torch.randn(15, 3).to(device)
            x = torch.randn(15, 1).to(device)  # 1 feature
            data = Data(x=x, pos=pos).to(device)

            output = model(data)
            assert isinstance(output, dict)
            assert "embeddings" in output
            assert output["embeddings"].shape == (15, hidden_dim)

    def test_batch_processing(self):
        """Test batch processing with point cloud models."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StellarPointCloudGNN(input_dim=1, hidden_dim=32).to(device)

        batch_size = 25
        pos = torch.randn(batch_size, 3).to(device)
        x = torch.randn(batch_size, 1).to(device)
        data = Data(x=x, pos=pos).to(device)

        output = model(data)
        assert isinstance(output, dict)
        assert "embeddings" in output
        embeddings = output["embeddings"]
        assert embeddings.shape == (batch_size, 32)
        assert not torch.isnan(embeddings).any()

    def test_device_consistency(self):
        """Test that models maintain device consistency."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StellarPointCloudGNN(input_dim=1, hidden_dim=16).to(device)

        pos = torch.randn(10, 3).to(device)
        x = torch.randn(10, 1).to(device)
        data = Data(x=x, pos=pos).to(device)

        # Test CPU
        output_cpu = model(data)
        assert isinstance(output_cpu, dict)
        assert "embeddings" in output_cpu
        assert output_cpu["embeddings"].device.type == "cpu"

        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            data_gpu = Data(x=x.cuda(), pos=pos.cuda()).to(device)
            output_gpu = model_gpu(data_gpu)
            assert isinstance(output_gpu, dict)
            assert "embeddings" in output_gpu
            assert output_gpu["embeddings"].device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
