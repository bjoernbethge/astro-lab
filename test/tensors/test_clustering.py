"""
Tests for ClusteringTensor
=========================

Test suite for astronomical clustering operations.
"""

import numpy as np
import pytest
import torch

from astro_lab.tensors import ClusteringTensor

# Skip all tests in this file for now
# pytestmark = pytest.mark.skip(reason="Skipping clustering tests temporarily to focus on other failures.")


class TestClusteringTensor:
    """Test ClusteringTensor functionality."""

    @pytest.fixture
    def sample_2d_data(self):
        """Create sample 3D astronomical data for testing (z=0)."""
        np.random.seed(42)
        n_objects = 100

        # Create clustered data (3 clusters)
        cluster_centers = np.array([[0, 0, 0], [5, 5, 0], [-3, 3, 0]])
        positions = []

        for center in cluster_centers:
            # Add some objects around each center
            n_cluster = n_objects // 3
            cluster_pos = np.random.normal(center, 0.5, (n_cluster, 3))
            positions.append(cluster_pos)

        # Add remaining objects
        remaining = n_objects - len(positions) * (n_objects // 3)
        if remaining > 0:
            extra_pos = np.random.normal(cluster_centers[0], 0.5, (remaining, 3))
            positions.append(extra_pos)

        positions = np.vstack(positions)

        # Add some features
        features = np.random.randn(n_objects, 3)

        return positions, features

    @pytest.fixture
    def sample_sky_data(self):
        """Create sample sky coordinate data."""
        np.random.seed(42)
        n_objects = 50

        # Create RA/Dec positions with clustering
        ra_centers = [150.0, 200.0, 250.0]  # degrees
        dec_centers = [30.0, -10.0, 60.0]  # degrees

        positions = []
        for ra_c, dec_c in zip(ra_centers, dec_centers):
            n_cluster = n_objects // 3
            ra_cluster = np.random.normal(ra_c, 2.0, n_cluster)  # 2 degree spread
            dec_cluster = np.random.normal(dec_c, 2.0, n_cluster)
            cluster_pos = np.column_stack([ra_cluster, dec_cluster])
            positions.append(cluster_pos)

        positions = np.vstack(positions)
        return positions

    @pytest.fixture
    def clustering_tensor_2d(self, sample_2d_data):
        """Create ClusteringTensor instance with 3D data (z=0)."""
        positions, features = sample_2d_data
        # Kombiniere Positionen und Features zu einem Array
        data = np.concatenate([positions, features], axis=1)
        meta = {
            "n_spatial_dims": 3,
            "n_features": 3,
            "coordinate_system": "cartesian",
            "astronomical_context": "general"
        }
        return ClusteringTensor(data=data, meta=meta)

    @pytest.fixture
    def clustering_tensor_sky(self, sample_sky_data):
        """Create ClusteringTensor instance with sky coordinates."""
        positions = sample_sky_data
        meta = {
            "n_spatial_dims": 2,
            "n_features": 0,
            "coordinate_system": "spherical",
            "astronomical_context": "general"
        }
        return ClusteringTensor(data=positions, meta=meta)

    def test_initialization(self, sample_2d_data):
        """Test ClusteringTensor initialization."""
        positions, features = sample_2d_data
        data = np.concatenate([positions, features], axis=1)
        meta = {"n_spatial_dims": 3, "n_features": 3, "coordinate_system": "cartesian"}
        tensor = ClusteringTensor(data=data, meta=meta)
        assert len(tensor) == 100
        assert tensor.get_metadata("n_spatial_dims") == 3
        assert tensor.get_metadata("n_features") == 3
        # Test ohne Features
        tensor_no_feat = ClusteringTensor(data=positions, meta={"n_spatial_dims": 3, "n_features": 0, "coordinate_system": "cartesian"})
        assert len(tensor_no_feat) == 100
        assert tensor_no_feat.get_metadata("n_features") == 0
        assert tensor_no_feat.features is None

    def test_position_feature_access(self, clustering_tensor_2d):
        """Test position and feature access."""
        positions = clustering_tensor_2d.positions
        features = clustering_tensor_2d.features
        assert positions.shape == (100, 3)
        assert features.shape == (100, 3)
        assert torch.is_tensor(positions)
        assert torch.is_tensor(features)

    def test_dbscan_clustering(self, clustering_tensor_2d):
        """Test DBSCAN clustering (zentralisierte Logik)."""
        labels = clustering_tensor_2d.dbscan_clustering(eps=1.0, min_samples=5)
        assert isinstance(labels, torch.Tensor)
        assert len(labels) == len(clustering_tensor_2d)
        # Prüfe, dass Cluster gefunden wurden
        unique_labels = torch.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        assert n_clusters > 0
        # Prüfe, dass Statistiken in den Metadaten sind
        stats = clustering_tensor_2d.get_metadata("cluster_stats")
        assert isinstance(stats, dict)
        assert len(stats) == n_clusters

    def test_dbscan_clustering_gpu(self, clustering_tensor_2d):
        """Test DBSCAN clustering mit use_gpu=True (zentralisierte Logik)."""
        from astro_lab.utils.viz.graph import cluster_and_analyze
        coords = clustering_tensor_2d.positions
        result = cluster_and_analyze(coords, algorithm="dbscan", eps=1.0, min_samples=5, use_gpu=True)
        labels = result["cluster_labels"]
        assert isinstance(labels, torch.Tensor)
        assert len(labels) == len(clustering_tensor_2d)
        stats = result["cluster_stats"]
        assert isinstance(stats, dict)
        assert len(stats) == result["n_clusters"]

    def test_friends_of_friends(self, clustering_tensor_2d):
        """Test Friends-of-Friends clustering (zentralisierte Logik)."""
        labels = clustering_tensor_2d.friends_of_friends(linking_length=1.5)
        assert isinstance(labels, torch.Tensor)
        assert len(labels) == len(clustering_tensor_2d)
        unique_labels = torch.unique(labels)
        n_groups = len(unique_labels[unique_labels >= 0])
        assert n_groups >= 0
        stats = clustering_tensor_2d.get_metadata("cluster_stats")
        assert isinstance(stats, dict)

    def test_hierarchical_clustering(self, clustering_tensor_2d):
        """Test hierarchical clustering (zentralisierte Logik)."""
        labels = clustering_tensor_2d.hierarchical_clustering(n_clusters=3)
        assert isinstance(labels, torch.Tensor)
        assert len(labels) == len(clustering_tensor_2d)
        unique_labels = torch.unique(labels)
        assert len(unique_labels) == 3
        stats = clustering_tensor_2d.get_metadata("cluster_stats")
        assert isinstance(stats, dict)
        assert len(stats) == 3

    def test_tensor_metadata(self, clustering_tensor_2d):
        """Test tensor metadata access."""
        assert clustering_tensor_2d.get_metadata("n_spatial_dims") == 3
        assert clustering_tensor_2d.get_metadata("n_features") == 3
        assert clustering_tensor_2d.get_metadata("coordinate_system") == "cartesian"
        assert clustering_tensor_2d.get_metadata("astronomical_context") == "general"

    def test_repr(self, clustering_tensor_2d):
        """Test __repr__ output."""
        repr_str = repr(clustering_tensor_2d)
        assert "coord_sys='cartesian'" in repr_str


class TestClusteringTensorIntegration:
    """Test ClusteringTensor integration with other components."""

    def test_3d_clustering(self):
        """Test clustering with 3D positions."""
        # Create 3D clustered data
        np.random.seed(42)
        n_objects = 60

        # Create 3 clusters in 3D
        cluster_centers = np.array([[0, 0, 0], [5, 5, 5], [-3, 3, -2]])
        positions = []

        for center in cluster_centers:
            n_cluster = n_objects // 3
            cluster_pos = np.random.normal(center, 0.8, (n_cluster, 3))
            positions.append(cluster_pos)

        positions = np.vstack(positions)

        tensor = ClusteringTensor(positions, coordinate_system="cartesian")
        assert tensor.get_metadata("n_spatial_dims") == 3

        try:
            labels = tensor.dbscan_clustering(eps=1.5, min_samples=4)
            assert len(labels) == n_objects

        except ImportError:
            pytest.skip("sklearn not available for 3D clustering")

    def test_large_scale_structure(self):
        """Test clustering for large-scale structure analysis."""
        # Create cosmological-like data
        np.random.seed(42)
        n_objects = 200

        # Create filamentary structure
        positions = []

        # Main filament
        t = np.linspace(0, 10, n_objects // 2)
        filament = np.column_stack([t, np.sin(t), np.cos(t)])
        positions.append(filament)

        # Secondary structure
        t2 = np.linspace(0, 8, n_objects // 2)
        structure2 = np.column_stack([t2 + 2, t2 * 0.5, -t2 * 0.3])
        positions.append(structure2)

        positions = np.vstack(positions)

        tensor = ClusteringTensor(data=positions, meta={"n_spatial_dims": 3, "n_features": 0, "coordinate_system": "cartesian", "astronomical_context": "lss"})
        assert tensor.get_metadata("astronomical_context") == "lss"

        try:
            # Use Friends-of-Friends for LSS
            labels = tensor.friends_of_friends(linking_length=2.0)
            assert len(labels) == n_objects

        except ImportError:
            pytest.skip("sklearn not available for LSS clustering")


@pytest.mark.parametrize("coordinate_system", ["cartesian", "spherical", "sky"])
def test_coordinate_systems(coordinate_system):
    """Test different coordinate systems."""
    if coordinate_system == "spherical":
        # RA, Dec data
        positions = np.random.uniform([0, -90], [360, 90], (50, 2))
    else:
        # Cartesian data
        positions = np.random.randn(50, 2)

    tensor = ClusteringTensor(positions, coordinate_system=coordinate_system)
    assert tensor.get_metadata("coordinate_system") == coordinate_system


@pytest.mark.parametrize(
    "astronomical_context", ["galaxies", "stars", "lss", "general"]
)
def test_astronomical_contexts(astronomical_context):
    """Test different astronomical contexts."""
    positions = np.random.randn(30, 2)
    tensor = ClusteringTensor(positions, astronomical_context=astronomical_context)
    assert tensor.get_metadata("astronomical_context") == astronomical_context
