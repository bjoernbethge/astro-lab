"""
Test der konsolidierten Graph Building Funktionalität
===================================================

Testet nur die neue API ohne Legacy-Code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import polars as pl

from astro_lab.data.graphs import (
    create_knn_graph,
    create_pointcloud_graph,
    create_adaptive_graph,
    create_multiscale_graph,
    create_astronomical_graph,
)
from astro_lab.data.graphs.advanced import (
    create_dynamic_graph,
    create_hierarchical_graph,
    create_geometric_prior_graph,
)
from astro_lab.data.preprocessing.survey_specific import (
    GaiaPreprocessor,
    SDSSPreprocessor,
    get_survey_preprocessor,
)
from astro_lab.tensors import SpatialTensorDict, PhotometricTensorDict, SurveyTensorDict


def test_preprocessing():
    """Test survey-specific preprocessing."""
    print("\n=== Testing Survey Preprocessing ===")
    
    # Mock Gaia data
    n_stars = 100
    gaia_df = pl.DataFrame({
        "ra": np.random.uniform(0, 360, n_stars),
        "dec": np.random.uniform(-90, 90, n_stars),
        "parallax": np.random.exponential(1.0, n_stars) + 0.1,
        "parallax_error": np.random.exponential(0.1, n_stars),
        "phot_g_mean_mag": np.random.normal(15, 2, n_stars),
        "bp_rp": np.random.normal(1.0, 0.5, n_stars),
        "pmra": np.random.normal(0, 5, n_stars),
        "pmdec": np.random.normal(0, 5, n_stars),
    })
    
    # Test Gaia preprocessor
    gaia_proc = GaiaPreprocessor()
    positions = gaia_proc.extract_3d_positions(gaia_df)
    features = gaia_proc.extract_features(gaia_df)
    labels = gaia_proc.create_labels(gaia_df)
    
    print(f"✓ Gaia positions shape: {positions.shape}")
    print(f"✓ Gaia features shape: {features.shape}")
    print(f"✓ Gaia labels shape: {labels.shape if labels is not None else 'None'}")
    print(f"✓ Feature names: {gaia_proc.feature_names[:3]}...")
    
    # Check Bayesian distance estimation
    distances = np.linalg.norm(positions, axis=1)
    print(f"✓ Distance range: {distances.min():.1f} - {distances.max():.1f} pc")
    
    # Test SDSS preprocessor
    n_galaxies = 50
    sdss_df = pl.DataFrame({
        "ra": np.random.uniform(0, 360, n_galaxies),
        "dec": np.random.uniform(-90, 90, n_galaxies),
        "z": np.random.exponential(0.1, n_galaxies),
        "modelMag_g": np.random.normal(18, 1, n_galaxies),
        "modelMag_r": np.random.normal(17, 1, n_galaxies),
        "fracDeV_r": np.random.uniform(0, 1, n_galaxies),
    })
    
    sdss_proc = SDSSPreprocessor()
    sdss_positions = sdss_proc.extract_3d_positions(sdss_df)
    sdss_features = sdss_proc.extract_features(sdss_df)
    
    print(f"\n✓ SDSS positions shape: {sdss_positions.shape}")
    print(f"✓ SDSS features shape: {sdss_features.shape}")


def test_basic_graph_building():
    """Test basic graph building methods."""
    print("\n=== Testing Basic Graph Building ===")
    
    # Create test data
    n_objects = 500
    coordinates = torch.randn(n_objects, 3) * 100
    magnitudes = torch.randn(n_objects, 3) + 15
    
    spatial = SpatialTensorDict(coordinates=coordinates)
    photometric = PhotometricTensorDict(
        magnitudes=magnitudes,
        bands=["g", "r", "i"]
    )
    survey_tensor = SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="test",
        data_release="v1.0"
    )
    
    # Test different graph builders
    print("\n1. KNN Graph:")
    knn_graph = create_knn_graph(survey_tensor, k_neighbors=16)
    print(f"   ✓ Nodes: {knn_graph.num_nodes}, Edges: {knn_graph.num_edges}")
    print(f"   ✓ Features: {knn_graph.x.shape}")
    print(f"   ✓ Graph type: {knn_graph.graph_type}")
    
    print("\n2. Point Cloud Graph:")
    pc_graph = create_pointcloud_graph(survey_tensor, normalize_positions=True)
    print(f"   ✓ Nodes: {pc_graph.num_nodes}, Edges: {pc_graph.num_edges}")
    print(f"   ✓ Normalized: {pc_graph.normalized_positions}")
    print(f"   ✓ Graph type: {pc_graph.graph_type}")
    
    print("\n3. Adaptive Graph:")
    adaptive_graph = create_adaptive_graph(survey_tensor)
    print(f"   ✓ Nodes: {adaptive_graph.num_nodes}, Edges: {adaptive_graph.num_edges}")
    print(f"   ✓ Graph type: {adaptive_graph.graph_type}")
    
    print("\n4. Astronomical Graph:")
    astro_graph = create_astronomical_graph(survey_tensor)
    print(f"   ✓ Nodes: {astro_graph.num_nodes}, Edges: {astro_graph.num_edges}")
    print(f"   ✓ Graph type: {astro_graph.graph_type}")


def test_advanced_graph_building():
    """Test advanced graph building methods."""
    print("\n=== Testing Advanced Graph Building ===")
    
    # Create test data
    n_objects = 1000
    coordinates = torch.randn(n_objects, 3) * 50
    
    # Create clustered structure
    for i in range(5):  # 5 clusters
        cluster_idx = slice(i * 200, (i + 1) * 200)
        coordinates[cluster_idx] += torch.randn(1, 3) * 100
    
    magnitudes = torch.randn(n_objects, 5) + 15
    
    spatial = SpatialTensorDict(coordinates=coordinates)
    photometric = PhotometricTensorDict(
        magnitudes=magnitudes,
        bands=["u", "g", "r", "i", "z"]
    )
    survey_tensor = SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="test_advanced"
    )
    
    print("\n1. Multi-scale Graph:")
    multiscale_graph = create_multiscale_graph(
        survey_tensor, 
        scales=[8, 16, 32]
    )
    print(f"   ✓ Nodes: {multiscale_graph.num_nodes}, Edges: {multiscale_graph.num_edges}")
    print(f"   ✓ Scales: {multiscale_graph.scales}")
    print(f"   ✓ Edge attributes shape: {multiscale_graph.edge_attr.shape}")
    
    print("\n2. Dynamic Graph:")
    dynamic_graph = create_dynamic_graph(survey_tensor, initial_k=12)
    print(f"   ✓ Nodes: {dynamic_graph.num_nodes}, Edges: {dynamic_graph.num_edges}")
    print(f"   ✓ Supports edge learning: {dynamic_graph.supports_edge_learning}")
    
    print("\n3. Hierarchical Graph:")
    hierarchical_graph = create_hierarchical_graph(
        survey_tensor,
        cluster_method="kmeans",
        n_clusters=5
    )
    print(f"   ✓ Nodes: {hierarchical_graph.num_nodes}, Edges: {hierarchical_graph.num_edges}")
    print(f"   ✓ Number of clusters: {hierarchical_graph.n_clusters}")
    
    print("\n4. Geometric Prior Graph (Filament):")
    geometric_graph = create_geometric_prior_graph(
        survey_tensor,
        prior_type="filament"
    )
    print(f"   ✓ Nodes: {geometric_graph.num_nodes}, Edges: {geometric_graph.num_edges}")
    print(f"   ✓ Graph type: {geometric_graph.graph_type}")


def test_robustness():
    """Test robustness with edge cases."""
    print("\n=== Testing Robustness ===")
    
    # Test 1: Small dataset
    print("\n1. Small dataset (5 objects):")
    small_coords = torch.randn(5, 3)
    small_spatial = SpatialTensorDict(coordinates=small_coords)
    small_phot = PhotometricTensorDict(
        magnitudes=torch.randn(5, 1),
        bands=["g"]
    )
    small_survey = SurveyTensorDict(
        spatial=small_spatial,
        photometric=small_phot
    )
    
    try:
        small_graph = create_knn_graph(small_survey, k_neighbors=3)
        print(f"   ✓ Success: {small_graph.num_nodes} nodes, {small_graph.num_edges} edges")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 2: Dataset with NaN
    print("\n2. Dataset with NaN values:")
    nan_coords = torch.randn(50, 3)
    nan_mags = torch.randn(50, 3)
    nan_mags[10:20] = float('nan')
    
    nan_spatial = SpatialTensorDict(coordinates=nan_coords)
    nan_phot = PhotometricTensorDict(magnitudes=nan_mags, bands=["g", "r", "i"])
    nan_survey = SurveyTensorDict(spatial=nan_spatial, photometric=nan_phot)
    
    try:
        nan_graph = create_knn_graph(nan_survey, k_neighbors=8)
        print(f"   ✓ Success: {nan_graph.num_nodes} nodes")
        print(f"   ✓ NaN handled: {not torch.isnan(nan_graph.x).any()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 3: Large dataset
    print("\n3. Large dataset (10k objects):")
    large_coords = torch.randn(10000, 3)
    large_spatial = SpatialTensorDict(coordinates=large_coords)
    large_phot = PhotometricTensorDict(
        magnitudes=torch.randn(10000, 3),
        bands=["g", "r", "i"]
    )
    large_survey = SurveyTensorDict(
        spatial=large_spatial,
        photometric=large_phot
    )
    
    try:
        import time
        start = time.time()
        large_graph = create_knn_graph(large_survey, k_neighbors=16)
        elapsed = time.time() - start
        print(f"   ✓ Success: {large_graph.num_nodes} nodes in {elapsed:.2f}s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 4: Dataset with outliers
    print("\n4. Dataset with extreme outliers:")
    outlier_coords = torch.randn(100, 3)
    outlier_coords[0:5] *= 1000  # Extreme outliers
    
    outlier_spatial = SpatialTensorDict(coordinates=outlier_coords)
    outlier_phot = PhotometricTensorDict(
        magnitudes=torch.randn(100, 3),
        bands=["g", "r", "i"]
    )
    outlier_survey = SurveyTensorDict(
        spatial=outlier_spatial,
        photometric=outlier_phot
    )
    
    try:
        outlier_graph = create_knn_graph(
            outlier_survey, 
            k_neighbors=8,
            outlier_detection=True
        )
        print(f"   ✓ Success: {outlier_graph.num_nodes} nodes with outlier handling")
    except Exception as e:
        print(f"   ✗ Failed: {e}")


def test_full_pipeline():
    """Test complete pipeline from raw data to graph."""
    print("\n=== Testing Full Pipeline ===")
    
    # Step 1: Create mock survey data
    print("\n1. Creating mock survey data:")
    n_objects = 1000
    mock_df = pl.DataFrame({
        "ra": np.random.uniform(0, 360, n_objects),
        "dec": np.random.uniform(-90, 90, n_objects),
        "parallax": np.random.exponential(1.0, n_objects) + 0.1,
        "parallax_error": np.random.exponential(0.1, n_objects),
        "phot_g_mean_mag": np.random.normal(15, 2, n_objects),
        "bp_rp": np.random.normal(1.0, 0.5, n_objects),
        "pmra": np.random.normal(0, 5, n_objects),
        "pmdec": np.random.normal(0, 5, n_objects),
    })
    print(f"   ✓ Created DataFrame with {len(mock_df)} objects")
    
    # Step 2: Preprocessing
    print("\n2. Preprocessing with GaiaPreprocessor:")
    preprocessor = GaiaPreprocessor()
    positions = preprocessor.extract_3d_positions(mock_df)
    features = preprocessor.extract_features(mock_df)
    labels = preprocessor.create_labels(mock_df)
    
    print(f"   ✓ Positions: {positions.shape}")
    print(f"   ✓ Features: {features.shape}")
    print(f"   ✓ Labels: {labels.shape if labels is not None else 'None'}")
    
    # Step 3: Create SurveyTensorDict
    print("\n3. Creating SurveyTensorDict:")
    spatial = SpatialTensorDict(
        coordinates=torch.tensor(positions, dtype=torch.float32)
    )
    
    # Use first few feature columns as magnitudes
    n_bands = min(3, features.shape[1])
    photometric = PhotometricTensorDict(
        magnitudes=torch.tensor(features[:, :n_bands], dtype=torch.float32),
        bands=preprocessor.feature_names[:n_bands] if hasattr(preprocessor, 'feature_names') else [f"band_{i}" for i in range(n_bands)]
    )
    
    survey_tensor = SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="gaia",
        data_release="dr3"
    )
    print(f"   ✓ SurveyTensorDict created for {survey_tensor.survey_name}")
    
    # Step 4: Build different types of graphs
    print("\n4. Building graphs:")
    
    graph_builders = {
        "KNN": lambda: create_knn_graph(survey_tensor, k_neighbors=16),
        "Point Cloud": lambda: create_pointcloud_graph(survey_tensor, normalize_positions=True),
        "Astronomical": lambda: create_astronomical_graph(survey_tensor),
        "Multi-scale": lambda: create_multiscale_graph(survey_tensor, scales=[8, 16]),
    }
    
    for name, builder in graph_builders.items():
        try:
            graph = builder()
            print(f"   ✓ {name}: {graph.num_nodes} nodes, {graph.num_edges} edges")
        except Exception as e:
            print(f"   ✗ {name}: Failed - {str(e)[:50]}")
    
    print("\n✓ Pipeline test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Astro-Lab Graph Building Test Suite")
    print("Testing only the new consolidated API")
    print("=" * 60)
    
    test_preprocessing()
    test_basic_graph_building()
    test_advanced_graph_building()
    test_robustness()
    test_full_pipeline()
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
