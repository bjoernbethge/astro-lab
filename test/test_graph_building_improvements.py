"""
Test Script für verbesserte Graph Building Funktionalität
========================================================

Demonstriert die neuen State-of-the-Art Features.
"""

import logging
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astro_lab.data.graphs import (
    GraphConfig,
    create_adaptive_graph,
    create_astronomical_graph,
    create_knn_graph,
    create_multiscale_graph,
    create_pointcloud_graph,
)
from astro_lab.data.graphs.advanced import (
    create_dynamic_graph,
    create_geometric_prior_graph,
    create_hierarchical_graph,
)
from astro_lab.data.preprocessing.survey_specific import (
    GaiaPreprocessor,
    SDSSPreprocessor,
    get_survey_preprocessor,
)
from astro_lab.tensors import PhotometricTensorDict, SpatialTensorDict, SurveyTensorDict

# Setup
console = Console()
logging.basicConfig(level=logging.INFO)


def create_test_survey_data(n_objects: int = 1000) -> SurveyTensorDict:
    """Create test survey data with realistic properties."""
    # Spatial data - simulate galaxy positions
    # Create clustered distribution
    n_clusters = 5
    cluster_centers = torch.randn(n_clusters, 3) * 50
    
    coords = []
    for i in range(n_objects):
        cluster_id = i % n_clusters
        center = cluster_centers[cluster_id]
        # Add noise around cluster center
        noise = torch.randn(3) * 10
        coords.append(center + noise)
    
    coordinates = torch.stack(coords)
    
    # Add some outliers
    n_outliers = int(n_objects * 0.05)
    outlier_indices = torch.randperm(n_objects)[:n_outliers]
    coordinates[outlier_indices] = torch.randn(n_outliers, 3) * 100
    
    spatial = SpatialTensorDict(coordinates=coordinates)
    
    # Photometric data - simulate magnitudes
    # Correlate brightness with distance
    distances = torch.norm(coordinates, dim=1)
    base_magnitude = 15 + 2.5 * torch.log10(distances / 10)
    
    # Add noise and multiple bands
    magnitudes = torch.stack([
        base_magnitude + torch.randn(n_objects) * 0.1,  # g band
        base_magnitude + 0.5 + torch.randn(n_objects) * 0.1,  # r band
        base_magnitude + 1.0 + torch.randn(n_objects) * 0.1,  # i band
    ], dim=1)
    
    photometric = PhotometricTensorDict(
        magnitudes=magnitudes,
        bands=["g", "r", "i"],
        filter_system="AB"
    )
    
    # Create SurveyTensorDict
    survey_tensor = SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="test_survey",
        data_release="v1.0"
    )
    
    return survey_tensor


def test_basic_graph_builders():
    """Test basic graph building methods."""
    console.print("\n[bold blue]Testing Basic Graph Builders[/bold blue]")
    
    # Create test data
    survey_data = create_test_survey_data(n_objects=500)
    
    # Test different builders
    builders = {
        "KNN (k=8)": lambda: create_knn_graph(survey_data, k_neighbors=8),
        "KNN (k=16)": lambda: create_knn_graph(survey_data, k_neighbors=16),
        "Astronomical": lambda: create_astronomical_graph(survey_data),
        "Point Cloud": lambda: create_pointcloud_graph(survey_data),
    }
    
    results = Table(title="Basic Graph Building Results")
    results.add_column("Method", style="cyan")
    results.add_column("Nodes", style="green")
    results.add_column("Edges", style="yellow")
    results.add_column("Features", style="magenta")
    results.add_column("Build Time", style="red")
    
    for name, builder_func in builders.items():
        import time
        start = time.time()
        
        try:
            graph = builder_func()
            build_time = time.time() - start
            
            results.add_row(
                name,
                str(graph.num_nodes),
                str(graph.num_edges),
                str(graph.x.shape[1]),
                f"{build_time:.3f}s"
            )
            
            # Validate graph
            assert graph.edge_index.shape[0] == 2
            assert graph.edge_index.max() < graph.num_nodes
            assert not torch.isnan(graph.x).any()
            
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
    
    console.print(results)


def test_advanced_graph_builders():
    """Test advanced graph building methods."""
    console.print("\n[bold blue]Testing Advanced Graph Builders[/bold blue]")
    
    # Create test data
    survey_data = create_test_survey_data(n_objects=1000)
    
    # Test configuration with outlier detection
    config = GraphConfig(
        normalize_features=True,
        outlier_detection=True,
        outlier_threshold=2.5,
        handle_nan="median"
    )
    
    # Test different advanced builders
    builders = {
        "Multi-scale": lambda: create_multiscale_graph(
            survey_data, 
            scales=[8, 16, 32],
            **config.__dict__
        ),
        "Adaptive": lambda: create_adaptive_graph(
            survey_data,
            **config.__dict__
        ),
        "Dynamic": lambda: create_dynamic_graph(
            survey_data,
            initial_k=12,
            **config.__dict__
        ),
        "Hierarchical": lambda: create_hierarchical_graph(
            survey_data,
            cluster_method="kmeans",
            n_clusters=10,
            **config.__dict__
        ),
        "Geometric (Filament)": lambda: create_geometric_prior_graph(
            survey_data,
            prior_type="filament",
            **config.__dict__
        ),
    }
    
    results = Table(title="Advanced Graph Building Results")
    results.add_column("Method", style="cyan")
    results.add_column("Nodes", style="green")
    results.add_column("Edges", style="yellow")
    results.add_column("Special Properties", style="magenta")
    results.add_column("Build Time", style="red")
    
    for name, builder_func in builders.items():
        import time
        start = time.time()
        
        try:
            graph = builder_func()
            build_time = time.time() - start
            
            # Get special properties
            special_props = []
            if hasattr(graph, "scales"):
                special_props.append(f"Scales: {graph.scales}")
            if hasattr(graph, "n_clusters"):
                special_props.append(f"Clusters: {graph.n_clusters}")
            if hasattr(graph, "supports_edge_learning"):
                special_props.append("Dynamic edges")
            if hasattr(graph, "edge_type"):
                n_edge_types = len(torch.unique(graph.edge_type))
                special_props.append(f"Edge types: {n_edge_types}")
            
            results.add_row(
                name,
                str(graph.num_nodes),
                str(graph.num_edges),
                ", ".join(special_props) or "None",
                f"{build_time:.3f}s"
            )
            
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    console.print(results)


def test_robustness():
    """Test robustness with edge cases."""
    console.print("\n[bold blue]Testing Robustness[/bold blue]")
    
    test_cases = {
        "Small dataset (10 objects)": create_test_survey_data(10),
        "Large dataset (10k objects)": create_test_survey_data(10000),
        "Dataset with NaN values": create_test_survey_data(100),
        "Dataset with outliers": create_test_survey_data(100),
    }
    
    # Add NaN values to one dataset
    nan_data = test_cases["Dataset with NaN values"]
    nan_data["photometric"].magnitudes[10:20, :] = float('nan')
    
    # Add extreme outliers to another
    outlier_data = test_cases["Dataset with outliers"]
    outlier_data["spatial"].coordinates[5:10] *= 1000
    
    results = Table(title="Robustness Test Results")
    results.add_column("Test Case", style="cyan")
    results.add_column("KNN Graph", style="green")
    results.add_column("Adaptive Graph", style="yellow")
    results.add_column("Issues", style="red")
    
    for test_name, test_data in test_cases.items():
        issues = []
        
        # Test KNN
        try:
            knn_graph = create_knn_graph(test_data, k_neighbors=8)
            knn_status = "✓ Success"
        except Exception as e:
            knn_status = "✗ Failed"
            issues.append(f"KNN: {str(e)[:30]}")
        
        # Test Adaptive
        try:
            adaptive_graph = create_adaptive_graph(test_data)
            adaptive_status = "✓ Success"
        except Exception as e:
            adaptive_status = "✗ Failed"
            issues.append(f"Adaptive: {str(e)[:30]}")
        
        results.add_row(
            test_name,
            knn_status,
            adaptive_status,
            "\n".join(issues) or "None"
        )
    
    console.print(results)


def test_preprocessing_improvements():
    """Test improved preprocessing capabilities."""
    console.print("\n[bold blue]Testing Preprocessing Improvements[/bold blue]")
    
    # Create mock survey data
    import polars as pl
    import numpy as np
    
    # Mock Gaia data
    n_stars = 1000
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
    
    # Mock SDSS data
    n_galaxies = 500
    sdss_df = pl.DataFrame({
        "ra": np.random.uniform(0, 360, n_galaxies),
        "dec": np.random.uniform(-90, 90, n_galaxies),
        "z": np.random.exponential(0.1, n_galaxies),
        "modelMag_g": np.random.normal(18, 1, n_galaxies),
        "modelMag_r": np.random.normal(17, 1, n_galaxies),
        "fracDeV_r": np.random.uniform(0, 1, n_galaxies),
        "petroR50_r": np.random.exponential(1.0, n_galaxies),
        "petroR90_r": np.random.exponential(2.0, n_galaxies),
    })
    
    results = Table(title="Preprocessing Test Results")
    results.add_column("Survey", style="cyan")
    results.add_column("Preprocessor", style="green")
    results.add_column("3D Positions", style="yellow")
    results.add_column("Features", style="magenta")
    results.add_column("Labels", style="red")
    
    # Test Gaia preprocessor
    gaia_proc = GaiaPreprocessor()
    gaia_pos = gaia_proc.extract_3d_positions(gaia_df)
    gaia_feat = gaia_proc.extract_features(gaia_df)
    gaia_labels = gaia_proc.create_labels(gaia_df)
    
    results.add_row(
        "Gaia",
        "✓ GaiaPreprocessor",
        f"✓ {gaia_pos.shape}",
        f"✓ {gaia_feat.shape}",
        f"✓ {gaia_labels.shape if gaia_labels is not None else 'None'}"
    )
    
    # Test SDSS preprocessor
    sdss_proc = SDSSPreprocessor()
    sdss_pos = sdss_proc.extract_3d_positions(sdss_df)
    sdss_feat = sdss_proc.extract_features(sdss_df)
    sdss_labels = sdss_proc.create_labels(sdss_df)
    
    results.add_row(
        "SDSS",
        "✓ SDSSPreprocessor",
        f"✓ {sdss_pos.shape}",
        f"✓ {sdss_feat.shape}",
        f"✓ {sdss_labels.shape if sdss_labels is not None else 'None'}"
    )
    
    console.print(results)


def test_full_pipeline():
    """Test full pipeline from raw data to graph."""
    console.print("\n[bold blue]Testing Full Pipeline Integration[/bold blue]")
    
    # Create mock raw data
    import polars as pl
    import numpy as np
    
    n_objects = 1000
    mock_df = pl.DataFrame({
        "ra": np.random.uniform(0, 360, n_objects),
        "dec": np.random.uniform(-90, 90, n_objects),
        "parallax": np.random.exponential(1.0, n_objects) + 0.1,
        "phot_g_mean_mag": np.random.normal(15, 2, n_objects),
        "bp_rp": np.random.normal(1.0, 0.5, n_objects),
    })
    
    # Step 1: Preprocessing
    console.print("[yellow]Step 1: Preprocessing[/yellow]")
    preprocessor = get_survey_preprocessor("gaia")
    positions = preprocessor.extract_3d_positions(mock_df)
    features = preprocessor.extract_features(mock_df)
    labels = preprocessor.create_labels(mock_df)
    console.print(f"  ✓ Positions: {positions.shape}")
    console.print(f"  ✓ Features: {features.shape}")
    console.print(f"  ✓ Labels: {labels.shape if labels is not None else 'None'}")
    
    # Step 2: Create SurveyTensorDict
    console.print("\n[yellow]Step 2: Create SurveyTensorDict[/yellow]")
    spatial = SpatialTensorDict(coordinates=torch.tensor(positions, dtype=torch.float32))
    photometric = PhotometricTensorDict(
        magnitudes=torch.tensor(features[:, :3], dtype=torch.float32),
        bands=["g", "bp_rp", "feature_2"]
    )
    survey_tensor = SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name="gaia",
        data_release="dr3"
    )
    console.print("  ✓ SurveyTensorDict created")
    
    # Step 3: Build graphs
    console.print("\n[yellow]Step 3: Build Graphs[/yellow]")
    
    graph_types = {
        "Standard KNN": lambda: create_knn_graph(survey_tensor, k_neighbors=16),
        "Point Cloud": lambda: create_pointcloud_graph(survey_tensor, normalize_positions=True),
        "Multi-scale": lambda: create_multiscale_graph(survey_tensor, scales=[8, 16, 32]),
        "Adaptive": lambda: create_adaptive_graph(survey_tensor),
    }
    
    for name, builder in graph_types.items():
        try:
            graph = builder()
            console.print(f"  ✓ {name}: {graph.num_nodes} nodes, {graph.num_edges} edges")
        except Exception as e:
            console.print(f"  ✗ {name}: {str(e)[:50]}")
    
    console.print("\n[green]✓ Pipeline test completed![/green]")


if __name__ == "__main__":
    console.print("[bold magenta]Astro-Lab Graph Building Test Suite[/bold magenta]")
    console.print("=" * 60)
    
    test_basic_graph_builders()
    test_advanced_graph_builders()
    test_robustness()
    test_preprocessing_improvements()
    test_full_pipeline()
    
    console.print("\n[bold green]All tests completed![/bold green]")
