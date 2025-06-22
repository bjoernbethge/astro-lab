#!/usr/bin/env python3
"""
Simple Widget Example - Demonstrates the modular AstroLab Widget
===============================================================

Shows how to use the separated modules for graph creation, clustering, and analysis.
"""

import logging
import numpy as np
import torch

from astro_lab.data.core import AstroDataset
from astro_lab.widgets import AstroLabWidget

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_points: int = 1000) -> AstroDataset:
    """Create sample astronomical data for demonstration."""
    logger.info(f"Creating sample data with {n_points} points...")
    
    # For demonstration, we'll create a simple dataset
    # In practice, you would load real survey data
    try:
        # Try to load existing Gaia data as example
        dataset = AstroDataset(survey="gaia", max_samples=n_points, k_neighbors=8)
        logger.info(f"Loaded existing Gaia dataset with {len(dataset)} samples")
        return dataset
    except FileNotFoundError:
        # Fallback: create a mock dataset
        logger.info("No existing data found, creating mock dataset...")
        
        # Create mock data structure
        from torch_geometric.data import Data
        import tempfile
        import os
        
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        raw_dir = os.path.join(temp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # Create mock Parquet file
        import polars as pl
        
        # Generate mock astronomical data
        coords = torch.randn(n_points, 3) * 100  # 3D positions
        photometric = torch.randn(n_points, 5)  # 5 photometric bands
        
        # Create DataFrame
        df = pl.DataFrame({
            "ra": coords[:, 0].numpy(),
            "dec": coords[:, 1].numpy(),
            "distance": coords[:, 2].numpy(),
            "g_mag": photometric[:, 0].numpy(),
            "r_mag": photometric[:, 1].numpy(),
            "i_mag": photometric[:, 2].numpy(),
            "z_mag": photometric[:, 3].numpy(),
            "y_mag": photometric[:, 4].numpy(),
        })
        
        # Save as Parquet
        parquet_path = os.path.join(raw_dir, "gaia.parquet")
        df.write_parquet(parquet_path)
        
        # Create dataset with mock data
        dataset = AstroDataset(
            survey="gaia", 
            max_samples=n_points, 
            k_neighbors=8,
            root=temp_dir
        )
        
        logger.info(f"Created mock dataset with {len(dataset)} samples")
        return dataset


def main():
    """Main demonstration function."""
    logger.info("Starting AstroLab Widget demonstration...")
    
    # Create widget
    widget = AstroLabWidget()
    logger.info("Widget initialized successfully")
    
    # Create sample data
    dataset = create_sample_data(n_points=500)
    
    # 1. GRAPH MODULE - Create PyTorch Geometric graph
    logger.info("\n=== GRAPH MODULE ===")
    
    # Create graph with k-NN
    graph_data = widget.create_graph(dataset, k=8, use_gpu=True)
    logger.info(f"Created graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Find neighbors
    neighbors = widget.find_neighbors(dataset, k=5, use_gpu=True)
    logger.info(f"Found {neighbors['edge_index'].shape[1]} neighbor connections")
    
    # Get model input features
    features = widget.get_model_input_features(dataset)
    logger.info(f"Available features: {list(features.keys())}")
    
    # 2. CLUSTERING MODULE - GPU-accelerated clustering
    logger.info("\n=== CLUSTERING MODULE ===")
    
    # Perform clustering
    clustering_result = widget.cluster_data(
        dataset, 
        eps=15.0, 
        min_samples=3, 
        algorithm="dbscan", 
        use_gpu=True
    )
    
    logger.info(f"Clustering results:")
    logger.info(f"  - Found {clustering_result['n_clusters']} clusters")
    logger.info(f"  - {clustering_result['n_noise']} noise points")
    
    # Show cluster statistics
    for cluster_id, stats in clustering_result['cluster_stats'].items():
        logger.info(f"  - Cluster {cluster_id}: {stats['n_points']} points, "
                   f"radius {stats['radius']:.2f}, density {stats['density']:.2f}")
    
    # 3. ANALYSIS MODULE - Structure and density analysis
    logger.info("\n=== ANALYSIS MODULE ===")
    
    # Analyze density
    densities = widget.analyze_density(dataset, radius=10.0, use_gpu=True)
    logger.info(f"Density analysis: mean={densities.mean():.3f}, std={densities.std():.3f}")
    
    # Analyze structure
    structure = widget.analyze_structure(dataset, k=8, use_gpu=True)
    logger.info(f"Structure analysis:")
    logger.info(f"  - {structure['num_nodes']} nodes, {structure['num_edges']} edges")
    logger.info(f"  - Average degree: {structure['avg_degree']:.2f}")
    logger.info(f"  - Graph density: {structure['graph_density']:.4f}")
    
    # 4. VISUALIZATION - Plot results
    logger.info("\n=== VISUALIZATION ===")
    
    try:
        # Plot original data
        viz = widget.plot(dataset, plot_type="scatter_3d", backend="auto", max_points=200)
        logger.info("Visualization created successfully")
        
        # Show the plot
        widget.show()
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        logger.info("This is normal if no display is available")
    
    logger.info("\n=== DEMONSTRATION COMPLETE ===")
    logger.info("All modules working correctly!")


if __name__ == "__main__":
    main()
