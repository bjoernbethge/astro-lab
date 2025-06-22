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
    
    # Generate 3D coordinates (galaxy positions)
    coords = torch.randn(n_points, 3) * 100  # 3D positions
    
    # Generate photometric data (magnitudes in different bands)
    photometric = torch.randn(n_points, 5)  # 5 photometric bands
    
    # Combine features
    features = torch.cat([coords, photometric], dim=1)
    
    # Create PyTorch Geometric Data object
    from torch_geometric.data import Data
    pyg_data = Data(x=features)
    
    # Create AstroDataset
    dataset = AstroDataset([pyg_data], survey="sample_survey")
    
    logger.info(f"Created dataset with {len(dataset)} samples")
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
