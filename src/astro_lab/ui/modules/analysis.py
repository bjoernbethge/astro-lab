"""
Analysis UI Module - Direct integration with AstroLab analysis
=============================================================

UI components for data analysis using AstroLab functions.
"""

import marimo as mo
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Direct imports from AstroLab
from astro_lab.data.utils import (
    get_graph_statistics,
    check_graph_consistency,
    sample_subgraph_random,
)
from astro_lab.widgets import cluster_and_analyze
from .data import get_current_datamodule


def analysis_panel() -> mo.Html:
    """Main analysis panel with various analysis options."""
    # Analysis method selector
    analysis_type = mo.ui.dropdown(
        options={
            "graph_metrics": "Graph Metrics",
            "clustering": "Clustering Analysis",
            "statistics": "Statistical Analysis",
            "dimensionality": "Dimensionality Reduction",
            "consistency": "Data Consistency Check",
        },
        value="graph_metrics",
        label="Analysis Type",
    )
    
    def run_analysis():
        """Run selected analysis."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No data loaded!"))
            return
        
        try:
            if analysis_type.value == "graph_metrics":
                # Get graph statistics using actual AstroLab function
                stats = get_graph_statistics(dm._main_data)
                
                mo.output.append(mo.md(f"""
                ### Graph Metrics
                - **Number of Nodes:** {stats['num_nodes']:,}
                - **Number of Edges:** {stats['num_edges']:,}
                - **Average Degree:** {stats['avg_degree']:.2f}
                - **Density:** {stats['density']:.6f}
                - **Has Self-Loops:** {stats['has_self_loops']}
                - **Is Directed:** {stats['is_directed']}
                """))
                
            elif analysis_type.value == "consistency":
                # Check graph consistency
                is_consistent = check_graph_consistency(dm._main_data)
                
                if is_consistent:
                    mo.output.append(mo.md("✅ **Graph data is consistent!**"))
                else:
                    mo.output.append(mo.md("❌ **Graph data has consistency issues!**"))
                    
            elif analysis_type.value == "statistics":
                # Basic statistics on node features
                if hasattr(dm._main_data, 'x') and dm._main_data.x is not None:
                    features = dm._main_data.x.cpu().numpy()
                    
                    mo.output.append(mo.md(f"""
                    ### Feature Statistics
                    - **Shape:** {features.shape}
                    - **Mean:** {features.mean():.4f}
                    - **Std:** {features.std():.4f}
                    - **Min:** {features.min():.4f}
                    - **Max:** {features.max():.4f}
                    
                    #### Per-Feature Statistics:
                    """))
                    
                    # Per-feature stats
                    for i in range(min(5, features.shape[1])):  # Show first 5 features
                        mo.output.append(mo.md(f"""
                        **Feature {i}:**
                        Mean: {features[:, i].mean():.4f}, 
                        Std: {features[:, i].std():.4f}, 
                        Min: {features[:, i].min():.4f}, 
                        Max: {features[:, i].max():.4f}
                        """))
                else:
                    mo.output.append(mo.md("❌ No feature data available"))
                    
            else:
                mo.output.append(mo.md("🔄 Analysis in progress..."))
                # Other analysis types to be implemented
                
        except Exception as e:
            mo.output.append(mo.md(f"❌ Analysis error: {str(e)}"))
    
    run_btn = mo.ui.button(
        "Run Analysis",
        on_click=run_analysis,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 🔬 Analysis Panel"),
        analysis_type,
        run_btn,
    ])


def clustering_tool() -> mo.Html:
    """Clustering analysis tool using AstroLab's cluster_and_analyze."""
    # State
    state, set_state = mo.state({
        "cluster_results": None,
    })
    
    # Algorithm selection
    algorithm = mo.ui.dropdown(
        options={
            "dbscan": "DBSCAN",
            "kmeans": "K-Means",
            "spectral": "Spectral Clustering",
            "hierarchical": "Hierarchical",
        },
        value="dbscan",
        label="Clustering Algorithm",
    )
    
    # Algorithm-specific parameters
    n_clusters = mo.ui.slider(
        value=5,
        min=2,
        max=20,
        step=1,
        label="Number of Clusters",
    )
    
    eps = mo.ui.number(
        value=10.0,
        min=0.1,
        max=100.0,
        step=0.1,
        label="DBSCAN Epsilon",
    )
    
    min_samples = mo.ui.slider(
        value=5,
        min=2,
        max=20,
        step=1,
        label="DBSCAN Min Samples",
    )
    
    def perform_clustering_analysis():
        """Perform clustering using AstroLab's cluster_and_analyze."""
        dm = get_current_datamodule()
        if not dm or not hasattr(dm._main_data, 'x'):
            mo.output.append(mo.md("❌ No feature data available!"))
            return
        
        try:
            # Get coordinates (use first 3 features or position)
            if hasattr(dm._main_data, 'pos') and dm._main_data.pos is not None:
                coords = dm._main_data.pos.cpu().numpy()
            else:
                features = dm._main_data.x.cpu().numpy()
                coords = features[:, :min(3, features.shape[1])]
            
            # Use AstroLab's cluster_and_analyze function
            if algorithm.value == "dbscan":
                results = cluster_and_analyze(
                    coords,
                    algorithm=algorithm.value,
                    eps=eps.value,
                    min_samples=min_samples.value,
                )
            elif algorithm.value == "kmeans":
                results = cluster_and_analyze(
                    coords,
                    algorithm=algorithm.value,
                    n_clusters=n_clusters.value,
                )
            else:
                mo.output.append(mo.md(f"❌ {algorithm.value} not implemented in cluster_and_analyze"))
                return
            
            set_state(lambda s: {**s, "cluster_results": results})
            
            # Display results
            labels = results.get("labels", [])
            if hasattr(labels, '__len__'):
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
            else:
                n_clusters_found = "Unknown"
                n_noise = "Unknown"
            
            mo.output.append(mo.md(f"""
            ### Clustering Results
            - **Algorithm:** {algorithm.value}
            - **Clusters found:** {n_clusters_found}
            - **Noise points:** {n_noise}
            - **Total points:** {len(coords)}
            """))
            
            # Additional metrics if available
            if "silhouette_score" in results:
                mo.output.append(mo.md(f"- **Silhouette Score:** {results['silhouette_score']:.4f}"))
                
        except Exception as e:
            mo.output.append(mo.md(f"❌ Clustering error: {str(e)}"))
    
    cluster_btn = mo.ui.button(
        "Perform Clustering",
        on_click=perform_clustering_analysis,
        kind="primary",
    )
    
    # Parameter display based on algorithm
    if algorithm.value == "kmeans":
        params = mo.vstack([n_clusters])
    elif algorithm.value == "dbscan":
        params = mo.vstack([eps, min_samples])
    else:
        params = mo.md("*Select algorithm to see parameters*")
    
    return mo.vstack([
        mo.md("## 🎯 Clustering Tool"),
        algorithm,
        params,
        cluster_btn,
    ])


def statistics_viewer() -> mo.Html:
    """View statistical analysis results."""
    # State
    state, set_state = mo.state({
        "stats_results": None,
        "pca_results": None,
    })
    
    # Statistics options
    stats_options = mo.ui.multiselect(
        options=[
            "basic_stats",
            "correlation_matrix",
            "pca_analysis",
            "feature_importance",
        ],
        value=["basic_stats"],
        label="Statistics to Compute",
    )
    
    # PCA components
    n_components = mo.ui.slider(
        value=3,
        min=2,
        max=10,
        step=1,
        label="PCA Components",
    )
    
    def compute_stats():
        """Compute selected statistics."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No data loaded!"))
            return
        
        try:
            results = {}
            
            if "basic_stats" in stats_options.value:
                if hasattr(dm._main_data, 'x') and dm._main_data.x is not None:
                    features = dm._main_data.x.cpu()
                    
                    # Compute basic statistics
                    results["basic"] = {
                        "shape": list(features.shape),
                        "mean": features.mean().item(),
                        "std": features.std().item(),
                        "min": features.min().item(),
                        "max": features.max().item(),
                    }
                    
                    # Per-feature stats
                    feature_stats = []
                    for i in range(min(10, features.shape[1])):
                        feature_stats.append({
                            "feature": i,
                            "mean": features[:, i].mean().item(),
                            "std": features[:, i].std().item(),
                            "min": features[:, i].min().item(),
                            "max": features[:, i].max().item(),
                        })
                    results["per_feature"] = feature_stats
            
            if "pca_analysis" in stats_options.value:
                if hasattr(dm._main_data, 'x') and dm._main_data.x is not None:
                    features = dm._main_data.x.cpu().numpy()
                    
                    # Perform PCA
                    pca = PCA(n_components=min(n_components.value, features.shape[1]))
                    pca_features = pca.fit_transform(features)
                    
                    results["pca"] = {
                        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                        "n_components": pca.n_components_,
                    }
                    
                    set_state(lambda s: {**s, "pca_results": pca_features})
            
            set_state(lambda s: {**s, "stats_results": results})
            mo.output.append(mo.md("✅ Statistics computed successfully!"))
            
            # Display results
            if "basic" in results:
                basic = results["basic"]
                mo.output.append(mo.md(f"""
                ### Basic Statistics
                - **Shape:** {basic['shape']}
                - **Mean:** {basic['mean']:.4f}
                - **Std:** {basic['std']:.4f}
                - **Min:** {basic['min']:.4f}
                - **Max:** {basic['max']:.4f}
                """))
            
            if "pca" in results:
                pca_info = results["pca"]
                mo.output.append(mo.md(f"""
                ### PCA Analysis
                - **Components:** {pca_info['n_components']}
                - **Explained Variance:** {[f"{v:.2%}" for v in pca_info['explained_variance_ratio']]}
                - **Cumulative Variance:** {pca_info['cumulative_variance'][-1]:.2%}
                """))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Statistics error: {str(e)}"))
    
    compute_btn = mo.ui.button(
        "Compute Statistics",
        on_click=compute_stats,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 📊 Statistics Viewer"),
        stats_options,
        n_components,
        compute_btn,
    ])


def subgraph_sampler() -> mo.Html:
    """Tool for sampling subgraphs."""
    # Sampling parameters
    max_nodes = mo.ui.slider(
        value=1000,
        min=100,
        max=10000,
        step=100,
        label="Max Nodes in Subgraph",
    )
    
    seed = mo.ui.number(
        value=42,
        min=0,
        max=9999,
        step=1,
        label="Random Seed",
    )
    
    def sample_subgraph():
        """Sample a subgraph using AstroLab's sample_subgraph_random."""
        dm = get_current_datamodule()
        if not dm or not hasattr(dm, '_main_data'):
            mo.output.append(mo.md("❌ No graph data loaded!"))
            return
        
        try:
            original_graph = dm._main_data
            
            # Sample subgraph using AstroLab function
            subgraph = sample_subgraph_random(
                original_graph,
                max_nodes=max_nodes.value,
                seed=int(seed.value),
            )
            
            # Display results
            mo.output.append(mo.md(f"""
            ### Subgraph Sampled
            
            **Original Graph:**
            - Nodes: {original_graph.num_nodes:,}
            - Edges: {original_graph.num_edges:,}
            
            **Subgraph:**
            - Nodes: {subgraph.num_nodes:,}
            - Edges: {subgraph.num_edges:,}
            - Sampling ratio: {(subgraph.num_nodes / original_graph.num_nodes):.2%}
            """))
            
            # Check consistency of subgraph
            if check_graph_consistency(subgraph):
                mo.output.append(mo.md("✅ Subgraph is consistent"))
            else:
                mo.output.append(mo.md("❌ Subgraph has consistency issues"))
                
        except Exception as e:
            mo.output.append(mo.md(f"❌ Sampling error: {str(e)}"))
    
    sample_btn = mo.ui.button(
        "Sample Subgraph",
        on_click=sample_subgraph,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 🎲 Subgraph Sampler"),
        mo.md("*Sample a smaller subgraph from the loaded graph data*"),
        max_nodes,
        seed,
        sample_btn,
    ])
