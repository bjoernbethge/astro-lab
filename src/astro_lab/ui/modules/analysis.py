"""
Analysis UI Module - Direct integration with AstroLab analysis
=============================================================

UI components for data analysis using AstroLab functions.
"""

import marimo as mo
from typing import Dict, List, Optional, Any, Tuple
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

# Import cosmic web functionality
from astro_lab.data.cosmic_web import CosmicWebAnalyzer
from astro_lab.tensors import SpatialTensorDict
from astro_lab.widgets.graph import (
    analyze_cosmic_web_structure,
    cosmic_web_connectivity_analysis,
)
from astro_lab.widgets.plotly_bridge import (
    plot_cosmic_web_3d,
    plot_density_heatmap,
    plot_multi_scale_clustering,
)


def analysis_panel() -> mo.Html:
    """Main analysis panel with various analysis options."""
    # Analysis method selector
    analysis_type = mo.ui.dropdown(
        options={
            "graph_metrics": "Graph Metrics",
            "clustering": "Clustering Analysis",
            "statistics": "Statistical Analysis",
            "dimensionality": "Dimensionality Reduction",
            "cosmic_web": "🌌 Cosmic Web Analysis",
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
                
            elif analysis_type.value == "cosmic_web":
                # Show cosmic web analysis interface
                mo.output.append(cosmic_web_analysis_interface())
                
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
            "agglomerative": "Hierarchical",
            "meanshift": "Mean Shift",
            "optics": "OPTICS",
            "hdbscan": "HDBSCAN",
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
        label="DBSCAN/OPTICS Epsilon",
    )
    
    min_samples = mo.ui.slider(
        value=5,
        min=2,
        max=20,
        step=1,
        label="Min Samples",
    )
    
    bandwidth = mo.ui.number(
        value=10.0,
        min=0.1,
        max=100.0,
        step=0.1,
        label="Mean Shift Bandwidth",
    )
    
    min_cluster_size = mo.ui.slider(
        value=5,
        min=2,
        max=50,
        step=1,
        label="HDBSCAN Min Cluster Size",
    )
    
    def perform_clustering_analysis():
        """Perform clustering using AstroLab's cluster_and_analyze or scikit-learn."""
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
            
            # Use appropriate clustering based on algorithm
            if algorithm.value in ["dbscan", "kmeans"]:
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
                # Use scikit-learn directly for other algorithms
                from sklearn.cluster import SpectralClustering, AgglomerativeClustering, MeanShift, OPTICS
                try:
                    import hdbscan
                    HDBSCAN_AVAILABLE = True
                except ImportError:
                    HDBSCAN_AVAILABLE = False
                
                if algorithm.value == "spectral":
                    clusterer = SpectralClustering(n_clusters=n_clusters.value, affinity='nearest_neighbors')
                    labels = clusterer.fit_predict(coords)
                elif algorithm.value == "agglomerative":
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters.value)
                    labels = clusterer.fit_predict(coords)
                elif algorithm.value == "meanshift":
                    clusterer = MeanShift(bandwidth=bandwidth.value)
                    labels = clusterer.fit_predict(coords)
                elif algorithm.value == "optics":
                    clusterer = OPTICS(min_samples=min_samples.value, eps=eps.value)
                    labels = clusterer.fit_predict(coords)
                elif algorithm.value == "hdbscan" and HDBSCAN_AVAILABLE:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size.value)
                    labels = clusterer.fit_predict(coords)
                else:
                    mo.output.append(mo.md(f"❌ {algorithm.value} not available"))
                    return
                
                results = {"labels": labels}
            
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
    if algorithm.value in ["kmeans", "spectral", "agglomerative"]:
        params = mo.vstack([n_clusters])
    elif algorithm.value in ["dbscan", "optics"]:
        params = mo.vstack([eps, min_samples])
    elif algorithm.value == "meanshift":
        params = mo.vstack([bandwidth])
    elif algorithm.value == "hdbscan":
        params = mo.vstack([min_cluster_size])
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


# ===== COSMIC WEB ANALYSIS FUNCTIONS =====

@mo.cache
def load_cosmic_web_data(
    survey: str,
    max_samples: Optional[int] = None,
) -> Tuple[SpatialTensorDict, Dict[str, Any]]:
    """
    Load and cache cosmic web data.
    
    Args:
        survey: Survey name ("gaia", "nsa", "exoplanet")
        max_samples: Maximum number of samples
        
    Returns:
        Tuple of (spatial_tensor, metadata)
    """
    analyzer = CosmicWebAnalyzer()
    
    # Load survey data
    from astro_lab.data import load_survey_catalog
    
    if survey == "gaia":
        df = load_survey_catalog("gaia", max_samples=max_samples)
        spatial_tensor = analyzer._gaia_to_spatial_tensor(df)
        metadata = {
            "survey": "gaia",
            "n_objects": len(spatial_tensor),
            "unit": "parsec",
            "magnitude_limit": 12.0,
        }
    elif survey == "nsa":
        df = load_survey_catalog("nsa", max_samples=max_samples)
        spatial_tensor = analyzer._nsa_to_spatial_tensor(df)
        metadata = {
            "survey": "nsa",
            "n_objects": len(spatial_tensor),
            "unit": "parsec",
            "redshift_limit": 0.15,
        }
    elif survey == "exoplanet":
        df = load_survey_catalog("exoplanet", max_samples=max_samples)
        spatial_tensor = analyzer._exoplanet_to_spatial_tensor(df)
        metadata = {
            "survey": "exoplanet",
            "n_objects": len(spatial_tensor),
            "unit": "parsec",
        }
    else:
        raise ValueError(f"Unknown survey: {survey}")
        
    return spatial_tensor, metadata


@mo.cache
def perform_cosmic_web_analysis(
    spatial_tensor: SpatialTensorDict,
    scales: List[float],
    min_samples: int = 5,
    algorithm: str = "dbscan",
) -> Dict[str, Any]:
    """
    Perform cached multi-scale cosmic web analysis.
    
    Args:
        spatial_tensor: Spatial coordinates
        scales: List of clustering scales
        min_samples: Minimum samples for clustering
        algorithm: Clustering algorithm
        
    Returns:
        Analysis results
    """
    return analyze_cosmic_web_structure(
        spatial_tensor,
        scales=scales,
        min_samples=min_samples,
        algorithm=algorithm,
        use_existing_analyzer=True,
    )


def cosmic_web_analysis_interface() -> mo.Html:
    """Create the cosmic web analysis interface."""
    # State management
    state, set_state = mo.state({
        "survey": "gaia",
        "spatial_tensor": None,
        "metadata": None,
        "analysis_results": None,
    })
    
    # Survey selection
    survey_select = mo.ui.dropdown(
        options={
            "gaia": "Gaia DR3 (Stellar)",
            "nsa": "NSA (Galaxies)",
            "exoplanet": "Exoplanet Hosts",
        },
        value="gaia",
        label="Select Survey",
    )
    
    # Analysis parameters
    max_samples = mo.ui.number(
        value=10000,
        min=1000,
        max=1000000,
        step=1000,
        label="Max Samples",
    )
    
    # Scale selection based on survey
    default_scales = {
        "gaia": [5.0, 10.0, 25.0, 50.0],
        "nsa": [5.0, 10.0, 20.0, 50.0],
        "exoplanet": [10.0, 25.0, 50.0, 100.0, 200.0],
    }
    
    scales_input = mo.ui.text(
        value=", ".join(map(str, default_scales.get(survey_select.value, [10.0, 50.0]))),
        label=f"Clustering Scales (pc)",
    )
    
    min_samples = mo.ui.slider(
        value=5,
        min=2,
        max=20,
        step=1,
        label="Min Samples (DBSCAN)",
    )
    
    algorithm = mo.ui.dropdown(
        options={
            "dbscan": "DBSCAN",
            "kmeans": "K-Means",
            "agglomerative": "Hierarchical",
            "spectral": "Spectral",
        },
        value="dbscan",
        label="Clustering Algorithm",
    )
    
    def load_data():
        """Load cosmic web data."""
        try:
            mo.output.append(mo.md("🔄 Loading cosmic web data..."))
            
            spatial_tensor, metadata = load_cosmic_web_data(
                survey_select.value,
                max_samples=int(max_samples.value),
            )
            
            set_state(lambda s: {
                **s,
                "survey": survey_select.value,
                "spatial_tensor": spatial_tensor,
                "metadata": metadata,
            })
            
            mo.output.append(mo.md(f"""
            ✅ **Cosmic Web Data Loaded!**
            - Survey: {metadata['survey'].upper()}
            - Objects: {metadata['n_objects']:,}
            - Unit: {metadata['unit']}
            """))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error loading data: {str(e)}"))
    
    def analyze_structure():
        """Perform cosmic web analysis."""
        if state["spatial_tensor"] is None:
            mo.output.append(mo.md("❌ Please load data first!"))
            return
            
        try:
            mo.output.append(mo.md("🔄 Analyzing cosmic web structure..."))
            
            # Parse scales
            scales = [float(s.strip()) for s in scales_input.value.split(",")]
            
            # Perform analysis
            results = perform_cosmic_web_analysis(
                state["spatial_tensor"],
                scales=scales,
                min_samples=min_samples.value,
                algorithm=algorithm.value,
            )
            
            set_state(lambda s: {**s, "analysis_results": results})
            
            # Display results
            mo.output.append(mo.md("### 🌌 Cosmic Web Analysis Results"))
            
            for scale_key, stats in results["clustering_results"].items():
                mo.output.append(mo.md(f"""
                **Scale: {scale_key}**
                - Clusters: {stats['n_clusters']}
                - Grouped: {stats['n_grouped']:,} ({stats['grouped_fraction']:.1%})
                - Isolated: {stats['n_noise']:,}
                """))
                
            # Show density statistics if available
            if results.get("density_analysis"):
                density_stats = results["density_analysis"]
                mo.output.append(mo.md(f"""
                **Density Analysis:**
                - Mean density: {density_stats['mean_density']:.2f} neighbors
                - Std density: {density_stats['std_density']:.2f}
                """))
                
        except Exception as e:
            mo.output.append(mo.md(f"❌ Analysis error: {str(e)}"))
    
    def visualize_results():
        """Create visualization."""
        if state["spatial_tensor"] is None or state["analysis_results"] is None:
            mo.output.append(mo.md("❌ Please load data and run analysis first!"))
            return
            
        try:
            mo.output.append(mo.md("🎨 Creating cosmic web visualization..."))
            
            # Get first scale results for visualization
            first_scale = list(state["analysis_results"]["clustering_results"].keys())[0]
            cluster_labels = state["analysis_results"]["clustering_results"][first_scale]["labels"]
            
            # Create 3D plot
            fig = plot_cosmic_web_3d(
                state["spatial_tensor"],
                cluster_labels=cluster_labels,
                title=f"Cosmic Web Structure - {state['metadata']['survey'].upper()}",
                point_size=3,
                show_clusters=True,
                unit=state["metadata"]["unit"],
            )
            
            mo.output.append(mo.plotly(fig))
            
            # Create multi-scale comparison if multiple scales
            if len(state["analysis_results"]["clustering_results"]) > 1:
                multi_fig = plot_multi_scale_clustering(
                    state["spatial_tensor"],
                    state["analysis_results"]["clustering_results"],
                    title="Multi-Scale Clustering Comparison",
                )
                mo.output.append(mo.plotly(multi_fig))
                
            # Create density heatmap if available
            if state["analysis_results"].get("density_analysis"):
                density_counts = state["analysis_results"]["density_analysis"]["counts"]
                density_fig = plot_density_heatmap(
                    state["spatial_tensor"],
                    density_counts,
                    radius=state["analysis_results"]["scales"][0],
                    title="Local Density Distribution",
                )
                mo.output.append(mo.plotly(density_fig))
                
        except Exception as e:
            mo.output.append(mo.md(f"❌ Visualization error: {str(e)}"))
    
    def analyze_filaments():
        """Detect and analyze filamentary structures."""
        if state["spatial_tensor"] is None:
            mo.output.append(mo.md("❌ Please load data first!"))
            return
            
        try:
            mo.output.append(mo.md("🔄 Detecting cosmic filaments..."))
            
            # Use the CosmicWebAnalyzer for filament detection
            analyzer = CosmicWebAnalyzer()
            filament_results = analyzer.detect_filaments(
                state["spatial_tensor"],
                method="mst",  # or other methods
                n_neighbors=20,
                distance_threshold=scales[0] * 2 if scales else 100.0
            )
            
            mo.output.append(mo.md(f"""
            ### 🌊 Filament Detection Results
            - Method: {filament_results['method']}
            - Filament segments: {filament_results['n_filament_segments']}
            - Mean segment length: {filament_results['mean_segment_length']:.2f}
            - Total filament length: {filament_results['total_filament_length']:.2f}
            """))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Filament detection error: {str(e)}"))
    
    # Action buttons
    load_btn = mo.ui.button("📥 Load Data", on_click=load_data, kind="primary")
    analyze_btn = mo.ui.button("🔬 Analyze", on_click=analyze_structure, kind="secondary")
    visualize_btn = mo.ui.button("🎨 Visualize", on_click=visualize_results, kind="success")
    filament_btn = mo.ui.button("🌊 Detect Filaments", on_click=analyze_filaments, kind="info")
    
    return mo.vstack([
        mo.md("### 🌌 Cosmic Web Analysis"),
        mo.hstack([survey_select, max_samples]),
        mo.hstack([scales_input, min_samples]),
        algorithm,
        mo.hstack([load_btn, analyze_btn, visualize_btn, filament_btn]),
    ])
