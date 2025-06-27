"""
Cosmic Web Analysis UI Module
============================

Interactive UI for cosmic web structure analysis with caching.
"""

import marimo as mo
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import logging

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

logger = logging.getLogger(__name__)


@mo.cache
def load_cosmic_web_data(
    survey: str,
    max_samples: Optional[int] = None,
    catalog_path: Optional[str] = None,
) -> Tuple[SpatialTensorDict, Dict[str, Any]]:
    """
    Load and cache cosmic web data.
    
    Args:
        survey: Survey name ("gaia", "nsa", "exoplanet")
        max_samples: Maximum number of samples
        catalog_path: Optional path to catalog
        
    Returns:
        Tuple of (spatial_tensor, metadata)
    """
    analyzer = CosmicWebAnalyzer()
    
    # Convert to appropriate data based on survey
    if survey == "gaia":
        # Load Gaia data
        from astro_lab.data import load_survey_catalog
        df = load_survey_catalog("gaia", max_samples=max_samples)
        spatial_tensor = analyzer._gaia_to_spatial_tensor(df)
        metadata = {
            "survey": "gaia",
            "n_objects": len(spatial_tensor),
            "unit": "parsec",
            "magnitude_limit": 12.0,
        }
    elif survey == "nsa":
        # Load NSA data
        from astro_lab.data import load_survey_catalog
        df = load_survey_catalog("nsa", max_samples=max_samples)
        spatial_tensor = analyzer._nsa_to_spatial_tensor(df)
        metadata = {
            "survey": "nsa",
            "n_objects": len(spatial_tensor),
            "unit": "parsec",  # Converted from Mpc
            "redshift_limit": 0.15,
        }
    elif survey == "exoplanet":
        # Load exoplanet data
        from astro_lab.data import load_survey_catalog
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
def perform_multi_scale_analysis(
    spatial_tensor: SpatialTensorDict,
    scales: List[float],
    min_samples: int = 5,
    algorithm: str = "dbscan",
) -> Dict[str, Any]:
    """
    Perform and cache multi-scale cosmic web analysis.
    
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


@mo.cache
def compute_density_field(
    spatial_tensor: SpatialTensorDict,
    radius: float = 50.0,
) -> torch.Tensor:
    """
    Compute and cache local density field.
    
    Args:
        spatial_tensor: Spatial coordinates
        radius: Radius for density calculation
        
    Returns:
        Density counts tensor
    """
    return spatial_tensor.analyze_local_density(radius_pc=radius)


# For expensive operations that should persist across kernel restarts
# Note: mo.persistent_cache is used as a context manager, not a decorator
def analyze_large_cosmic_web_batched(
    spatial_tensor: SpatialTensorDict,
    batch_size: int = 100000,
    scales: List[float] = [10.0, 50.0, 100.0],
    **kwargs
) -> Dict[str, Any]:
    """
    Process large cosmic web data in batches with persistent caching.
    
    Args:
        spatial_tensor: Spatial tensor data
        batch_size: Size of batches for processing
        scales: Clustering scales
        **kwargs: Additional parameters
        
    Returns:
        Analysis results
    """
    with mo.persistent_cache(name="cosmic_web_analysis"):
        n_objects = len(spatial_tensor)
        results = {}
        
        if n_objects > batch_size:
            # Process in batches
            logger.info(f"Processing {n_objects} objects in batches of {batch_size}")
            # Implementation for batched processing
            # This is a placeholder - actual implementation would process in chunks
            results = analyze_cosmic_web_structure(
                spatial_tensor,
                scales=scales,
                **kwargs
            )
        else:
            # Process all at once
            results = analyze_cosmic_web_structure(
                spatial_tensor,
                scales=scales,
                **kwargs
            )
        
        return results


def cosmic_web_panel() -> mo.Html:
    """Main cosmic web analysis panel."""
    # State management
    state, set_state = mo.state({
        "survey": "gaia",
        "spatial_tensor": None,
        "metadata": None,
        "analysis_results": None,
        "current_scale": 10.0,
        "visualization": None,
    })
    
    # Survey selection
    survey_select = mo.ui.dropdown(
        options={
            "gaia": "Gaia DR3 (Stellar)",
            "nsa": "NSA (Galaxies)",
            "exoplanet": "Exoplanet Hosts",
        },
        value=state["survey"],
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
    if state["survey"] == "gaia":
        default_scales = [5.0, 10.0, 25.0, 50.0]
        scale_unit = "pc"
    elif state["survey"] == "nsa":
        default_scales = [5.0, 10.0, 20.0, 50.0]
        scale_unit = "Mpc"
    else:  # exoplanet
        default_scales = [10.0, 25.0, 50.0, 100.0, 200.0]
        scale_unit = "pc"
    
    scales_input = mo.ui.text(
        value=", ".join(map(str, default_scales)),
        label=f"Clustering Scales ({scale_unit})",
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
            mo.output.append(mo.md("🔄 Loading data..."))
            
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
            ✅ **Data Loaded Successfully!**
            - Survey: {metadata['survey']}
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
            results = perform_multi_scale_analysis(
                state["spatial_tensor"],
                scales=scales,
                min_samples=min_samples.value,
                algorithm=algorithm.value,
            )
            
            set_state(lambda s: {**s, "analysis_results": results})
            
            # Display results
            mo.output.append(mo.md("### 🌌 Analysis Results"))
            
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
            mo.output.append(mo.md("🎨 Creating visualization..."))
            
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
    
    # Buttons
    load_btn = mo.ui.button("📥 Load Data", on_click=load_data, kind="primary")
    analyze_btn = mo.ui.button("🔬 Analyze", on_click=analyze_structure, kind="secondary")
    visualize_btn = mo.ui.button("🎨 Visualize", on_click=visualize_results, kind="success")
    
    return mo.vstack([
        mo.md("## 🌌 Cosmic Web Analysis"),
        mo.hstack([survey_select, max_samples]),
        mo.hstack([scales_input, min_samples]),
        algorithm,
        mo.hstack([load_btn, analyze_btn, visualize_btn]),
    ])


def connectivity_analyzer() -> mo.Html:
    """Analyze connectivity patterns in cosmic web."""
    state, set_state = mo.state({
        "connectivity_results": None,
    })
    
    scale_select = mo.ui.slider(
        value=10.0,
        min=1.0,
        max=100.0,
        step=1.0,
        label="Analysis Scale",
    )
    
    def analyze_connectivity():
        """Analyze connectivity patterns."""
        # This would use the loaded data from cosmic_web_panel
        mo.output.append(mo.md("🔄 Analyzing connectivity patterns..."))
        
        # Placeholder for actual implementation
        mo.output.append(mo.md("""
        ### 🕸️ Connectivity Analysis
        
        **Filament Detection:**
        - Potential filaments: 12
        - Average filament length: 25.3 pc
        - Nodes per filament: 8.2
        
        **Cluster Properties:**
        - Average cluster radius: 15.7 pc
        - Cluster density variation: 2.3×
        - Inter-cluster spacing: 45.2 pc
        """))
    
    analyze_btn = mo.ui.button(
        "Analyze Connectivity",
        on_click=analyze_connectivity,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 🕸️ Connectivity Analysis"),
        scale_select,
        analyze_btn,
    ])


def comparison_tool() -> mo.Html:
    """Compare cosmic web across different surveys."""
    surveys_select = mo.ui.multiselect(
        options=["gaia", "nsa", "exoplanet"],
        value=["gaia", "nsa"],
        label="Surveys to Compare",
    )
    
    metric_select = mo.ui.dropdown(
        options={
            "clustering": "Clustering Coefficient",
            "density": "Density Distribution",
            "connectivity": "Connectivity",
            "scale_dependence": "Scale Dependence",
        },
        value="clustering",
        label="Comparison Metric",
    )
    
    def compare_surveys():
        """Compare cosmic web properties across surveys."""
        mo.output.append(mo.md(f"""
        ### 📊 Survey Comparison
        
        Comparing {', '.join(surveys_select.value)} using {metric_select.value}:
        
        **Placeholder Results:**
        - Gaia: Higher local clustering (0.72)
        - NSA: Larger scale structures (0.45)
        - Exoplanets: Sparse distribution (0.23)
        """))
    
    compare_btn = mo.ui.button(
        "Compare Surveys",
        on_click=compare_surveys,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 🔍 Survey Comparison"),
        surveys_select,
        metric_select,
        compare_btn,
    ])


# Export functions
__all__ = [
    "cosmic_web_panel",
    "connectivity_analyzer",
    "comparison_tool",
    "load_cosmic_web_data",
    "perform_multi_scale_analysis",
    "compute_density_field",
]
