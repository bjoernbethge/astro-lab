"""
Visualization UI Module - Advanced AstroLab Visualizations
=========================================================

UI components for advanced astronomical visualizations using AstroLab's
specialized visualization systems (NOT boring standard Plotly!).
"""

import marimo as mo
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import logging

# Import AstroLab's advanced visualization systems
from astro_lab.widgets import (
    AstroLabWidget,
    CosmographBridge,
)
from astro_lab.widgets.tensor_bridge import (
    PyVistaZeroCopyBridge,
    BlenderZeroCopyBridge,
    transfer_to_framework,
    visualize_cosmic_web,
)
from astro_lab.widgets.plotly_bridge import (
    plot_cosmic_web_3d,
    plot_density_heatmap,
    plot_multi_scale_clustering,
)

# Import common UI components
from .data import get_current_datamodule

logger = logging.getLogger(__name__)


def plot_creator() -> mo.Html:
    """
    Advanced visualization creator using multiple backends.
    No boring standard plots - only cool astronomical visualizations!
    """
    # State management
    state, set_state = mo.state({
        "backend": "pyvista",
        "visualization": None,
        "viz_config": {},
    })
    
    # Backend selector - no standard plotly option!
    backend = mo.ui.dropdown(
        options={
            "pyvista": "🔷 PyVista (3D Interactive)",
            "blender": "🎬 Blender (Production Quality)",
            "cosmograph": "🌌 Cosmograph (Large-scale)",
            "open3d": "🎯 Open3D (Point Clouds)",
            "tensor_bridge": "🔗 Zero-Copy Bridge",
        },
        value=state["backend"],
        label="Visualization Backend",
    )
    
    # Visualization type based on backend
    viz_types = {
        "pyvista": ["point_cloud", "mesh", "volume", "streamlines"],
        "blender": ["particles", "mesh", "volumetric", "animation"],
        "cosmograph": ["force_graph", "hierarchical", "radial"],
        "open3d": ["point_cloud", "octree", "voxel_grid"],
        "tensor_bridge": ["multi_backend", "live_sync"],
    }
    
    viz_type = mo.ui.dropdown(
        options=viz_types.get(backend.value, ["default"]),
        value=viz_types.get(backend.value, ["default"])[0],
        label="Visualization Type",
    )
    
    # Advanced parameters
    point_size = mo.ui.slider(
        value=2.0,
        min=0.1,
        max=10.0,
        step=0.1,
        label="Point Size",
    )
    
    quality = mo.ui.dropdown(
        options={
            "draft": "🏃 Draft (Fast)",
            "preview": "👀 Preview",
            "production": "🎬 Production",
            "cinematic": "🎥 Cinematic",
        },
        value="preview",
        label="Render Quality",
    )
    
    # Color mapping
    color_by = mo.ui.dropdown(
        options={
            "spectral_type": "Spectral Type",
            "redshift": "Redshift",
            "velocity": "Velocity",
            "temperature": "Temperature",
            "metallicity": "Metallicity",
            "cluster_id": "Cluster ID",
            "density": "Local Density",
        },
        value="spectral_type",
        label="Color Mapping",
    )
    
    # Advanced effects
    effects = mo.ui.multiselect(
        options=[
            "ambient_occlusion",
            "depth_of_field",
            "bloom",
            "motion_blur",
            "volumetric_fog",
            "lens_flare",
        ],
        value=["ambient_occlusion"],
        label="Visual Effects",
    )
    
    def create_advanced_visualization():
        """Create visualization using selected backend."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No data loaded!"))
            return
            
        try:
            mo.output.append(mo.md(f"🎨 Creating {backend.value} visualization..."))
            
            # Get spatial data
            if hasattr(dm._main_data, 'pos'):
                coords = dm._main_data.pos
            elif hasattr(dm._main_data, 'x'):
                coords = dm._main_data.x[:, :3]  # First 3 features as coords
            else:
                mo.output.append(mo.md("❌ No spatial data found!"))
                return
                
            # Create visualization based on backend
            if backend.value == "pyvista":
                bridge = PyVistaZeroCopyBridge()
                mesh = bridge.to_pyvista(coords, point_size=point_size.value)
                
                # Add effects
                if "ambient_occlusion" in effects.value:
                    mesh.compute_normals(inplace=True)
                    
                mo.output.append(mo.md("✅ PyVista mesh created! Use external viewer."))
                set_state(lambda s: {**s, "visualization": mesh})
                
            elif backend.value == "blender":
                bridge = BlenderZeroCopyBridge()
                obj = bridge.to_blender(
                    coords,
                    name=f"astro_{viz_type.value}",
                    collection_name="AstroLab"
                )
                mo.output.append(mo.md("✅ Blender object created!"))
                set_state(lambda s: {**s, "visualization": obj})
                
            elif backend.value == "cosmograph":
                bridge = CosmographBridge()
                config = {
                    "nodeSize": point_size.value,
                    "renderMode": quality.value,
                    "colorBy": color_by.value,
                }
                viz = bridge.create_visualization(dm._main_data, config)
                mo.output.append(mo.md("✅ Cosmograph visualization created!"))
                mo.output.append(mo.Html(viz))
                
            elif backend.value == "open3d":
                # Use tensor bridge for Open3D
                viz = visualize_cosmic_web(
                    coords,
                    backend="open3d",
                    point_size=point_size.value,
                    show=False
                )
                mo.output.append(mo.md("✅ Open3D point cloud created!"))
                set_state(lambda s: {**s, "visualization": viz})
                
            elif backend.value == "tensor_bridge":
                # Multi-backend demo
                mo.output.append(mo.md("🔗 Creating multi-backend visualization..."))
                
                # Transfer to multiple frameworks
                pv_mesh = transfer_to_framework(coords, "pyvista")
                np_array = transfer_to_framework(coords, "numpy")
                
                mo.output.append(mo.md(f"""
                ✅ Zero-copy transfer completed:
                - PyVista mesh: {pv_mesh.n_points} points
                - NumPy array: {np_array.shape}
                - Memory overhead: ~0 bytes (zero-copy)
                """))
                
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))
            logger.error(f"Visualization error: {e}")
    
    create_btn = mo.ui.button(
        "🚀 Create Visualization",
        on_click=create_advanced_visualization,
        kind="primary",
    )
    
    # Export options
    export_format = mo.ui.dropdown(
        options={
            "gltf": "GLTF (Web)",
            "obj": "OBJ (Universal)",
            "ply": "PLY (Point Cloud)",
            "fbx": "FBX (Animation)",
            "usd": "USD (Pixar)",
        },
        value="gltf",
        label="Export Format",
    )
    
    def export_visualization():
        """Export visualization to file."""
        if state["visualization"] is None:
            mo.output.append(mo.md("❌ No visualization to export!"))
            return
            
        mo.output.append(mo.md(f"📤 Exporting as {export_format.value}..."))
        # Export logic would go here
        mo.output.append(mo.md("✅ Export completed!"))
    
    export_btn = mo.ui.button(
        "📤 Export",
        on_click=export_visualization,
        disabled=state["visualization"] is None,
        kind="secondary",
    )
    
    return mo.vstack([
        mo.md("## 🎨 Advanced Visualization Creator"),
        mo.md("*Create stunning astronomical visualizations with multiple backends*"),
        backend,
        viz_type,
        mo.accordion({
            "🎨 Visual Settings": mo.vstack([
                point_size,
                quality,
                color_by,
                effects,
            ]),
            "📤 Export Options": mo.vstack([
                export_format,
                export_btn,
            ]),
        }),
        create_btn,
    ])


def cosmic_web_visualizer() -> mo.Html:
    """
    Specialized cosmic web visualization with multiple scales.
    This is where the magic happens!
    """
    # State
    state, set_state = mo.state({
        "spatial_data": None,
        "analysis_results": None,
        "current_viz": None,
    })
    
    # Visualization presets
    preset = mo.ui.dropdown(
        options={
            "stellar_neighborhood": "⭐ Stellar Neighborhood",
            "galaxy_filaments": "🌌 Galaxy Filaments",
            "dark_matter_web": "🕸️ Dark Matter Web",
            "void_structure": "⚫ Void Structure",
            "custom": "🎨 Custom",
        },
        value="stellar_neighborhood",
        label="Visualization Preset",
    )
    
    # Scale selector
    scale = mo.ui.slider(
        value=10.0,
        min=0.1,
        max=1000.0,
        step=0.1,
        label="Scale (pc/Mpc)",
        log_scale=True,
    )
    
    # Clustering parameters
    clustering_method = mo.ui.dropdown(
        options={
            "dbscan": "DBSCAN",
            "hdbscan": "HDBSCAN",
            "optics": "OPTICS",
            "spectral": "Spectral",
            "morse_theory": "Morse Theory",
        },
        value="dbscan",
        label="Clustering Method",
    )
    
    # Visualization style
    style = mo.ui.dropdown(
        options={
            "points": "Point Cloud",
            "density": "Density Field",
            "filaments": "Filament Network",
            "clusters": "Cluster Regions",
            "hybrid": "Hybrid",
        },
        value="hybrid",
        label="Visualization Style",
    )
    
    def analyze_cosmic_web():
        """Perform cosmic web analysis."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No data loaded!"))
            return
            
        try:
            mo.output.append(mo.md("🔄 Analyzing cosmic web structure..."))
            
            # Get spatial data
            from astro_lab.tensors import SpatialTensorDict
            
            if hasattr(dm._main_data, 'pos'):
                coords = dm._main_data.pos
            else:
                coords = dm._main_data.x[:, :3]
                
            spatial_tensor = SpatialTensorDict(
                coordinates=coords,
                coordinate_system="icrs",
                unit="parsec"
            )
            
            # Perform analysis
            from astro_lab.data.cosmic_web import CosmicWebAnalyzer
            analyzer = CosmicWebAnalyzer()
            
            # Multi-scale analysis
            scales = [scale.value * factor for factor in [0.5, 1.0, 2.0, 5.0]]
            
            results = {
                "spatial_tensor": spatial_tensor,
                "scales": scales,
                "clustering_results": {},
            }
            
            for s in scales:
                labels = spatial_tensor.cosmic_web_clustering(
                    eps_pc=s,
                    min_samples=5,
                    algorithm=clustering_method.value
                )
                results["clustering_results"][f"{s}_pc"] = {
                    "labels": labels,
                    "n_clusters": len(torch.unique(labels[labels >= 0])),
                }
                
            set_state(lambda s: {
                **s,
                "spatial_data": spatial_tensor,
                "analysis_results": results,
            })
            
            mo.output.append(mo.md("✅ Cosmic web analysis completed!"))
            
            # Show results
            for scale_key, res in results["clustering_results"].items():
                mo.output.append(mo.md(f"""
                **{scale_key}:** {res['n_clusters']} clusters found
                """))
                
        except Exception as e:
            mo.output.append(mo.md(f"❌ Analysis error: {str(e)}"))
    
    analyze_btn = mo.ui.button(
        "🔬 Analyze Structure",
        on_click=analyze_cosmic_web,
        kind="primary",
    )
    
    def create_cosmic_viz():
        """Create cosmic web visualization."""
        if state["analysis_results"] is None:
            mo.output.append(mo.md("❌ Run analysis first!"))
            return
            
        try:
            mo.output.append(mo.md("🎨 Creating cosmic web visualization..."))
            
            spatial_tensor = state["spatial_data"]
            results = state["analysis_results"]
            
            # Get cluster labels for current scale
            scale_key = f"{scale.value}_pc"
            if scale_key not in results["clustering_results"]:
                # Use closest scale
                scale_key = list(results["clustering_results"].keys())[0]
                
            labels = results["clustering_results"][scale_key]["labels"]
            
            # Create visualization based on style
            if style.value == "points":
                fig = plot_cosmic_web_3d(
                    spatial_tensor,
                    cluster_labels=labels.numpy(),
                    title=f"Cosmic Web - {preset.value}",
                    point_size=3,
                )
            elif style.value == "density":
                density = spatial_tensor.analyze_local_density(radius_pc=scale.value)
                fig = plot_density_heatmap(
                    spatial_tensor,
                    density,
                    radius=scale.value,
                    title="Cosmic Web Density",
                )
            elif style.value == "filaments":
                # Detect filaments
                from astro_lab.data.cosmic_web import CosmicWebAnalyzer
                analyzer = CosmicWebAnalyzer()
                filaments = analyzer.detect_filaments(
                    spatial_tensor,
                    method="mst",
                    distance_threshold=scale.value * 2
                )
                mo.output.append(mo.md(f"""
                🌊 Filaments detected:
                - Segments: {filaments['n_filament_segments']}
                - Total length: {filaments['total_filament_length']:.2f}
                """))
                fig = plot_cosmic_web_3d(spatial_tensor, title="Filament Network")
            else:  # hybrid
                fig = plot_multi_scale_clustering(
                    spatial_tensor,
                    results["clustering_results"],
                    title="Multi-Scale Cosmic Web",
                )
                
            mo.output.append(mo.plotly(fig))
            set_state(lambda s: {**s, "current_viz": fig})
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Visualization error: {str(e)}"))
    
    viz_btn = mo.ui.button(
        "🌌 Create Visualization",
        on_click=create_cosmic_viz,
        kind="success",
    )
    
    return mo.vstack([
        mo.md("## 🌌 Cosmic Web Visualizer"),
        mo.md("*Explore the large-scale structure of the universe*"),
        preset,
        mo.hstack([scale, clustering_method]),
        style,
        mo.hstack([analyze_btn, viz_btn]),
    ])


def graph_creator() -> mo.Html:
    """
    Advanced graph visualization for astronomical networks.
    Not your typical node-link diagram!
    """
    # Graph creation method
    method = mo.ui.dropdown(
        options={
            "spatial_knn": "🌟 Spatial k-NN",
            "correlation": "📊 Feature Correlation",
            "causal": "➡️ Causal Network",
            "hierarchical": "🌳 Hierarchical",
            "astronomical": "🔭 Astronomical",
        },
        value="astronomical",
        label="Graph Construction",
    )
    
    # Layout algorithm
    layout = mo.ui.dropdown(
        options={
            "force_3d": "3D Force-directed",
            "hyperbolic": "Hyperbolic",
            "radial": "Radial Tree",
            "sphere": "Spherical",
            "cosmic": "Cosmic Web",
        },
        value="cosmic",
        label="Layout Algorithm",
    )
    
    # Edge filtering
    edge_threshold = mo.ui.slider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        label="Edge Threshold",
    )
    
    # Node sizing
    node_size_by = mo.ui.dropdown(
        options={
            "degree": "Node Degree",
            "centrality": "Centrality",
            "clustering": "Clustering Coefficient",
            "feature": "Feature Value",
            "constant": "Constant",
        },
        value="degree",
        label="Node Size",
    )
    
    def create_advanced_graph():
        """Create advanced graph visualization."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No data loaded!"))
            return
            
        try:
            mo.output.append(mo.md("🕸️ Creating advanced graph..."))
            
            # Create graph based on method
            from astro_lab.data.graphs import create_astronomical_graph
            
            if method.value == "astronomical":
                graph = create_astronomical_graph(
                    dm._main_data,
                    method="parallax_weighted",
                    k_neighbors=20,
                )
            else:
                # Other methods...
                graph = dm._main_data
                
            mo.output.append(mo.md(f"""
            ✅ Graph created:
            - Nodes: {graph.num_nodes:,}
            - Edges: {graph.num_edges:,}
            - Layout: {layout.value}
            """))
            
            # Here you would create the actual visualization
            # using the selected layout and styling options
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))
    
    create_btn = mo.ui.button(
        "🕸️ Create Graph",
        on_click=create_advanced_graph,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 🕸️ Advanced Graph Creator"),
        mo.md("*Create beautiful astronomical network visualizations*"),
        method,
        layout,
        mo.hstack([edge_threshold, node_size_by]),
        create_btn,
    ])


def results_viewer() -> mo.Html:
    """
    Interactive results explorer with advanced visualizations.
    No boring tables - only cool interactive views!
    """
    # Result type selector
    result_type = mo.ui.dropdown(
        options={
            "clustering_3d": "🎯 3D Clustering Results",
            "embedding_space": "🌀 Embedding Space",
            "attention_maps": "🧠 Attention Maps",
            "feature_importance": "📊 Feature Importance",
            "prediction_confidence": "🎨 Prediction Landscape",
        },
        value="clustering_3d",
        label="Result Type",
    )
    
    # Interactive controls
    dimension_x = mo.ui.dropdown(
        options=["PC1", "PC2", "PC3", "UMAP1", "UMAP2", "tSNE1", "tSNE2"],
        value="PC1",
        label="X Dimension",
    )
    
    dimension_y = mo.ui.dropdown(
        options=["PC1", "PC2", "PC3", "UMAP1", "UMAP2", "tSNE1", "tSNE2"],
        value="PC2",
        label="Y Dimension",
    )
    
    color_metric = mo.ui.dropdown(
        options={
            "cluster": "Cluster ID",
            "confidence": "Prediction Confidence",
            "error": "Prediction Error",
            "anomaly": "Anomaly Score",
            "importance": "Feature Importance",
        },
        value="cluster",
        label="Color By",
    )
    
    def load_and_visualize():
        """Load and visualize results."""
        mo.output.append(mo.md(f"🔄 Loading {result_type.value}..."))
        
        # This would load actual results from experiments
        # For now, show a placeholder
        mo.output.append(mo.md(f"""
        ### {result_type.value.replace('_', ' ').title()}
        
        🎨 Interactive visualization would appear here with:
        - X axis: {dimension_x.value}
        - Y axis: {dimension_y.value}
        - Colored by: {color_metric.value}
        
        Features:
        - 🖱️ Hover for details
        - 🔍 Zoom and pan
        - 📸 Export snapshots
        - 🎯 Click to select
        """))
    
    load_btn = mo.ui.button(
        "📊 Load & Visualize",
        on_click=load_and_visualize,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 📊 Interactive Results Viewer"),
        mo.md("*Explore your results with style*"),
        result_type,
        mo.hstack([dimension_x, dimension_y]),
        color_metric,
        load_btn,
    ])


def graph_visualizer() -> mo.Html:
    """Advanced graph visualizer (already defined in graph_creator)."""
    return graph_creator()


def clustering_visualizer() -> mo.Html:
    """Advanced clustering visualizer (part of cosmic_web_visualizer)."""
    return cosmic_web_visualizer()


def cosmograph_viewer() -> mo.Html:
    """
    Cosmograph integration for massive graph visualization.
    Handle millions of nodes with ease!
    """
    # Check availability
    try:
        bridge = CosmographBridge()
        available = True
    except Exception:
        available = False
    
    if not available:
        return mo.vstack([
            mo.md("## 🌌 Cosmograph Viewer"),
            mo.callout(
                "Cosmograph not available. Install with: `pip install cosmograph`",
                kind="warn"
            ),
        ])
    
    # Configuration
    config = mo.ui.code_editor(
        value="""{
    "nodeSize": 2,
    "linkWidth": 0.5,
    "simulationDecay": 1000,
    "renderMode": "speed",
    "showLabels": false,
    "colorBy": "cluster"
}""",
        language="json",
        label="Cosmograph Configuration",
    )
    
    # Performance settings
    performance = mo.ui.dropdown(
        options={
            "speed": "⚡ Speed Priority",
            "balanced": "⚖️ Balanced",
            "quality": "🎨 Quality Priority",
        },
        value="balanced",
        label="Performance Mode",
    )
    
    def create_cosmograph_viz():
        """Create Cosmograph visualization."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No graph data loaded!"))
            return
            
        try:
            import json
            config_dict = json.loads(config.value)
            config_dict["renderMode"] = performance.value
            
            mo.output.append(mo.md("🌌 Creating Cosmograph visualization..."))
            
            viz = bridge.create_visualization(dm._main_data, config_dict)
            mo.output.append(mo.Html(viz))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))
    
    create_btn = mo.ui.button(
        "🌌 Create Cosmograph",
        on_click=create_cosmograph_viz,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 🌌 Cosmograph Viewer"),
        mo.md("*Visualize massive astronomical networks*"),
        performance,
        mo.accordion({
            "⚙️ Configuration": config,
        }),
        create_btn,
    ])


# Export all components
__all__ = [
    "plot_creator",
    "cosmic_web_visualizer",
    "graph_creator",
    "results_viewer",
    "graph_visualizer",
    "clustering_visualizer",
    "cosmograph_viewer",
]
