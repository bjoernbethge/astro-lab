"""
Visualization UI Module - Direct integration with AstroLab widgets
=================================================================

UI components that use the actual AstroLab widget system.
"""

import marimo as mo
from typing import Dict, List, Optional, Any
import numpy as np
import torch

# Direct imports from AstroLab widgets
from astro_lab.widgets import (
    AstroLabWidget,
    create_plotly_visualization,
    cluster_and_analyze,
    CosmographBridge,
)

# Import data module to get current data
from .data import get_current_datamodule


def plot_creator() -> mo.Html:
    """Create plots using AstroLabWidget."""
    # State
    state, set_state = mo.state({
        "widget": AstroLabWidget(),
        "current_figure": None,
        "plot_config": {},
    })
    
    # Plot type selector
    plot_type = mo.ui.dropdown(
        options={
            "scatter_3d": "3D Scatter Plot",
            "density": "Density Plot",
            "projection": "2D Projection",
            "graph": "Graph Visualization",
        },
        value="scatter_3d",
        label="Plot Type",
    )
    
    # Plot configuration
    max_points = mo.ui.slider(
        value=10000,
        min=100,
        max=100000,
        step=100,
        label="Max Points",
    )
    
    point_size = mo.ui.slider(
        value=2,
        min=1,
        max=10,
        step=1,
        label="Point Size",
    )
    
    opacity = mo.ui.slider(
        value=0.8,
        min=0.1,
        max=1.0,
        step=0.1,
        label="Opacity",
    )
    
    colorscale = mo.ui.dropdown(
        options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
        value="Viridis",
        label="Color Scale",
    )
    
    def create_plot():
        """Create plot using AstroLabWidget."""
        dm = get_current_datamodule()
        if not dm or not hasattr(dm, '_main_data'):
            mo.output.append(mo.md("❌ No data loaded!"))
            return
        
        try:
            # Get the widget
            widget = state()["widget"]
            
            # Configure plot
            config = {
                "max_points": max_points.value,
                "point_size": point_size.value,
                "opacity": opacity.value,
                "colorscale": colorscale.value.lower(),
                "show": False,  # Don't auto-show, we'll display in UI
                "title": f"AstroLab {plot_type.value.replace('_', ' ').title()}",
            }
            
            # Create visualization using widget
            fig = widget.plot(
                data=dm._main_data,
                plot_type=plot_type.value,
                backend="plotly",
                **config
            )
            
            # Store figure
            set_state(lambda s: {
                **s,
                "current_figure": fig,
                "plot_config": config,
            })
            
            mo.output.append(mo.md("✅ Plot created successfully!"))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error creating plot: {str(e)}"))
    
    create_btn = mo.ui.button(
        "Create Plot",
        on_click=create_plot,
        kind="primary",
    )
    
    # Display current plot
    fig = state()["current_figure"]
    if fig:
        plot_display = mo.Html(
            fig.to_html(include_plotlyjs='cdn', config={'responsive': True})
        )
    else:
        plot_display = mo.md("*No plot created yet*")
    
    return mo.vstack([
        mo.md("## 🎨 Plot Creator"),
        plot_type,
        mo.accordion({
            "⚙️ Plot Settings": mo.vstack([
                max_points,
                mo.hstack([point_size, opacity]),
                colorscale,
            ])
        }),
        create_btn,
        plot_display,
    ])


def graph_creator() -> mo.Html:
    """Create graph visualizations using AstroLabWidget."""
    # State
    state, set_state = mo.state({
        "widget": AstroLabWidget(),
        "graph_data": None,
    })
    
    # Graph creation method
    method = mo.ui.dropdown(
        options={
            "knn": "K-Nearest Neighbors",
            "radius": "Radius-based",
            "astronomical": "Astronomical (parallax-based)",
        },
        value="knn",
        label="Graph Method",
    )
    
    # Method-specific parameters
    k_neighbors = mo.ui.slider(
        value=10,
        min=3,
        max=50,
        step=1,
        label="K-Neighbors",
    )
    
    radius = mo.ui.number(
        value=10.0,
        min=1.0,
        max=100.0,
        step=1.0,
        label="Radius (pc)",
    )
    
    def create_graph():
        """Create graph using AstroLabWidget."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No data loaded!"))
            return
        
        try:
            widget = state()["widget"]
            
            # Create graph
            if method.value == "radius":
                graph = widget.create_graph(
                    data=dm._main_data,
                    method=method.value,
                    radius=radius.value,
                )
            else:
                graph = widget.create_graph(
                    data=dm._main_data,
                    method=method.value,
                    k=k_neighbors.value,
                )
            
            set_state(lambda s: {**s, "graph_data": graph})
            
            mo.output.append(mo.md(f"""
            ✅ Graph created successfully!
            - **Nodes:** {graph.num_nodes:,}
            - **Edges:** {graph.num_edges:,}
            - **Method:** {method.value}
            """))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error creating graph: {str(e)}"))
    
    create_btn = mo.ui.button(
        "Create Graph",
        on_click=create_graph,
        kind="primary",
    )
    
    def visualize_graph():
        """Visualize the created graph."""
        graph = state()["graph_data"]
        if not graph:
            mo.output.append(mo.md("❌ No graph created yet!"))
            return
        
        try:
            widget = state()["widget"]
            
            # Visualize using widget
            fig = widget.plot(
                data=graph,
                plot_type="graph",
                backend="plotly",
                max_points=min(1000, graph.num_nodes),
                show=False,
            )
            
            mo.output.replace(
                mo.Html(fig.to_html(include_plotlyjs='cdn', config={'responsive': True}))
            )
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error visualizing graph: {str(e)}"))
    
    viz_btn = mo.ui.button(
        "Visualize Graph",
        on_click=visualize_graph,
        disabled=state()["graph_data"] is None,
    )
    
    # Parameter display based on method
    if method.value == "radius":
        params = radius
    else:
        params = k_neighbors
    
    return mo.vstack([
        mo.md("## 🕸️ Graph Creator"),
        method,
        params,
        mo.hstack([create_btn, viz_btn]),
    ])


def clustering_visualizer() -> mo.Html:
    """Clustering visualization using AstroLabWidget."""
    # State
    state, set_state = mo.state({
        "widget": AstroLabWidget(),
        "cluster_results": None,
    })
    
    # Clustering algorithm
    algorithm = mo.ui.dropdown(
        options={
            "dbscan": "DBSCAN",
            "kmeans": "K-Means",
            "spectral": "Spectral Clustering",
        },
        value="dbscan",
        label="Algorithm",
    )
    
    # Algorithm parameters
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
    
    n_clusters = mo.ui.slider(
        value=5,
        min=2,
        max=20,
        step=1,
        label="Number of Clusters",
    )
    
    def perform_clustering():
        """Perform clustering using widget."""
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No data loaded!"))
            return
        
        try:
            widget = state()["widget"]
            
            # Perform clustering
            if algorithm.value == "dbscan":
                results = widget.cluster_data(
                    data=dm._main_data,
                    algorithm=algorithm.value,
                    eps=eps.value,
                    min_samples=min_samples.value,
                )
            elif algorithm.value == "kmeans":
                results = widget.cluster_data(
                    data=dm._main_data,
                    algorithm=algorithm.value,
                    n_clusters=n_clusters.value,
                )
            else:
                mo.output.append(mo.md(f"❌ {algorithm.value} not implemented yet"))
                return
            
            set_state(lambda s: {**s, "cluster_results": results})
            
            # Display results
            labels = results.get("labels", [])
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1) if hasattr(labels, '__iter__') else 0
            
            mo.output.append(mo.md(f"""
            ✅ Clustering completed!
            - **Algorithm:** {algorithm.value}
            - **Clusters found:** {n_clusters_found}
            - **Noise points:** {n_noise}
            """))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error in clustering: {str(e)}"))
    
    cluster_btn = mo.ui.button(
        "Perform Clustering",
        on_click=perform_clustering,
        kind="primary",
    )
    
    # Parameter display based on algorithm
    if algorithm.value == "dbscan":
        params = mo.vstack([eps, min_samples])
    elif algorithm.value == "kmeans":
        params = n_clusters
    else:
        params = mo.md("*Select algorithm to see parameters*")
    
    return mo.vstack([
        mo.md("## 🎯 Clustering Visualizer"),
        algorithm,
        params,
        cluster_btn,
    ])


def cosmograph_viewer() -> mo.Html:
    """Cosmograph visualization for large-scale graphs."""
    # Check if Cosmograph is available
    try:
        bridge = CosmographBridge()
        available = True
    except Exception:
        available = False
    
    if not available:
        return mo.vstack([
            mo.md("## 🌌 Cosmograph Viewer"),
            mo.md("*Cosmograph not available. Install with: `pip install cosmograph`*"),
        ])
    
    # State
    state, set_state = mo.state({
        "bridge": bridge,
        "config": {},
    })
    
    # Cosmograph settings
    node_size = mo.ui.slider(
        value=3,
        min=1,
        max=10,
        step=1,
        label="Node Size",
    )
    
    link_width = mo.ui.slider(
        value=1,
        min=0.5,
        max=5,
        step=0.5,
        label="Link Width",
    )
    
    simulation_decay = mo.ui.slider(
        value=1000,
        min=100,
        max=5000,
        step=100,
        label="Simulation Decay",
    )
    
    def create_cosmograph():
        """Create Cosmograph visualization."""
        dm = get_current_datamodule()
        if not dm or not hasattr(dm, '_main_data'):
            mo.output.append(mo.md("❌ No graph data loaded!"))
            return
        
        try:
            config = {
                "nodeSize": node_size.value,
                "linkWidth": link_width.value,
                "simulationDecay": simulation_decay.value,
            }
            
            # Create visualization
            viz = state()["bridge"].create_visualization(dm._main_data, config)
            
            mo.output.append(mo.md("✅ Cosmograph visualization created!"))
            mo.output.append(mo.Html(viz))
            
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error creating Cosmograph: {str(e)}"))
    
    create_btn = mo.ui.button(
        "Create Cosmograph",
        on_click=create_cosmograph,
        kind="primary",
    )
    
    return mo.vstack([
        mo.md("## 🌌 Cosmograph Viewer"),
        mo.md("*Large-scale graph visualization*"),
        node_size,
        link_width,
        simulation_decay,
        create_btn,
    ])
