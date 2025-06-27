"""
Data UI Module - Direct integration with AstroLab data backend
=============================================================

UI components that directly use AstroLab data classes and methods with
integrated visualization and analysis capabilities using native AstroLab widgets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import marimo as mo
import numpy as np
import polars as pl
import torch

# Direct imports from AstroLab
from astro_lab.data import AstroDataModule
from astro_lab.data.config import data_config
from astro_lab.data.cosmic_web import CosmicWebAnalyzer
from astro_lab.data.datasets import SUPPORTED_SURVEYS, SurveyGraphDataset
from astro_lab.data.loaders import (
    download_survey,
    list_available_catalogs,
    load_catalog,
    load_survey_catalog,
)
from astro_lab.tensors import SpatialTensorDict

# Import AstroLab visualization systems (native widgets)
from astro_lab.widgets import AstroLabWidget
from astro_lab.widgets.cosmograph_bridge import CosmographBridge
from astro_lab.widgets.tensor_bridge import visualize_cosmic_web

_current_datamodule = None


def set_current_datamodule(dm: Optional[AstroDataModule]):
    """Set the current DataModule for other modules to use."""
    global _current_datamodule
    _current_datamodule = dm


def get_current_datamodule() -> Optional[AstroDataModule]:
    """Get the current DataModule."""
    return _current_datamodule


def data_loader() -> mo.Html:
    """Data loader with integrated visualization and analysis using real AstroLab components."""
    # Enhanced state with real data
    state, set_state = mo.state(
        {
            "datamodule": None,
            "setup_complete": False,
            "spatial_tensor": None,
            "analysis_results": None,
            "current_viz": None,
            "survey_data": None,
            "auto_load": True,  # Automatische Datenladung
        }
    )

    # Real survey configuration
    survey = mo.ui.dropdown(
        options=SUPPORTED_SURVEYS,  # Echte unterstützte Surveys
        value="gaia",
        label="Survey",
    )

    # Data loading parameters
    max_samples = mo.ui.number(
        value=10000,
        start=100,
        stop=1000000,
        step=1000,
        label="Max Samples",
    )

    batch_size = mo.ui.slider(
        value=32,
        start=1,
        stop=256,
        step=1,
        label="Batch Size",
    )

    k_neighbors = mo.ui.slider(
        value=8,
        start=3,
        stop=50,
        step=1,
        label="K-Neighbors",
    )

    num_workers = mo.ui.number(
        value=4,
        start=0,
        stop=16,
        step=1,
        label="Num Workers",
    )

    # Auto-loading toggle
    auto_load_toggle = mo.ui.checkbox(
        value=True,
        label="Auto-load data on survey change",
    )

    def load_survey_data(survey_name: str, max_samples_val: int = 10000):
        """Load real survey data using AstroLab loaders."""
        try:
            mo.output.append(
                mo.md(
                    f"📥 Loading {survey_name.upper()} data ({max_samples_val:,} samples)..."
                )
            )

            # Use real AstroLab data loaders
            catalog_path = load_survey_catalog(
                survey=survey_name,
                max_samples=max_samples_val,
                cache_dir=Path("data/cache"),
            )

            if not catalog_path.exists():
                # Try to download if not available
                mo.output.append(mo.md(f"📦 Downloading {survey_name} catalog..."))
                download_survey(survey_name, max_samples=max_samples_val)
                catalog_path = load_survey_catalog(
                    survey_name, max_samples=max_samples_val
                )

            # Load the actual catalog data
            survey_data = load_catalog(catalog_path)

            new_state = state().copy()
            new_state["survey_data"] = survey_data
            set_state(new_state)

            mo.output.append(
                mo.md(f"✅ Loaded {len(survey_data):,} {survey_name.upper()} objects")
            )

            # Show data info
            if hasattr(survey_data, "columns"):
                mo.output.append(
                    mo.md(
                        f"📊 **Columns:** {', '.join(survey_data.columns[:10])}{'...' if len(survey_data.columns) > 10 else ''}"
                    )
                )

            return survey_data

        except Exception as e:
            mo.output.append(mo.md(f"❌ Failed to load {survey_name} data: {str(e)}"))
            return None

    def create_datamodule():
        """Create AstroDataModule from loaded survey data."""
        survey_data = state()["survey_data"]
        if survey_data is None:
            mo.output.append(mo.md("❌ No survey data loaded! Load data first."))
            return

        try:
            survey_name = survey.value if hasattr(survey, "value") else "gaia"
            batch_size_val = (
                int(batch_size.value) if hasattr(batch_size, "value") else 32
            )
            k_neighbors_val = (
                int(k_neighbors.value) if hasattr(k_neighbors, "value") else 8
            )
            num_workers_val = (
                int(num_workers.value) if hasattr(num_workers, "value") else 4
            )

            # Create real AstroDataModule
            dm = AstroDataModule(
                survey=survey_name,
                batch_size=batch_size_val,
                k_neighbors=k_neighbors_val,
                num_workers=num_workers_val,
                pin_memory=False,
                max_samples=int(max_samples.value)
                if hasattr(max_samples, "value")
                else 10000,
            )

            new_state = state().copy()
            new_state.update({"datamodule": dm, "setup_complete": False})
            set_state(new_state)
            set_current_datamodule(dm)

            mo.output.append(mo.md("✅ AstroDataModule created from survey data"))

        except Exception as e:
            mo.output.append(mo.md(f"❌ DataModule creation failed: {str(e)}"))

    def setup_datamodule():
        """Setup the DataModule and create spatial tensor."""
        dm = state()["datamodule"]
        if dm is None:
            mo.output.append(mo.md("❌ No DataModule! Create one first."))
            return

        try:
            mo.output.append(mo.md("⚙️ Setting up DataModule..."))
            dm.setup("fit")

            # Extract spatial coordinates from real data
            spatial_tensor = None
            if hasattr(dm, "_main_data") and dm._main_data is not None:
                if hasattr(dm._main_data, "pos"):
                    coords = dm._main_data.pos
                elif hasattr(dm._main_data, "x") and dm._main_data.x.shape[-1] >= 3:
                    coords = dm._main_data.x[:, :3]
                else:
                    coords = None

                if coords is not None:
                    # Create real SpatialTensorDict
                    spatial_tensor = SpatialTensorDict(
                        coordinates=coords, coordinate_system="icrs", unit="parsec"
                    )

            new_state = state().copy()
            new_state.update({"setup_complete": True, "spatial_tensor": spatial_tensor})
            set_state(new_state)

            # Show real info from DataModule
            info = dm.get_info()
            survey_name = survey.value if hasattr(survey, "value") else "unknown"

            mo.output.append(
                mo.md(f"""
            ✅ **DataModule ready!**
            - **Survey:** {survey_name.upper()}
            - **Samples:** {info.get("num_samples", "Unknown")}
            - **Features:** {info.get("num_features", "Unknown")}
            - **Classes:** {info.get("num_classes", "Unknown")}
            - **Graph Nodes:** {info.get("num_nodes", "Unknown")}
            - **Graph Edges:** {info.get("num_edges", "Unknown")}
            - **Spatial Data:** {"✅ Available" if spatial_tensor else "❌ Not found"}
            """)
            )

        except Exception as e:
            mo.output.append(mo.md(f"❌ Setup failed: {str(e)}"))

    # === VISUALIZATION SECTION (using real AstroLab widgets) ===

    viz_backend = mo.ui.dropdown(
        options={
            "astrolab": "🌟 AstroLab Native (Recommended)",
            "open3d": "🎯 Open3D Point Clouds",
            "cosmograph": "🌌 Cosmograph Physics",
            "pyvista": "🔷 PyVista 3D",
        },
        value="astrolab",
        label="Visualization Backend",
    )

    point_size = mo.ui.slider(
        value=2.0,
        start=0.1,
        stop=10.0,
        step=0.1,
        label="Point Size",
    )

    def visualize_with_astrolab():
        """Visualize using native AstroLabWidget."""
        spatial_tensor = state()["spatial_tensor"]
        if spatial_tensor is None:
            mo.output.append(mo.md("❌ No spatial data! Setup DataModule first."))
            return

        try:
            backend = viz_backend.value if hasattr(viz_backend, "value") else "astrolab"
            size = point_size.value if hasattr(point_size, "value") else 2.0
            survey_name = survey.value if hasattr(survey, "value") else "gaia"

            mo.output.append(
                mo.md(
                    f"🎨 Creating {backend} visualization for {survey_name.upper()}..."
                )
            )

            if backend == "astrolab":
                # Use native AstroLabWidget
                widget = AstroLabWidget()

                # Create visualization using AstroLab widget
                viz = widget.plot(
                    spatial_tensor,
                    plot_type="scatter_3d",
                    backend="plotly",  # Internal backend for AstroLabWidget
                    title=f"{survey_name.upper()} Survey Data",
                    point_size=int(size),
                    color_by="density",
                    max_points=50000,
                )

                # Display the result (AstroLabWidget returns plotly figures)
                if viz is not None:
                    mo.output.append(
                        viz
                    )  # AstroLabWidget should return displayable object
                    mo.output.append(mo.md("✅ AstroLab native visualization created!"))
                else:
                    mo.output.append(
                        mo.md("⚠️ Visualization created but not displayable in web UI")
                    )

            elif backend == "open3d":
                viz = visualize_cosmic_web(
                    spatial_tensor,
                    backend="open3d",
                    point_size=size,
                    show=True,
                    window_name=f"AstroLab - {survey_name.upper()} Data",
                )
                mo.output.append(mo.md("✅ Open3D visualization opened!"))

            elif backend == "cosmograph":
                bridge = CosmographBridge()
                widget = bridge.from_spatial_tensor(
                    spatial_tensor,
                    radius=10.0,
                    point_color="#ffd700" if survey_name == "gaia" else "#4a90e2",
                )
                mo.output.append(mo.Html(str(widget)))
                mo.output.append(mo.md("✅ Cosmograph visualization created!"))

            elif backend == "pyvista":
                viz = visualize_cosmic_web(
                    spatial_tensor, backend="pyvista", point_size=size, show=False
                )
                mo.output.append(mo.md("✅ PyVista mesh created!"))

        except Exception as e:
            mo.output.append(mo.md(f"❌ Visualization error: {str(e)}"))

    # === COSMIC WEB ANALYSIS ===

    clustering_scales = mo.ui.text(
        value="5, 10, 25, 50",
        label="Clustering Scales (parsec/Mpc)",
    )

    min_samples = mo.ui.slider(
        value=5,
        start=2,
        stop=20,
        step=1,
        label="Min Samples (DBSCAN)",
    )

    def analyze_cosmic_web_structure():
        """Perform real cosmic web analysis using CosmicWebAnalyzer."""
        spatial_tensor = state()["spatial_tensor"]
        if spatial_tensor is None:
            mo.output.append(mo.md("❌ No spatial data! Setup DataModule first."))
            return

        try:
            mo.output.append(mo.md("🔬 **Analyzing cosmic web structure...**"))

            # Parse scales
            scales_text = (
                clustering_scales.value
                if hasattr(clustering_scales, "value")
                else "5, 10, 25, 50"
            )
            scales = [float(s.strip()) for s in scales_text.split(",")]
            min_samples_val = (
                int(min_samples.value) if hasattr(min_samples, "value") else 5
            )

            # Use real CosmicWebAnalyzer
            analyzer = CosmicWebAnalyzer()

            # Perform real multi-scale analysis
            results = {
                "spatial_tensor": spatial_tensor,
                "clustering_results": {},
                "scales": scales,
                "metadata": {
                    "survey": survey.value if hasattr(survey, "value") else "gaia",
                    "min_samples": min_samples_val,
                    "algorithm": "dbscan",
                },
            }

            for scale in scales:
                mo.output.append(mo.md(f"  📊 Clustering at {scale} scale..."))

                # Real cosmic web clustering
                labels = spatial_tensor.cosmic_web_clustering(
                    eps_pc=scale, min_samples=min_samples_val, algorithm="dbscan"
                )

                # Real analysis
                unique_labels = torch.unique(labels)
                n_clusters = len(unique_labels[unique_labels >= 0])
                n_noise = torch.sum(labels == -1).item()
                n_grouped = len(labels) - n_noise

                results["clustering_results"][f"{scale}_pc"] = {
                    "labels": labels,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "n_grouped": n_grouped,
                    "grouped_fraction": n_grouped / len(labels),
                }

                mo.output.append(
                    mo.md(
                        f"    ✅ {n_clusters} clusters, {n_grouped:,} grouped ({n_grouped / len(labels):.1%})"
                    )
                )

            new_state = state().copy()
            new_state["analysis_results"] = results
            set_state(new_state)

            mo.output.append(mo.md("✅ **Cosmic web analysis completed!**"))

        except Exception as e:
            mo.output.append(mo.md(f"❌ Analysis error: {str(e)}"))

    def visualize_analysis_results():
        """Visualize analysis results with AstroLab native tools."""
        analysis_results = state()["analysis_results"]
        if analysis_results is None:
            mo.output.append(mo.md("❌ No analysis results! Run analysis first."))
            return

        try:
            mo.output.append(mo.md("🎨 **Creating analysis visualization...**"))

            spatial_tensor = analysis_results["spatial_tensor"]
            first_scale = list(analysis_results["clustering_results"].keys())[0]
            cluster_data = analysis_results["clustering_results"][first_scale]
            labels = cluster_data["labels"]

            backend = viz_backend.value if hasattr(viz_backend, "value") else "astrolab"
            survey_name = analysis_results["metadata"]["survey"]

            if backend == "astrolab":
                # Use AstroLabWidget for cluster visualization
                widget = AstroLabWidget()

                # Create clustered visualization
                viz = widget.plot(
                    spatial_tensor,
                    plot_type="scatter_3d",
                    cluster_labels=labels.numpy(),
                    title=f"Cosmic Web Clustering - {survey_name.upper()} - {first_scale}",
                    point_size=int(point_size.value)
                    if hasattr(point_size, "value")
                    else 2,
                    show_clusters=True,
                    backend="plotly",
                )

                if viz is not None:
                    mo.output.append(viz)
                    mo.output.append(
                        mo.md("✅ **Cluster analysis visualization created!**")
                    )

            elif backend == "open3d":
                viz = visualize_cosmic_web(
                    spatial_tensor,
                    cluster_labels=labels.numpy(),
                    backend="open3d",
                    show=True,
                    window_name=f"Cosmic Web Analysis - {survey_name.upper()}",
                )
                mo.output.append(mo.md("✅ **Open3D cluster visualization opened!**"))

        except Exception as e:
            mo.output.append(mo.md(f"❌ Visualization error: {str(e)}"))

    # Auto-loading when survey changes
    def on_survey_change():
        """Automatically load data when survey selection changes."""
        if auto_load_toggle.value:
            survey_name = survey.value if hasattr(survey, "value") else "gaia"
            max_samples_val = (
                int(max_samples.value) if hasattr(max_samples, "value") else 10000
            )
            load_survey_data(survey_name, max_samples_val)

    # Buttons with proper callbacks
    load_data_btn = mo.ui.button(
        label="📥 Load Survey Data",
        on_click=lambda _: load_survey_data(
            survey.value if hasattr(survey, "value") else "gaia",
            int(max_samples.value) if hasattr(max_samples, "value") else 10000,
        ),
        kind="success",
    )

    create_dm_btn = mo.ui.button(
        label="🔧 Create DataModule",
        on_click=lambda _: create_datamodule(),
    )

    setup_btn = mo.ui.button(
        label="⚙️ Setup DataModule",
        on_click=lambda _: setup_datamodule(),
    )

    visualize_btn = mo.ui.button(
        label="🎨 Visualize Data",
        on_click=lambda _: visualize_with_astrolab(),
        kind="success",
    )

    analyze_btn = mo.ui.button(
        label="🔬 Analyze Cosmic Web",
        on_click=lambda _: analyze_cosmic_web_structure(),
        kind="success",
    )

    viz_analysis_btn = mo.ui.button(
        label="🌌 Visualize Analysis",
        on_click=lambda _: visualize_analysis_results(),
        kind="success",
    )

    # Status displays
    survey_data_available = state()["survey_data"] is not None
    dm = state()["datamodule"]
    spatial_available = state()["spatial_tensor"] is not None
    analysis_available = state()["analysis_results"] is not None

    # Status info
    if analysis_available:
        status_info = mo.md("🌟 **Ready for advanced analysis!**")
    elif spatial_available:
        status_info = mo.md("✅ **Data ready for visualization and analysis!**")
    elif dm and state()["setup_complete"]:
        status_info = mo.md("⚙️ **DataModule ready, extracting spatial data...**")
    elif dm:
        status_info = mo.md(
            "⚠️ **DataModule created but not setup. Click 'Setup DataModule'.**"
        )
    elif survey_data_available:
        status_info = mo.md("📊 **Survey data loaded. Create DataModule to continue.**")
    else:
        status_info = mo.md("📥 **Load survey data to begin.**")

    return mo.vstack(
        [
            mo.md("## 🔄 AstroLab Data Loader & Cosmic Web Analysis"),
            mo.md(
                "*Load real astronomical survey data and analyze with native AstroLab tools*"
            ),
            # Configuration
            mo.accordion(
                {
                    "📊 Survey & Data Configuration": mo.vstack(
                        [
                            mo.md(
                                f"**Available Surveys:** {', '.join(SUPPORTED_SURVEYS)}"
                            ),
                            mo.hstack([survey, max_samples]),
                            mo.hstack([batch_size, k_neighbors, num_workers]),
                            auto_load_toggle,
                        ]
                    ),
                    "🎨 Visualization Settings": mo.vstack(
                        [
                            viz_backend,
                            point_size,
                        ]
                    ),
                    "🔬 Analysis Parameters": mo.vstack(
                        [
                            clustering_scales,
                            min_samples,
                        ]
                    ),
                }
            ),
            # Data loading section
            mo.vstack(
                [
                    mo.md("### 📥 Data Loading"),
                    mo.hstack([load_data_btn, create_dm_btn, setup_btn]),
                    status_info,
                ]
            ),
            # Analysis & Visualization (only when ready)
            mo.vstack(
                [
                    mo.md("### 🎯 Visualization & Analysis"),
                    mo.hstack([visualize_btn, analyze_btn, viz_analysis_btn])
                    if spatial_available
                    else mo.md("*Setup DataModule first*"),
                    mo.md(
                        f"📈 **Analysis Status:** {'✅ Results available' if analysis_available else '⏳ Run analysis first'}"
                    )
                    if spatial_available
                    else mo.md(""),
                ]
            )
            if dm and state()["setup_complete"]
            else mo.md(""),
        ]
    )


def catalog_manager() -> mo.Html:
    """Catalog manager using AstroLab catalog functions."""
    # State
    state, set_state = mo.state(
        {
            "catalogs": pl.DataFrame(),
            "selected_catalog": None,
        }
    )

    def refresh_catalogs():
        """Refresh catalog list."""
        try:
            catalogs_df = list_available_catalogs()
            set_state(lambda s: {**s, "catalogs": catalogs_df})
            mo.output.append(mo.md(f"✅ Found {len(catalogs_df)} catalogs"))
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))

    def download_catalog(survey: str):
        """Download survey data."""
        try:
            path = download_survey(survey)
            mo.output.append(mo.md(f"✅ Downloaded to: {path}"))
            refresh_catalogs()  # Refresh list after download
        except Exception as e:
            mo.output.append(mo.md(f"❌ Download failed: {str(e)}"))

    # Refresh button
    refresh_btn = mo.ui.button(
        label="🔄 Refresh", on_click=refresh_catalogs, kind="neutral"
    )

    # Download section
    download_survey_select = mo.ui.dropdown(
        options=["gaia", "sdss", "2mass", "wise", "pan_starrs"],
        label="Download Survey",
    )

    download_btn = mo.ui.button(
        label="📥 Download Survey",
        on_click=lambda: download_catalog(download_survey_select.value),
        kind="primary",
    )

    # Catalog list
    catalogs_df = state()["catalogs"]
    if not catalogs_df.is_empty():
        catalog_table = mo.ui.table(catalogs_df.to_pandas())
    else:
        catalog_table = mo.md("*No catalogs found. Download data to get started.*")

    # File upload
    def handle_upload(file):
        if file:
            try:
                path = Path(data_config.raw_dir) / file.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(file.contents)
                mo.output.append(mo.md(f"✅ Uploaded: {file.name}"))
                refresh_catalogs()
            except Exception as e:
                mo.output.append(mo.md(f"❌ Upload failed: {str(e)}"))

    file_upload = mo.ui.file(
        label="Upload FITS/Parquet/CSV",
        on_change=handle_upload,
    )

    return mo.vstack(
        [
            mo.md("## 📁 Catalog Manager"),
            mo.hstack([refresh_btn]),
            mo.accordion(
                {
                    "📥 Download Data": mo.hstack(
                        [download_survey_select, download_btn]
                    ),
                    "📤 Upload File": file_upload,
                }
            ),
            mo.md("### Available Catalogs"),
            catalog_table,
        ]
    )
