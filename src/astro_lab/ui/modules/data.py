"""
Data UI Module - Direct integration with AstroLab data backend
=============================================================

UI components that directly use AstroLab data classes and methods.
"""

import marimo as mo
from typing import Dict, List, Optional, Any
import polars as pl
from pathlib import Path

# Direct imports from AstroLab
from astro_lab.data import AstroDataModule
from astro_lab.data.loaders import (
    load_catalog,
    load_survey_catalog,
    download_survey,
    list_available_catalogs,
)
from astro_lab.data.datasets import SurveyGraphDataset, SUPPORTED_SURVEYS
from astro_lab.data.config import data_config


def data_explorer() -> mo.Html:
    """Data explorer that uses actual AstroLab data loading."""
    # State for this component
    state, set_state = mo.state({
        "current_catalog": None,
        "dataframe": None,
        "loading": False,
    })
    
    # Survey selector from actual supported surveys
    survey = mo.ui.dropdown(
        options=list(SUPPORTED_SURVEYS.keys()),
        label="Select Survey",
    )
    
    # Max samples for testing
    max_samples = mo.ui.number(
        value=10000,
        min=100,
        max=1000000,
        step=1000,
        label="Max Samples",
    )
    
    def load_data():
        """Load data using AstroLab methods."""
        set_state(lambda s: {**s, "loading": True})
        try:
            df = load_survey_catalog(
                survey=survey.value,
                max_samples=max_samples.value,
                load_processed=True,
            )
            set_state(lambda s: {
                **s, 
                "dataframe": df,
                "current_catalog": survey.value,
                "loading": False,
            })
            mo.output.append(mo.md(f"✅ Loaded {len(df)} objects from {survey.value}"))
        except Exception as e:
            set_state(lambda s: {**s, "loading": False})
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))
    
    # Load button
    load_btn = mo.ui.button(
        "Load Data",
        on_click=load_data,
        disabled=state()["loading"],
    )
    
    # Data preview
    df = state()["dataframe"]
    if df is not None:
        preview = mo.vstack([
            mo.md(f"### Data Preview: {state()['current_catalog']}"),
            mo.md(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns"),
            mo.ui.table(df.head(100).to_pandas()),
        ])
    else:
        preview = mo.md("*No data loaded*")
    
    return mo.vstack([
        mo.md("## 📊 Data Explorer"),
        mo.hstack([survey, max_samples, load_btn]),
        preview,
    ])


def data_loader() -> mo.Html:
    """Data loader that creates AstroDataModule instances."""
    # State
    state, set_state = mo.state({
        "datamodule": None,
        "setup_complete": False,
    })
    
    # Configuration
    survey = mo.ui.dropdown(
        options=list(SUPPORTED_SURVEYS.keys()),
        value="gaia",
        label="Survey",
    )
    
    batch_size = mo.ui.slider(
        value=32,
        min=1,
        max=256,
        step=1,
        label="Batch Size",
    )
    
    k_neighbors = mo.ui.slider(
        value=8,
        min=3,
        max=50,
        step=1,
        label="K-Neighbors",
    )
    
    num_workers = mo.ui.number(
        value=4,
        min=0,
        max=16,
        step=1,
        label="Num Workers",
    )
    
    def create_datamodule():
        """Create AstroDataModule instance."""
        try:
            dm = AstroDataModule(
                survey=survey.value,
                batch_size=batch_size.value,
                k_neighbors=k_neighbors.value,
                num_workers=num_workers.value,
                pin_memory=False,  # Fixed for laptop
            )
            set_state(lambda s: {**s, "datamodule": dm, "setup_complete": False})
            mo.output.append(mo.md("✅ DataModule created"))
        except Exception as e:
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))
    
    def setup_datamodule():
        """Setup the DataModule."""
        dm = state()["datamodule"]
        if dm:
            try:
                dm.setup("fit")
                set_state(lambda s: {**s, "setup_complete": True})
                
                # Show info
                info = dm.get_info()
                mo.output.append(mo.md(f"""
                ✅ DataModule ready!
                - **Num Features:** {info.get('num_features', 'Unknown')}
                - **Num Classes:** {info.get('num_classes', 'Unknown')}
                - **Graph Nodes:** {info.get('num_nodes', 'Unknown')}
                - **Graph Edges:** {info.get('num_edges', 'Unknown')}
                """))
            except Exception as e:
                mo.output.append(mo.md(f"❌ Setup failed: {str(e)}"))
    
    create_btn = mo.ui.button(
        "Create DataModule",
        on_click=create_datamodule,
    )
    
    setup_btn = mo.ui.button(
        "Setup Data",
        on_click=setup_datamodule,
        disabled=state()["datamodule"] is None,
    )
    
    # Info display
    dm = state()["datamodule"]
    if dm and state()["setup_complete"]:
        info_display = mo.md("✅ DataModule is ready for training!")
    elif dm:
        info_display = mo.md("⚠️ DataModule created but not setup. Click 'Setup Data'.")
    else:
        info_display = mo.md("*No DataModule created*")
    
    return mo.vstack([
        mo.md("## 🔄 Data Loader"),
        mo.accordion({
            "Configuration": mo.vstack([
                survey,
                batch_size,
                k_neighbors,
                num_workers,
            ])
        }),
        mo.hstack([create_btn, setup_btn]),
        info_display,
    ])


def catalog_manager() -> mo.Html:
    """Catalog manager using AstroLab catalog functions."""
    # State
    state, set_state = mo.state({
        "catalogs": pl.DataFrame(),
        "selected_catalog": None,
    })
    
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
        "🔄 Refresh",
        on_click=refresh_catalogs,
    )
    
    # Download section
    download_survey_select = mo.ui.dropdown(
        options=["gaia", "sdss", "2mass", "wise", "pan_starrs"],
        label="Download Survey",
    )
    
    download_btn = mo.ui.button(
        "⬇️ Download",
        on_click=lambda: download_catalog(download_survey_select.value),
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
        accept=[".fits", ".parquet", ".csv"],
        on_change=handle_upload,
    )
    
    return mo.vstack([
        mo.md("## 📁 Catalog Manager"),
        mo.hstack([refresh_btn]),
        mo.accordion({
            "📥 Download Data": mo.hstack([download_survey_select, download_btn]),
            "📤 Upload File": file_upload,
        }),
        mo.md("### Available Catalogs"),
        catalog_table,
    ])


# Store DataModule instance for sharing with other modules
_shared_datamodule = None

def get_current_datamodule() -> Optional[AstroDataModule]:
    """Get the current DataModule instance for use in other modules."""
    return _shared_datamodule

def set_current_datamodule(dm: AstroDataModule):
    """Set the current DataModule instance."""
    global _shared_datamodule
    _shared_datamodule = dm
