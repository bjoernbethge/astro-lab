"""
AstroLab Settings UI Components
==============================

Modern settings and configuration UI components for AstroLab.
"""

import marimo as mo
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


def ui_experiment_settings() -> mo.Html:
    """Experiment configuration settings."""
    
    # Basic experiment settings
    name_input = mo.ui.text(
        value="stellar_classification_v1",
        label="Experiment Name",
        placeholder="my_experiment"
    )
    
    description = mo.ui.text_area(
        value="Classify stellar objects using Gaia DR3 photometry and astrometry",
        label="Description",
        placeholder="Describe your experiment..."
    )
    
    environment = mo.ui.dropdown(
        options=["development", "staging", "production"],
        value="development",
        label="Environment"
    )
    
    mlflow_enabled = mo.ui.checkbox(
        value=True,
        label="Enable MLflow Tracking"
    )
    
    auto_checkpoint = mo.ui.checkbox(
        value=True,
        label="Auto-save Checkpoints"
    )
    
    # Advanced settings
    max_epochs = mo.ui.number(
        value=100,
        label="Max Epochs"
    )
    
    early_stopping = mo.ui.number(
        value=3,
        label="Early Stopping Patience"
    )
    
    validation_split = mo.ui.slider(
        start=0,
        stop=1,
        value=0.1,
        step=0.01,
        label="Validation Split"
    )
    
    precision = mo.ui.dropdown(
        options=["16-mixed", "32-true", "bf16-mixed"],
        value="16-mixed",
        label="Training Precision"
    )
    
    advanced_settings = mo.accordion({
        "🔧 Advanced Settings": mo.vstack([
            max_epochs,
            early_stopping,
            validation_split,
            precision,
        ])
    })
    
    return mo.vstack([
        mo.md("### 🧪 Experiment Configuration"),
        name_input,
        description,
        environment,
        mlflow_enabled,
        auto_checkpoint,
        advanced_settings,
        mo.hstack([
            mo.ui.button("💾 Save Configuration"),
            mo.ui.button("📤 Export YAML"),
            mo.ui.button("🔄 Reset"),
        ])
    ])


def ui_data_settings() -> mo.Html:
    """Data configuration settings."""
    
    # Data paths
    data_root = mo.ui.text(
        value=str(Path.home() / "astrolab" / "data"),
        label="Data Root Directory"
    )
    
    cache_dir = mo.ui.text(
        value=str(Path.home() / "astrolab" / "cache"),
        label="Cache Directory"
    )
    
    processed_path = mo.ui.text(
        value="data/processed",
        label="Processed Data Path"
    )
    
    raw_path = mo.ui.text(
        value="data/raw",
        label="Raw Data Path"
    )
    
    # Processing settings
    sample_size = mo.ui.slider(
        start=1000,
        stop=10000000,
        value=100000,
        step=1000,
        label="Default Sample Size"
    )
    
    preprocessing = mo.ui.multiselect(
        options=["normalize", "standardize", "log_transform", "remove_outliers"],
        value=["normalize", "remove_outliers"],
        label="Preprocessing Steps"
    )
    
    outlier_threshold = mo.ui.slider(
        start=0,
        stop=100,
        value=5,
        label="Outlier Threshold (%)"
    )
    
    enable_cache = mo.ui.checkbox(
        value=True,
        label="Enable Data Caching"
    )
    
    # Survey tabs
    gaia_settings = mo.vstack([
        mo.ui.dropdown(
            options=["DR1", "DR2", "DR3", "DR4"],
            value="DR3",
            label="Data Release"
        ),
        mo.ui.multiselect(
            options=["astrometry", "photometry", "spectroscopy", "variability"],
            value=["astrometry", "photometry"],
            label="Data Products"
        ),
    ])
    
    sdss_settings = mo.vstack([
        mo.ui.dropdown(
            options=["DR16", "DR17", "DR18"],
            value="DR18",
            label="Data Release"
        ),
        mo.ui.multiselect(
            options=["photometry", "spectroscopy", "imaging"],
            value=["photometry", "spectroscopy"],
            label="Data Types"
        ),
    ])
    
    survey_tabs = mo.tabs({
        "Gaia": gaia_settings,
        "SDSS": sdss_settings,
    })
    
    return mo.vstack([
        mo.md("### 📊 Data Configuration"),
        mo.accordion({
            "📁 Data Paths": mo.vstack([data_root, cache_dir, processed_path, raw_path]),
            "⚙️ Processing": mo.vstack([sample_size, preprocessing, outlier_threshold, enable_cache]),
            "🔭 Survey Settings": survey_tabs,
        }),
        mo.hstack([
            mo.ui.button("💾 Save Settings"),
            mo.ui.button("🔍 Validate Paths"),
        ])
    ])


def ui_model_settings() -> mo.Html:
    """Model configuration settings."""
    
    # Model presets
    presets = mo.ui.dropdown(
        options=["small_fast", "balanced", "large_accurate", "custom"],
        value="balanced",
        label="Model Preset"
    )
    
    # Architecture settings
    architecture = mo.ui.dropdown(
        options=["gnn", "transformer", "cnn", "hybrid"],
        value="gnn",
        label="Base Architecture"
    )
    
    num_layers = mo.ui.slider(
        start=2,
        stop=16,
        value=4,
        label="Number of Layers"
    )
    
    hidden_dim = mo.ui.dropdown(
        options=["64", "128", "256", "512", "1024"],
        value="256",
        label="Hidden Dimension"
    )
    
    dropout = mo.ui.slider(
        start=0,
        stop=0.5,
        value=0.1,
        step=0.05,
        label="Dropout Rate"
    )
    
    # Training settings
    optimizer = mo.ui.dropdown(
        options=["adamw", "adam", "sgd", "rmsprop"],
        value="adamw",
        label="Optimizer"
    )
    
    learning_rate = mo.ui.number(
        value=0.001,
        label="Learning Rate"
    )
    
    weight_decay = mo.ui.number(
        value=0.01,
        label="Weight Decay"
    )
    
    scheduler = mo.ui.dropdown(
        options=["cosine", "step", "exponential", "onecycle", "none"],
        value="cosine",
        label="LR Scheduler"
    )
    
    batch_size = mo.ui.slider(
        start=8,
        stop=512,
        value=64,
        step=8,
        label="Batch Size"
    )
    
    return mo.vstack([
        mo.md("### 🤖 Model Configuration"),
        presets,
        mo.accordion({
            "🏗️ Architecture": mo.vstack([architecture, num_layers, hidden_dim, dropout]),
            "🎯 Training": mo.vstack([optimizer, learning_rate, weight_decay, scheduler, batch_size]),
        }),
        mo.hstack([
            mo.ui.button("💾 Save Configuration"),
            mo.ui.button("📊 Estimate Memory"),
            mo.ui.button("🔍 Validate Config"),
        ])
    ])


def ui_visualization_settings() -> mo.Html:
    """Visualization configuration settings."""
    
    # General settings
    theme = mo.ui.dropdown(
        options=["dark", "light", "auto"],
        value="dark",
        label="Theme"
    )
    
    colormap = mo.ui.dropdown(
        options=["viridis", "plasma", "inferno", "magma", "cividis"],
        value="viridis",
        label="Default Colormap"
    )
    
    dpi = mo.ui.slider(
        start=10,
        stop=100,
        value=50,
        label="Default DPI"
    )
    
    # Backend settings
    plotly_settings = mo.vstack([
        mo.ui.checkbox(value=True, label="Enable WebGL"),
        mo.ui.slider(start=1000, stop=1000000, value=100000, step=1000, label="Max Points"),
        mo.ui.checkbox(value=True, label="Show Toolbar"),
    ])
    
    matplotlib_settings = mo.vstack([
        mo.ui.dropdown(options=["Agg", "TkAgg", "Qt5Agg"], value="Agg", label="Backend"),
        mo.ui.checkbox(value=False, label="Use LaTeX"),
        mo.ui.slider(start=6, stop=20, value=10, label="Font Size"),
    ])
    
    backend_tabs = mo.tabs({
        "Plotly": plotly_settings,
        "Matplotlib": matplotlib_settings,
    })
    
    # Export settings
    export_format = mo.ui.dropdown(
        options=["png", "svg", "pdf", "html"],
        value="png",
        label="Default Export Format"
    )
    
    export_dpi = mo.ui.slider(
        start=72,
        stop=600,
        value=300,
        step=50,
        label="Export DPI"
    )
    
    include_metadata = mo.ui.checkbox(
        value=True,
        label="Include Metadata"
    )
    
    return mo.vstack([
        mo.md("### 🎨 Visualization Settings"),
        mo.accordion({
            "🎨 Appearance": mo.vstack([theme, colormap, dpi]),
            "⚙️ Backends": backend_tabs,
            "💾 Export": mo.vstack([export_format, export_dpi, include_metadata]),
        }),
        mo.hstack([
            mo.ui.button("💾 Save Settings"),
            mo.ui.button("👁️ Preview Theme"),
        ])
    ])


# Configuration management functions

def save_config(config: Dict[str, Any], filename: str = "astrolab_config.yaml"):
    """Save configuration to file."""
    config_path = Path("configs") / filename
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")
    return str(config_path)


def load_config(filename: str = "astrolab_config.yaml") -> Dict[str, Any]:
    """Load configuration from file."""
    config_path = Path("configs") / filename
    
    if not config_path.exists():
        logger.warning(f"Configuration file {config_path} not found")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def get_config_files() -> List[str]:
    """Get list of available configuration files."""
    config_dir = Path("configs")
    if not config_dir.exists():
        return []
    
    return [f.name for f in config_dir.glob("*.yaml")]


# Export all components
__all__ = [
    "ui_experiment_settings",
    "ui_data_settings",
    "ui_model_settings",
    "ui_visualization_settings",
    "save_config",
    "load_config",
    "get_config_files",
]
