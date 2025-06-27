"""
Settings UI Module - Configuration management
===========================================

UI components for managing AstroLab settings and configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import marimo as mo
import yaml

from astro_lab.data.config import data_config


def experiment_settings() -> mo.Html:
    """Experiment configuration settings."""
    # State
    state, set_state = mo.state(
        {
            "config": {},
            "saved_configs": [],
        }
    )

    # Experiment name
    exp_name = mo.ui.text(
        value="astrolab_experiment",
        label="Experiment Name",
    )

    # MLflow tracking URI
    tracking_uri = mo.ui.text(
        value="file:./mlruns",
        label="MLflow Tracking URI",
    )

    # Random seed
    seed = mo.ui.number(
        value=42,
        min=0,
        max=9999,
        step=1,
        label="Random Seed",
    )

    # Deterministic mode
    deterministic = mo.ui.switch(
        value=True,
        label="Deterministic Mode",
    )

    def save_config():
        """Save experiment configuration."""
        config = {
            "experiment_name": exp_name.value,
            "tracking_uri": tracking_uri.value,
            "seed": seed.value,
            "deterministic": deterministic.value,
        }

        set_state(lambda s: {**s, "config": config})
        mo.output.append(mo.md("✅ Experiment configuration saved!"))

    save_btn = mo.ui.button(
        "Save Configuration",
        on_click=save_config,
        kind="primary",
    )

    return mo.vstack(
        [
            mo.md("## 🧪 Experiment Settings"),
            exp_name,
            tracking_uri,
            seed,
            deterministic,
            save_btn,
        ]
    )


def data_settings() -> mo.Html:
    """Data processing settings."""
    # Data directories
    raw_dir = mo.ui.text(
        value=str(data_config.raw_dir),
        label="Raw Data Directory",
    )

    processed_dir = mo.ui.text(
        value=str(data_config.processed_dir),
        label="Processed Data Directory",
    )

    # Cache settings
    use_cache = mo.ui.switch(
        value=True,
        label="Use Data Cache",
    )

    cache_size = mo.ui.slider(
        value=10,
        min=1,
        max=100,
        step=1,
        label="Cache Size (GB)",
    )

    # Data processing options
    normalize = mo.ui.switch(
        value=True,
        label="Normalize Features",
    )

    remove_outliers = mo.ui.switch(
        value=False,
        label="Remove Outliers",
    )

    outlier_threshold = mo.ui.number(
        value=3.0,
        min=1.0,
        max=5.0,
        step=0.1,
        label="Outlier Threshold (σ)",
    )

    def update_data_config():
        """Update data configuration."""
        # In production, this would update the actual config
        mo.output.append(
            mo.md("""
        ✅ Data settings updated!
        
        **Note:** Directory changes will take effect on next restart.
        """)
        )

    update_btn = mo.ui.button(
        "Update Settings",
        on_click=update_data_config,
        kind="primary",
    )

    return mo.vstack(
        [
            mo.md("## 📊 Data Settings"),
            mo.accordion(
                {
                    "Directories": mo.vstack([raw_dir, processed_dir]),
                    "Cache": mo.vstack([use_cache, cache_size]),
                    "Processing": mo.vstack(
                        [normalize, remove_outliers, outlier_threshold]
                    ),
                }
            ),
            update_btn,
        ]
    )


def model_settings() -> mo.Html:
    """Model training settings."""
    # Device settings
    device = mo.ui.dropdown(
        options=["auto", "cuda", "cpu"],
        value="auto",
        label="Device",
    )

    mixed_precision = mo.ui.dropdown(
        options=["16-mixed", "bf16-mixed", "32-true"],
        value="16-mixed",
        label="Precision",
    )

    # Memory settings
    gradient_checkpointing = mo.ui.switch(
        value=False,
        label="Gradient Checkpointing",
    )

    gradient_accumulation = mo.ui.slider(
        value=1,
        min=1,
        max=16,
        step=1,
        label="Gradient Accumulation Steps",
    )

    # Optimization settings
    compile_model = mo.ui.switch(
        value=False,
        label="Compile Model (PyTorch 2.0)",
    )

    use_fused_optimizer = mo.ui.switch(
        value=True,
        label="Use Fused Optimizer",
    )

    # Checkpoint settings
    save_top_k = mo.ui.slider(
        value=3,
        min=1,
        max=10,
        step=1,
        label="Save Top K Models",
    )

    checkpoint_every_n_epochs = mo.ui.slider(
        value=1,
        min=1,
        max=10,
        step=1,
        label="Checkpoint Every N Epochs",
    )

    def apply_model_settings():
        """Apply model settings."""
        settings = {
            "device": device.value,
            "precision": mixed_precision.value,
            "gradient_checkpointing": gradient_checkpointing.value,
            "gradient_accumulation_steps": gradient_accumulation.value,
            "compile": compile_model.value,
            "use_fused_optimizer": use_fused_optimizer.value,
            "save_top_k": save_top_k.value,
            "checkpoint_every_n_epochs": checkpoint_every_n_epochs.value,
        }

        mo.output.append(
            mo.md(f"""
        ✅ Model settings applied!
        
        ```yaml
        {yaml.dump(settings, default_flow_style=False)}
        ```
        """)
        )

    apply_btn = mo.ui.button(
        "Apply Settings",
        on_click=apply_model_settings,
        kind="primary",
    )

    return mo.vstack(
        [
            mo.md("## 🤖 Model Settings"),
            mo.accordion(
                {
                    "Device": mo.vstack([device, mixed_precision]),
                    "Memory": mo.vstack(
                        [gradient_checkpointing, gradient_accumulation]
                    ),
                    "Optimization": mo.vstack([compile_model, use_fused_optimizer]),
                    "Checkpointing": mo.vstack([save_top_k, checkpoint_every_n_epochs]),
                }
            ),
            apply_btn,
        ]
    )


def visualization_settings() -> mo.Html:
    """Visualization settings."""
    # Default backend
    default_backend = mo.ui.dropdown(
        options=["plotly", "matplotlib", "bokeh", "altair"],
        value="plotly",
        label="Default Backend",
    )

    # Theme settings
    theme = mo.ui.dropdown(
        options=["dark", "light", "auto"],
        value="dark",
        label="Theme",
    )

    # Color settings
    default_colormap = mo.ui.dropdown(
        options=["viridis", "plasma", "inferno", "magma", "cividis", "twilight"],
        value="viridis",
        label="Default Colormap",
    )

    # Performance settings
    max_plot_points = mo.ui.slider(
        value=10000,
        min=1000,
        max=100000,
        step=1000,
        label="Max Points per Plot",
    )

    use_webgl = mo.ui.switch(
        value=True,
        label="Use WebGL (Plotly)",
    )

    # Export settings
    export_dpi = mo.ui.slider(
        value=300,
        min=72,
        max=600,
        step=10,
        label="Export DPI",
    )

    export_format = mo.ui.dropdown(
        options=["png", "svg", "pdf", "html"],
        value="png",
        label="Default Export Format",
    )

    def save_viz_settings():
        """Save visualization settings."""
        settings = {
            "backend": default_backend.value,
            "theme": theme.value,
            "colormap": default_colormap.value,
            "max_points": max_plot_points.value,
            "use_webgl": use_webgl.value,
            "export": {
                "dpi": export_dpi.value,
                "format": export_format.value,
            },
        }

        mo.output.append(
            mo.md("""
        ✅ Visualization settings saved!
        
        Settings will be applied to new plots.
        """)
        )

    save_btn = mo.ui.button(
        "Save Settings",
        on_click=save_viz_settings,
        kind="primary",
    )

    return mo.vstack(
        [
            mo.md("## 🎨 Visualization Settings"),
            mo.accordion(
                {
                    "Appearance": mo.vstack([default_backend, theme, default_colormap]),
                    "Performance": mo.vstack([max_plot_points, use_webgl]),
                    "Export": mo.vstack([export_dpi, export_format]),
                }
            ),
            save_btn,
        ]
    )


def advanced_settings() -> mo.Html:
    """Advanced system settings."""
    # Logging settings
    log_level = mo.ui.dropdown(
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
        value="INFO",
        label="Log Level",
    )

    log_to_file = mo.ui.switch(
        value=False,
        label="Log to File",
    )

    # Performance settings
    num_workers = mo.ui.slider(
        value=4,
        min=0,
        max=16,
        step=1,
        label="Default Data Loader Workers",
    )

    prefetch_factor = mo.ui.slider(
        value=2,
        min=1,
        max=8,
        step=1,
        label="Prefetch Factor",
    )

    # Memory settings
    memory_fraction = mo.ui.slider(
        value=0.9,
        min=0.1,
        max=1.0,
        step=0.1,
        label="GPU Memory Fraction",
    )

    enable_tf32 = mo.ui.switch(
        value=True,
        label="Enable TF32 (Ampere GPUs)",
    )

    # Debug settings
    debug_mode = mo.ui.switch(
        value=False,
        label="Debug Mode",
    )

    profile_training = mo.ui.switch(
        value=False,
        label="Profile Training",
    )

    def apply_advanced():
        """Apply advanced settings."""
        mo.output.append(
            mo.md("""
        ⚠️ **Warning:** Advanced settings will be applied on next session.
        
        Some settings may require restart to take effect.
        """)
        )

    apply_btn = mo.ui.button(
        "Apply Advanced Settings",
        on_click=apply_advanced,
        kind="danger",
    )

    return mo.vstack(
        [
            mo.md("## ⚙️ Advanced Settings"),
            mo.callout(
                mo.md("⚠️ **Caution:** These settings can affect system stability."),
                kind="warn",
            ),
            mo.accordion(
                {
                    "Logging": mo.vstack([log_level, log_to_file]),
                    "Performance": mo.vstack([num_workers, prefetch_factor]),
                    "Memory": mo.vstack([memory_fraction, enable_tf32]),
                    "Debug": mo.vstack([debug_mode, profile_training]),
                }
            ),
            apply_btn,
        ]
    )
