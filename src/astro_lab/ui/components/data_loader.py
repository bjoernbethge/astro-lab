"""
Data Loader Component
====================

Simple, functional data loading with Marimo.
"""

import marimo as mo

from astro_lab.data.datamodules.survey import get_survey_datamodule

# from astro_lab.data import load_survey_data


def create_data_loader():
    """Create a simple, working data loader interface using SurveyDataModule."""

    # Survey selector
    survey = mo.ui.dropdown(
        options=["gaia", "sdss", "nsa", "exoplanet", "tng50"],
        value="gaia",
        label="Survey",
    )

    # Sample size
    samples = mo.ui.slider(
        start=100, stop=100000, value=10000, step=100, label="Samples"
    )

    # Sampling strategy
    sampling_strategy = mo.ui.dropdown(
        options=["none", "cluster", "neighbor", "saint"],
        value="none",
        label="Sampling Strategy",
    )

    # Load button
    load_btn = mo.ui.button(label="Load Data")

    # Create UI
    ui = mo.vstack(
        [mo.md("### 📥 Load Data"), survey, samples, sampling_strategy, load_btn]
    )

    # State
    status = mo.md("")
    preview = mo.md("")

    if load_btn.value:
        try:
            # Create and setup DataModule
            batch_size = int(samples.value)
            max_samples = int(samples.value)
            datamodule = get_survey_datamodule(
                survey_name=survey.value,
                batch_size=batch_size,
                max_samples=max_samples,
                sampler_type=sampling_strategy.value
                if sampling_strategy.value != "none"
                else None,
            )
            datamodule.prepare_data()
            datamodule.setup()
            loader = datamodule.train_dataloader()
            batch = next(iter(loader))
            # Show batch size and preview
            if hasattr(batch, "num_nodes"):
                size = batch.num_nodes
            elif hasattr(batch, "shape"):
                size = batch.shape[0]
            elif isinstance(batch, dict) and "x" in batch:
                size = batch["x"].shape[0]
            else:
                size = "unknown"
            status = mo.callout(f"✅ Loaded batch with {size} objects", kind="success")
            # Preview: show first few entries if possible
            if isinstance(batch, dict):
                if "x" in batch:
                    preview_data = batch["x"][:5].cpu().numpy()
                    preview = mo.md(f"**Preview (first 5 x):**\n{preview_data}")
                else:
                    # Show keys and first 5 values for any key
                    preview_lines = []
                    for k, v in batch.items():
                        try:
                            preview_lines.append(f"{k}: {v[:5]}")
                        except Exception:
                            preview_lines.append(f"{k}: {str(v)}")
                    preview = mo.md(
                        "**Preview (first 5 per key):**\n" + "\n".join(preview_lines)
                    )
            elif hasattr(batch, "x"):
                preview_data = batch.x[:5].cpu().numpy()
                preview = mo.md(f"**Preview (first 5 x):**\n{preview_data}")
            else:
                preview = mo.md("No preview available.")
        except Exception as e:
            status = mo.callout(f"❌ Error: {str(e)}", kind="danger")
            preview = mo.md("")

    return ui, status, preview
