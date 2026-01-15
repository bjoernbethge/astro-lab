"""
Data Loader Component
====================

Real data loading using actual AstroLab preprocessors.
"""

import marimo as mo

from astro_lab.data.collectors.exoplanet import ExoplanetCollector
from astro_lab.data.collectors.gaia import GaiaCollector
from astro_lab.data.preprocessors.exoplanet import ExoplanetPreprocessor
from astro_lab.data.preprocessors.gaia import GaiaPreprocessor


def create_data_loader():
    """Create real data loader interface using actual preprocessors."""

    # Survey selector with real surveys
    survey = mo.ui.dropdown(
        options={
            "gaia": "Gaia DR3 (Stars)",
            "exoplanet": "NASA Exoplanet Archive",
        },
        value="gaia",
        label="Survey",
    )

    # Sample size
    samples = mo.ui.slider(
        start=100, stop=50000, value=5000, step=100, label="Max Samples"
    )

    # Download first if needed
    download_btn = mo.ui.button(label="📥 Download Data", kind="secondary")

    # Load and process button
    load_btn = mo.ui.button(label="🔄 Load & Process Data", kind="primary")

    # Create UI
    ui = mo.vstack(
        [
            mo.md("### 📡 Load Real Astronomical Data"),
            survey,
            samples,
            mo.hstack([download_btn, load_btn]),
        ]
    )

    # Initialize with default content
    status = mo.md("⏳ **Ready to load data** - Select survey and click buttons above")
    preview = mo.md("📊 **Data preview will appear here** after loading")
    loaded_data = None

    # Download data
    if download_btn.value:
        try:
            survey_name = survey.value
            status = mo.callout(f"📥 Downloading {survey_name} data...", kind="info")

            if survey_name == "gaia":
                collector = GaiaCollector()
                files = collector.download(force=False)
                status = mo.callout(f"✅ Downloaded {len(files)} files", kind="success")
            elif survey_name == "exoplanet":
                collector = ExoplanetCollector()
                files = collector.download(force=False)
                status = mo.callout(f"✅ Downloaded {len(files)} files", kind="success")

        except Exception as e:
            status = mo.callout(f"❌ Download error: {str(e)}", kind="danger")

    # Load and process data
    if load_btn.value:
        try:
            survey_name = survey.value
            max_samples = samples.value

            status = mo.callout(f"🔄 Processing {survey_name} data...", kind="info")

            if survey_name == "gaia":
                # Use real Gaia preprocessor
                preprocessor = GaiaPreprocessor(
                    {
                        "magnitude_limit": 15.0,
                        "parallax_snr_min": 5.0,
                        "distance_limit_pc": 1000.0,
                    }
                )

                # Try to load data
                from pathlib import Path

                import polars as pl

                data_file = Path("data/raw/gaia") / "gaia_sample.parquet"
                if data_file.exists():
                    df = pl.read_parquet(data_file).head(max_samples)
                    processed_df = preprocessor.preprocess(df)
                    loaded_data = processed_df

                    status = mo.callout(
                        f"✅ Processed {len(processed_df)} Gaia stars", kind="success"
                    )

                    preview = mo.vstack(
                        [
                            mo.md("### 📊 Data Preview"),
                            mo.ui.table(processed_df.head(10).to_dict(as_series=False)),
                            mo.md(
                                f"**Columns:** {', '.join(processed_df.columns[:10])}{'...' if len(processed_df.columns) > 10 else ''}"
                            ),
                            mo.md(f"**Features:** {len(processed_df.columns)} columns"),
                            mo.md(
                                f"**3D Coordinates:** {'✅' if all(col in processed_df.columns for col in ['x', 'y', 'z']) else '❌'}"
                            ),
                        ]
                    )
                else:
                    status = mo.callout(
                        "❌ No Gaia data found. Please download first.", kind="danger"
                    )

            elif survey_name == "exoplanet":
                # Use real Exoplanet preprocessor
                preprocessor = ExoplanetPreprocessor(
                    {
                        "distance_limit_pc": 500.0,
                        "use_gaia_host_coords": True,
                    }
                )

                from pathlib import Path

                import polars as pl

                data_file = Path("data/raw/exoplanet") / "exoplanet_sample.parquet"
                if data_file.exists():
                    df = pl.read_parquet(data_file).head(max_samples)
                    processed_df = preprocessor.preprocess(df)
                    loaded_data = processed_df

                    status = mo.callout(
                        f"✅ Processed {len(processed_df)} exoplanets", kind="success"
                    )

                    preview = mo.vstack(
                        [
                            mo.md("### 📊 Data Preview"),
                            mo.ui.table(processed_df.head(10).to_dict(as_series=False)),
                            mo.md(
                                f"**Columns:** {', '.join(processed_df.columns[:10])}{'...' if len(processed_df.columns) > 10 else ''}"
                            ),
                            mo.md(f"**Features:** {len(processed_df.columns)} columns"),
                            mo.md(
                                f"**Host Star Coords:** {'✅' if all(col in processed_df.columns for col in ['ra', 'dec']) else '❌'}"
                            ),
                        ]
                    )
                else:
                    status = mo.callout(
                        "❌ No exoplanet data found. Please download first.",
                        kind="danger",
                    )

        except Exception as e:
            status = mo.callout(f"❌ Processing error: {str(e)}", kind="danger")
            preview = mo.md("")

    return ui, status, preview, loaded_data
