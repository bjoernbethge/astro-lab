# AstroLab Data Directory

This directory contains astronomical data, processed catalogs, and visualizations for the AstroLab project.

## 📁 Directory Structure

```
data/
├── raw/              # Raw astronomical survey data (downloaded from archives)
├── catalogs/         # Consolidated AstroLab catalogs ready for analysis
├── visualizations/   # Generated visualizations and plots
└── DATA_PIPELINE_GUIDE.md  # Complete guide for data acquisition and processing
```

## 🚀 Quick Start

Download real survey data and build a consolidated catalog:

```bash
# Download raw data for a survey (requires network access)
astro-lab download gaia

# Build consolidated AstroLab catalog from the raw data
PYTHONPATH=src python scripts/generate_astrolab_catalog.py --surveys gaia --max-samples 10000

# Run the full pipeline (download + preprocess + catalog + visualization)
PYTHONPATH=src python scripts/complete_data_pipeline.py --max-samples 10000
```

Open the generated HTML in `data/visualizations/` in your browser to explore the 3D cosmic web.

## 📖 Documentation

For detailed information about:
- Data acquisition and processing pipeline
- Catalog schema and features
- Visualization options
- Extending the pipeline
- Performance tips

See **[DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)** in this directory.

## 📂 Directory Descriptions

### `raw/`
Contains raw astronomical survey data downloaded from various archives:
- **Gaia DR3**: Astrometric and photometric data for stars
- **SDSS**: Optical spectroscopy and photometry for galaxies
- **2MASS**: Near-infrared photometry
- **NASA archives**: Various astronomical catalogs

Files in this directory are typically in their original formats (FITS, CSV, Parquet).

### `catalogs/`
Contains consolidated AstroLab catalogs combining multiple surveys:

#### AstroLab Catalog v1.0
**File**: `astrolab_catalog_v1.parquet`

A comprehensive catalog with:
- **Astrometry**: Position (RA, Dec, distance), proper motion, parallax
- **Photometry**: Multi-wavelength magnitudes (optical + infrared)
- **3D Coordinates**: Cartesian coordinates (x, y, z) in parsecs
- **Derived Properties**: Colors, absolute magnitudes, velocities
- **Metadata**: Version, processing date

**Quick Load**:
```python
import polars as pl
catalog = pl.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")
```

### `visualizations/`
Contains generated visualizations from catalog analysis:
- **3D Scatter Plots**: Interactive HTML plots (Plotly)
- **Structure Maps**: Cosmic web structure visualizations
- **Statistical Plots**: Distribution and correlation plots

All HTML visualizations can be opened directly in a web browser.

## 🔬 Data Quality

The catalog includes quality-filtered data with:
- Valid parallax measurements
- Clean astrometric solutions
- Reliable photometry

Filters applied (when available):
- Parallax SNR > 5 (signal-to-noise ratio for distance measurement)
- RUWE < 1.4 (Renormalized Unit Weight Error - measures astrometric solution quality)
- Astrometric excess noise < 1.0 (indicates clean astrometric measurements)

## 💾 Data Size and Git

Large data files (> 100 MB) are excluded from git tracking via `.gitignore`.

Current file sizes:
- Raw data: ~900 KB
- Catalogs: ~440 KB
- Visualizations: ~4.8 MB

Total: ~6.2 MB (safe for git)

## 🌐 Generating the AstroLab Catalog

Multiple ways to generate catalogs:

### Option 1: Complete Pipeline (Recommended)
```bash
PYTHONPATH=src python scripts/complete_data_pipeline.py --max-samples 10000
```

### Option 2: Standalone Catalog Generation
```bash
PYTHONPATH=src python scripts/generate_astrolab_catalog.py --surveys gaia --max-samples 10000
```

### Option 3: Just Visualizations
```bash
python scripts/generate_visualizations.py --catalog data/catalogs/astrolab_catalog_v1.parquet
```

## 📈 Example Usage

```python
import polars as pl
import plotly.express as px

# Load catalog
catalog = pl.read_parquet("data/catalogs/astrolab_catalog_v1.parquet")

# Get bright, nearby stars
bright_nearby = catalog.filter(
    (pl.col("phot_g_mean_mag") < 10) & 
    (pl.col("distance_pc") < 100)
)

# Plot color-magnitude diagram
fig = px.scatter(
    bright_nearby.to_pandas(),
    x="bp_rp",
    y="mg_abs",
    title="Color-Magnitude Diagram"
)
fig.show()
```

## 🎯 Citing the AstroLab Catalog

If you use the AstroLab catalog in your research, please cite:

```bibtex
@software{astrolab_catalog,
  title = {AstroLab Catalog: Multi-Survey Cosmic Web Dataset},
  author = {AstroLab Team},
  year = {2026},
  url = {https://github.com/synapticore-io/astro-lab},
  version = {1.0}
}
```

## 📚 Data Sources and Acknowledgments

- **Gaia**: ESA Gaia mission (https://www.cosmos.esa.int/gaia)
- **SDSS**: Sloan Digital Sky Survey (https://www.sdss.org/)
- **2MASS**: Two Micron All Sky Survey (https://www.ipac.caltech.edu/2mass/)

Please cite the original surveys when using their data.

## 🐛 Known Issues

- Cosmic web analysis may fail due to tensor indexing issues (catalog still generated)
- Network access required for real data downloads
- Large catalogs (>1M sources) require 16GB+ RAM

See [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md) for details and workarounds.

## 🔧 Support

For help:
1. Read [DATA_PIPELINE_GUIDE.md](DATA_PIPELINE_GUIDE.md)
2. Check the example scripts in `scripts/`
3. Open an issue on GitHub
