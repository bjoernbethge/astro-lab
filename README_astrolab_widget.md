# 🌌 AstroLab Widget

Simple interactive astronomical visualization widget that combines **Polars**, **Astropy** and **PyVista** - without reinventing the wheel!

## ✨ Features

- 🚀 **Inheritance**: Builds on existing AstroLab components
- ⚡ **Polars**: High-performance data processing 
- 🔭 **Astropy**: Scientifically accurate coordinate transformations
- 🎨 **PyVista**: Interactive 3D visualization
- 🎛️ **Interactive Widgets**: Real-time parameter control
- 📦 **Blender Integration**: Export for advanced rendering (optional)
- 🕒 **Temporal Data**: TNG50 with temporal evolution
- 🔬 **Multi-Particle**: All TNG50 particle types (Gas, DM, Stars, BH)

## 📦 Installation

```bash
# Install from the main project
cd astro-lab
uv sync

# Install required dependencies
uv pip install polars pyvista astropy
```

Optional for Blender integration:
```bash
# Blender as Python module (advanced)
uv pip install bpy
```

## 🚀 Quick Start

### Process Data First
```bash
# Process all surveys (recommended)
uv run python -m astro_lab.cli process

# Or process specific surveys
uv run python -m astro_lab.cli process --surveys gaia tng50
```

### Use the Widget
```python
from astro_lab.widgets import AstroLabWidget

# 1. Simple demo with simulated data
widget = AstroLabWidget(num_galaxies=5000)
widget.show()

# 2. Load real data
widget = AstroLabWidget(data_source="data/processed/gaia/processed/gaia_k8_n1000.pt")
widget.analyze_data()
widget.show()

# 3. Add interactive controls
widget.add_interactive_controls()

# 4. Blender export (if available)
widget.export_blender_scene("my_universe.blend")

# 5. Survey comparison
from astro_lab.widgets import compare_surveys
tng_widget, gaia_widget = compare_surveys()
```

## 🎯 Run Demos

```bash
# 1. AstroLab Widget Demo (interactive)
uv run python -c "from astro_lab.widgets import AstroLabWidget; AstroLabWidget().show()"

# 2. Simple 3D scatterplots (all surveys)
uv run python demo_astrolab_widget.py

# 3. Survey comparison
uv run python -c "from astro_lab.widgets import compare_surveys; compare_surveys()"
```

## 🏗️ Architecture

### Inheritance instead of reinvention

```python
AstroPipeline                    # Base pipeline (Polars + Astropy + PyVista)
    ↓
AstroLabWidget(AstroPipeline)   # Extended with AstroLab integration
    ↓
- load_real_data()              # Uses existing AstroLab loaders
- _tensor_to_polars()           # Converts AstroLab tensors
- add_interactive_controls()    # PyVista widgets
- export_blender_scene()        # Blender integration
- analyze_data()                # Polars-based analysis
```

### Data flow

```
📂 AstroLab Data     →  🔄 Polars DataFrame  →  🔭 Astropy Coords  →  🎨 PyVista 3D
   (Gaia/SDSS/TNG50)      (Feature Engineering)     (RA/Dec → x,y,z)      (Interactive)
```

## 🛠️ Supported Data Sources

- **Gaia**: Stellar data with parallax → redshift estimation
- **SDSS**: Galaxy survey with photometry
- **NSA**: NASA Sloan Atlas galaxies
- **LINEAR**: Asteroid light curves
- **TNG50**: Cosmological simulation (3D positions)
  - **PartType0**: Gas particles
  - **PartType1**: Dark matter
  - **PartType4**: Stars
  - **PartType5**: Black holes
- **Exoplanet**: NASA Exoplanet Archive with Gaia crossmatching
- **Custom**: Parquet/CSV files

## 📊 Example Pipeline

```python
from astro_lab.widgets import AstroLabWidget

# 1. Generate/load data
widget = AstroLabWidget(num_galaxies=10000)

# 2. Transform coordinates (Astropy)
coords_3d, sky_coords = widget.get_3d_coordinates()

# 3. Create visualization (PyVista)
plotter = widget.create_visualization()
plotter.show()
```

## 🎛️ Interactive Features

- **Point Size Slider**: Size of data points
- **Opacity Slider**: Transparency
- **Mouse Controls**: 
  - Left mouse button: Rotate
  - Middle mouse button: Zoom  
  - Right mouse button: Pan

## 📈 Performance

- **Polars**: 10-100x faster than Pandas for large datasets
- **PyVista**: GPU-accelerated rendering
- **Sampling**: Automatic data reduction for performance
- **Memory**: Lazy-loading and efficient data structures

## 🔬 Scientific Accuracy

- **Astropy Cosmology**: Planck18 cosmology for redshift → distance
- **Coordinate Systems**: ICRS, Galactic, Cartesian
- **Units**: Automatic unit conversion (Mpc, ckpc/h, etc.)
- **Stellar Mass**: Empirical color-mass relations

## 🐛 Troubleshooting

### Import errors
```bash
# Check if astro-lab is installed
uv run python -c "import astro_lab; print('astro-lab installed successfully')"

# Or install in development mode
uv pip install -e .
```

### Performance issues
```python
# Fewer points for better performance
widget = AstroLabWidget(num_galaxies=1000)

# Or sampling with real data
widget.load_real_data("large_catalog.parquet") 
# Automatic sampling to 10k objects
```

### Missing dependencies
```bash
# Install all at once
uv pip install polars pyvista astropy matplotlib numpy

# Or use the project's dependencies
uv sync
```

## 🔮 Roadmap

- [x] **TNG50 Multi-Particle**: All particle types
- [x] **Temporal Data**: Temporal evolution
- [x] **Survey Comparison**: Comparisons between datasets
- [x] **CLI Integration**: Easy data processing
- [ ] **Streamlit Integration**: Web-based dashboards
- [ ] **Animation**: Temporal evolution and orbits
- [ ] **VR/AR**: PyVista VR backend
- [ ] **Cloud**: Dask for large datasets
- [ ] **AI**: Automatic classification

## 🤝 Contributing

See the [Development Guide](../docs/DEVGUIDE.md) for contribution guidelines.

## 📚 Related Documentation

- **[AstroViz](../astro-viz/README.md)** - Specialized visualization package
- **[Examples](../examples/README.md)** - Ready-to-run examples
- **[Main Documentation](../README.md)** - Complete framework overview 