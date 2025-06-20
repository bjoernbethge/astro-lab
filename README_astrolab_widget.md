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
pip install polars pyvista astropy
```

Optional for Blender integration:
```bash
# Blender as Python module (advanced)
pip install bpy
```

## 🚀 Quick Start

```python
from src.astro_lab.widgets import AstroLabWidget

# 1. Simple demo with simulated data
widget = AstroLabWidget(num_galaxies=5000)
widget.show()

# 2. Load real data
widget = AstroLabWidget(data_source="path/to/gaia_data.parquet")
widget.analyze_data()
widget.show()

# 3. Add interactive controls
widget.add_interactive_controls()

# 4. Blender export (if available)
widget.export_to_blender("my_universe.blend")

# 5. Survey comparison
from src.astro_lab.widgets.pyvista_bpy_widget import compare_surveys
tng_widget, gaia_widget = compare_surveys()
```

## 🎯 Run Demos

```bash
# 1. AstroLab Widget Demo (interactive)
python -c "from src.astro_lab.widgets import AstroLabWidget; AstroLabWidget().show()"

# 2. Simple 3D scatterplots (all surveys)
python demo_astrolab_widget.py

# 3. Survey comparison
python -c "from src.astro_lab.widgets.pyvista_bpy_widget import compare_surveys; compare_surveys()"
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
- export_to_blender()           # Blender integration
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
- **TNG50**: Cosmological simulation (3D positions)
  - **PartType0**: Gas particles
  - **PartType1**: Dark matter
  - **PartType4**: Stars
  - **PartType5**: Black holes
- **TNG50-Temporal**: Temporal evolution with 11 snapshots
- **Custom**: Parquet/CSV files

## 📊 Example Pipeline

```python
from src.astro_lab.widgets.pyvista_bpy_widget import AstroPipeline

# 1. Generate/load data
pipeline = AstroPipeline(num_galaxies=10000)

# 2. Transform coordinates (Astropy)
coords_3d, sky_coords = pipeline.get_3d_coordinates()

# 3. Create visualization (PyVista)
plotter = pipeline.create_visualization()
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
# Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/astro-lab"

# Or run from the correct directory
cd /path/to/astro-lab
python -c "from src.astro_lab.widgets import AstroLabWidget; AstroLabWidget().show()"
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
pip install polars pyvista astropy matplotlib numpy

# Conda alternative
conda install -c conda-forge polars pyvista astropy
```

## 🔮 Roadmap

- [x] **TNG50 Multi-Particle**: All particle types
- [x] **Temporal Data**: Temporal evolution
- [x] **Survey Comparison**: Comparisons between datasets
- [ ] **Streamlit Integration**: Web-based dashboards
- [ ] **Animation**: Temporal evolution and orbits
- [ ] **VR/AR**: PyVista VR backend
- [ ] **Cloud**: Dask for large datasets
- [ ] **AI**: Automatic classification

## 🤝 Contributing

The widget uses **inheritance** and builds on existing AstroLab components:

- `src/astro_lab/data/core.py` - Data loaders
- `src/astro_lab/widgets/` - Widget framework
- Extend easily instead of reinventing! 🎯

---

*"Simple is better than complex"* - The AstroLab Widget elegantly combines modern libraries without unnecessary complexity. 🌟 