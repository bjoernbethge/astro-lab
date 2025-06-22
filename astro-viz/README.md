# 🌌 AstroViz

AstroViz is a specialized package for astronomical visualization within the AstroLab ecosystem, providing interactive 3D visualization, Cosmograph integration, and Blender rendering support.

## ✨ Features

- **🌌 Interactive 3D Visualization**: Real-time cosmic web exploration
- **🎨 Survey-Specific Colors**: Gold for stars, blue for galaxies, green for simulations
- **⚡ Real-time Physics**: Gravity and repulsion simulation
- **🎬 Blender Integration**: Advanced 3D rendering capabilities
- **🔗 CosmographBridge**: Seamless integration with cosmic web analysis
- **📊 Multi-Scale Visualization**: From stellar to cosmological scales

## 📦 Installation

```bash
# Install from the main project
cd astro-lab
uv sync

# Or install directly
uv pip install -e ./astro-viz
```

## 🚀 Quick Start

### Basic Visualization
```python
from astro_viz import CosmographBridge
from astro_lab.data.core import create_cosmic_web_loader

# Load data
results = create_cosmic_web_loader(survey="gaia", max_samples=500)

# Create interactive visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results, 
    survey_name="gaia",
    radius=3.0,
    background_color='#000011',
    point_color='#ffd700'
)
```

### Blender Rendering
```python
from astro_viz import BlenderRenderer

# Render survey data
renderer = BlenderRenderer()
renderer.render_survey("gaia", output_path="renders/gaia.png")

# Custom rendering
renderer.render_custom(
    coordinates=results["coordinates"],
    colors=results["colors"],
    output_path="custom_render.png"
)
```

## 🎨 Visualization Components

### CosmographBridge
Seamless integration with cosmic web analysis:
- **Real-time updates**: Dynamic visualization as data changes
- **Physics simulation**: Gravity and repulsion effects
- **Survey-specific styling**: Automatic color schemes
- **Interactive controls**: Zoom, pan, rotate

### BlenderRenderer
Advanced 3D rendering capabilities:
- **High-quality output**: Professional-grade renders
- **Custom materials**: Advanced lighting and materials
- **Animation support**: Temporal evolution visualization
- **Batch processing**: Multiple survey rendering

### Survey Color Schemes
```python
color_map = {
    'gaia': '#ffd700',      # Gold for stars
    'sdss': '#4a90e2',      # Blue for galaxies
    'nsa': '#e24a4a',       # Red for NSA galaxies
    'tng50': '#00ff00',     # Green for simulation
    'linear': '#ff8800',    # Orange for asteroids
    'exoplanet': '#ff00ff'  # Magenta for exoplanets
}
```

## 🔧 Advanced Usage

### Custom Visualization
```python
from astro_viz import CosmographBridge

bridge = CosmographBridge()

# Custom parameters
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=5.0,
    background_color='#000000',
    point_color='#ffd700',
    physics_enabled=True,
    gravity_strength=0.1,
    repulsion_strength=0.05
)
```

### Multi-Survey Comparison
```python
# Compare different surveys
surveys = ["gaia", "nsa", "tng50"]
widgets = {}

for survey in surveys:
    results = create_cosmic_web_loader(survey=survey, max_samples=200)
    bridge = CosmographBridge()
    widgets[survey] = bridge.from_cosmic_web_results(
        results, 
        survey_name=survey
    )
```

### Animation and Temporal Data
```python
# For TNG50 temporal evolution
from astro_viz import TemporalVisualizer

visualizer = TemporalVisualizer()
animation = visualizer.create_temporal_animation(
    data_path="data/tng50_temporal/",
    output_path="animations/tng50_evolution.mp4"
)
```

## 🎯 Use Cases

### Stellar Structure Analysis
- **Gaia DR3**: Visualize stellar distributions and proper motions
- **Clustering**: Identify stellar associations and streams
- **Density mapping**: Show stellar density variations

### Galaxy Survey Visualization
- **SDSS/NSA**: Large-scale structure analysis
- **Redshift space**: 3D galaxy distribution
- **Morphology**: Galaxy shape and orientation

### Cosmological Simulations
- **TNG50**: Dark matter and gas distribution
- **Temporal evolution**: Universe evolution over time
- **Multi-particle**: Gas, stars, dark matter, black holes

### Exoplanet Analysis
- **Spatial distribution**: Exoplanet locations in 3D
- **Host star properties**: Stellar-exoplanet relationships
- **Detection methods**: Visualization by discovery technique

## 🔧 Development

This package is part of the larger AstroLab framework and follows the same development practices.

### Dependencies
- **Cosmograph**: Interactive graph visualization
- **Blender** (optional): Advanced 3D rendering
- **PyVista**: 3D visualization backend
- **NumPy/PyTorch**: Data processing

### Testing
```bash
# Run visualization tests
uv run pytest test/utils/test_visualization_utils.py
```

## 📚 Related Documentation

- **[Cosmograph Integration](../docs/COSMOGRAPH_INTEGRATION.md)** - Detailed guide to interactive visualization
- **[AstroLab Widget](../README_astrolab_widget.md)** - Interactive widget with Polars and PyVista
- **[Main Documentation](../README.md)** - Complete AstroLab framework overview
