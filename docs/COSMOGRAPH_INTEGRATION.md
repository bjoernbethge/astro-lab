# 🌌 Cosmograph Integration - Interactive Astronomical Graph Visualization

Simple and powerful integration of Cosmograph for interactive graph visualization of astronomical data from all surveys.

## 🚀 Quick Start

```python
from src.astro_lab.data.core import create_cosmic_web_loader
from src.astro_lab.utils.viz import CosmographBridge

# Load real survey data with cosmic web analysis
results = create_cosmic_web_loader(survey="gaia", max_samples=500)

# Create interactive visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")

# Widget is now ready for interaction
```

## 📊 Available Methods

### `CosmographBridge.from_cosmic_web_results()`
Create visualization from cosmic web analysis results:

```python
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,  # From create_cosmic_web_loader
    survey_name="gaia",
    radius=5.0,
    point_color='#ffd700'  # Gold for stars
)
```

### `CosmographBridge.from_spatial_tensor()`
Create visualization from Spatial3DTensor:

```python
bridge = CosmographBridge()
widget = bridge.from_spatial_tensor(
    tensor,
    radius=5.0,
    point_color='#ffd700',
    link_color='#666666'
)
```

### `CosmographBridge.from_survey_data()`
Create visualization from survey data (legacy support):

```python
widget = bridge.from_survey_data(
    data,
    survey_name="gaia",  # gaia, sdss, nsa, tng50, linear, exoplanet
    radius=5.0
)
```

### `CosmographBridge.from_coordinates()`
Create visualization from raw coordinates:

```python
widget = bridge.from_coordinates(
    coords,  # Nx3 array
    radius=5.0
)
```

## 🎨 Survey-Specific Colors

The bridge automatically applies appropriate colors for each survey:

```python
color_map = {
    'gaia': '#ffd700',      # Gold for stars
    'sdss': '#4a90e2',      # Blue for galaxies
    'nsa': '#e24a4a',       # Red for NSA
    'tng50': '#00ff00',     # Green for simulation
    'tng': '#00ff00',       # Green for simulation
    'linear': '#ff8800',    # Orange for asteroids
    'exoplanet': '#ff00ff'  # Magenta for exoplanets
}
```

## 🔧 Convenience Function

```python
from src.astro_lab.utils.viz import create_cosmograph_visualization

# Automatically detects data type and creates visualization
widget = create_cosmograph_visualization(
    data_source,  # Spatial3DTensor, cosmic web results, or coordinates array
    survey_name="gaia",
    radius=5.0
)
```

## ⚙️ Configuration Options

Default settings can be customized:

```python
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=5.0,
    background_color='#000011',
    simulation_gravity=0.1,
    simulation_repulsion=0.2,
    show_labels=True,
    show_top_labels_limit=10,
    curved_links=True,
    curved_link_weight=0.3
)
```

## 🌟 Supported Surveys

The bridge works seamlessly with all available surveys:

- **Gaia DR3**: Stellar data with parallax distances
- **SDSS DR17**: Galaxy survey with redshift data  
- **NSA**: NASA Sloan Atlas galaxy catalog
- **TNG50**: Cosmological simulation data
- **LINEAR**: Asteroid lightcurve data
- **Exoplanet**: Planetary system data

## 📝 Examples

See `examples/simple_cosmograph_demo.py` for complete examples with:

- **Gaia stars**: Gold visualization with stellar clustering
- **SDSS galaxies**: Blue visualization for extragalactic data
- **TNG50 simulation**: Green visualization for cosmological simulation
- **NSA galaxies**: Red visualization for galaxy survey data

## 🎮 Interactive Features

- **Click and drag** to navigate the 3D space
- **Scroll** to zoom in/out
- **Right-click** for simulation control
- **Hover** over points for information
- **Real-time physics simulation** with gravity and repulsion

## 🔗 Integration with Cosmic Web Analysis

The bridge is designed to work directly with the cosmic web analysis pipeline:

```python
# Complete workflow
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0, 20.0]
)

widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0
)
```

This provides a complete pipeline from raw survey data to interactive 3D visualization! 🚀 