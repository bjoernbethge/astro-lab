# 🌌 AstroLab Examples

Ready-to-run examples demonstrating AstroLab's astronomical data processing capabilities.

## 🚀 Quick Start

### Simple Example
```bash
# Quick start with Gaia stellar data
python examples/quick_start.py
```

### Advanced Example
```bash
# Full demonstration of all features
python examples/modern_data_analysis.py
```

## 📊 Available Examples

### `quick_start.py`
**Perfect for beginners!** Simple example showing basic cosmic web analysis.
- Loads Gaia stellar data
- Performs cosmic web analysis
- Shows clustering results
- Minimal code, maximum understanding

### `modern_data_analysis.py`
**Complete feature demonstration** showcasing all AstroLab capabilities:
- Single survey analysis
- Multi-survey comparison
- Interactive 3D visualization
- Advanced cross-survey analysis
- Data export and saving

### `astroquery_example.py`
**External data integration** using Astroquery:
- Download data from astronomical databases
- Process with AstroLab pipeline
- Cross-match with existing surveys

### `blender_widget_example.py`
**Advanced 3D rendering** with Blender integration:
- Export visualizations to Blender
- Create high-quality renders
- Advanced material and lighting setup

## 🌌 Available Surveys

| Survey | Type | Description | Default Scales |
|--------|------|-------------|----------------|
| **gaia** | Stellar | Gaia DR3 stellar catalog | 1.0, 2.0, 5.0, 10.0 Mpc |
| **sdss** | Galaxy | SDSS DR17 galaxies | 5.0, 10.0, 20.0, 50.0 Mpc |
| **nsa** | Galaxy | NASA Sloan Atlas | 5.0, 10.0, 20.0, 50.0 Mpc |
| **linear** | Solar System | LINEAR asteroids | 5.0, 10.0, 20.0, 50.0 Mpc |
| **tng50** | Simulation | TNG50 cosmological simulation | 5.0, 10.0, 20.0, 50.0 Mpc |
| **exoplanet** | Planetary | NASA Exoplanet Archive with Gaia crossmatching | 10.0, 25.0, 50.0, 100.0 Mpc |

## 🎨 Visualization Examples

### Interactive 3D Visualization
```python
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

# Load data
results = create_cosmic_web_loader(survey="gaia", max_samples=200)

# Create interactive visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0,
    background_color='#000011',
    point_color='#ffd700',
    physics_enabled=True
)
```

### Survey-Specific Colors
```python
color_map = {
    'gaia': '#ffd700',      # Gold for stars
    'sdss': '#4a90e2',      # Blue for galaxies
    'nsa': '#e24a4a',       # Red for NSA
    'tng50': '#00ff00',     # Green for simulation
    'linear': '#ff8800',    # Orange for asteroids
    'exoplanet': '#ff00ff'  # Magenta for exoplanets
}
```

## 📈 Analysis Examples

### Single Survey Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

# Analyze Gaia stellar data
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[1.0, 2.0, 5.0, 10.0]
)

print(f"Found {results['n_objects']:,} objects")
print(f"Volume: {results['total_volume']:.0f} Mpc³")
print(f"Density: {results['global_density']:.2e} obj/Mpc³")
```

### Multi-Survey Comparison
```python
# Compare different surveys
surveys = ["gaia", "nsa", "exoplanet"]
results = {}

for survey in surveys:
    results[survey] = create_cosmic_web_loader(
        survey=survey,
        max_samples=500,
        scales_mpc=[5.0, 10.0]
    )
    print(f"{survey}: {results[survey]['n_objects']} objects")
```

## 💾 Data Export

### Save Results
```python
import torch
from pathlib import Path

# Process survey
results = create_cosmic_web_loader(survey="gaia", max_samples=100)

# Save coordinates
coords_tensor = torch.tensor(results["coordinates"])
torch.save(coords_tensor, "gaia_coordinates.pt")

# Save summary
with open("gaia_summary.txt", "w") as f:
    f.write(f"Objects: {results['n_objects']:,}\n")
    f.write(f"Volume: {results['total_volume']:.0f} Mpc³\n")
```

## 📁 Example Structure

```
examples/
├── quick_start.py              # Beginner-friendly introduction
├── modern_data_analysis.py     # Complete feature demonstration
├── astroquery_example.py       # External data integration
├── blender_widget_example.py   # Advanced 3D rendering
├── data_loading/               # Data processing examples
│   └── data_processing_example.py
├── training/                   # Model training examples
│   └── hyperparameter_optimization.py  # Hyperparameter optimization with AstroTrainer
├── visualization/              # Visualization examples
│   ├── cosmograph_polars_example.py
│   └── polars_data_example.py
└── advanced/                   # Advanced usage examples
```

## 🚀 Quick CLI Reference

For detailed CLI documentation, see the [main README](../README.md#-cli-reference).

```bash
# Process data (required before running examples)
astro-lab process --surveys gaia --max-samples 1000

# Run examples
python examples/quick_start.py
python examples/modern_data_analysis.py
```