# 🪐 Exoplanet Cosmic Web Analysis

Complete 3D cosmic web analysis of confirmed exoplanet host stars using advanced spatial clustering techniques and interactive visualization.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📊 Dataset Overview](#-dataset-overview)
- [🌌 Multi-Scale Structure](#-multi-scale-structure)
- [🪐 Exoplanet Properties](#-exoplanet-properties)
- [🔬 Scientific Significance](#-scientific-significance)
- [🛠️ CLI Commands](#️-cli-commands)
- [📁 Data Products](#-data-products)
- [💡 Usage Examples](#-usage-examples)
- [📚 Related Documentation](#-related-documentation)

## 🚀 Quick Start

```python
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

# Load and analyze exoplanet cosmic web
results = create_cosmic_web_loader(
    survey="exoplanet",
    max_samples=5000,
    scales_mpc=[10.0, 25.0, 50.0]
)

# Create interactive visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="exoplanet",
    radius=2.0,
    point_color='#ff00ff'  # Magenta for exoplanets
)
```

## 📊 Dataset Overview

### NASA Exoplanet Archive Properties
- **Total Systems**: 5,798 confirmed exoplanet host stars
- **Distance Range**: 1.3 - 8,240 pc from Sun
- **Sky Coverage**: All-sky (multiple surveys)
- **Data Size**: ~247 KB parquet
- **Host Star Types**: Main sequence to giants

### Coordinate Transformation
```python
# RA, Dec, Distance → 3D Stellar Coordinates
ra_rad = np.radians(ra)
dec_rad = np.radians(dec)
distance_pc = sy_dist  # Already in parsecs

# Convert to Cartesian coordinates
x = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
y = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)
z = distance_pc * np.sin(dec_rad)
```

## 🌌 Multi-Scale Structure

### Small Scale (10 pc) - Stellar Associations
- **Star Groups**: 396
- **Grouped Systems**: 42.7%
- **Structure Type**: Local stellar associations
- **Physical Scale**: Stellar neighborhoods

### Medium Scale (25-50 pc) - Open Clusters
- **Star Groups**: 158-65
- **Grouped Systems**: 73-86%
- **Structure Type**: Open clusters & moving groups
- **Physical Scale**: Galactic disk structure

### Large Scale (100-200 pc) - Stellar Populations
- **Star Groups**: 31-17
- **Grouped Systems**: 93-98%
- **Structure Type**: Galactic stellar populations
- **Physical Scale**: Disk-halo structure

## 🪐 Exoplanet Properties

### Planet Size Distribution
| Size Category | Count | Percentage |
|---------------|-------|------------|
| Earth-size (0.5-1.5 R⊕) | 878 | 20.3% |
| Super-Earths (1.5-4 R⊕) | 2,383 | 55.1% |
| Mini-Neptunes (4-8 R⊕) | 265 | 6.1% |
| Gas Giants (>8 R⊕) | 792 | 18.3% |

### Discovery Methods
| Method | Count | Description |
|--------|-------|-------------|
| Transit | ~4,000 | Kepler/TESS surveys |
| Radial Velocity | ~1,000 | Ground-based spectroscopy |
| Microlensing | 245 | Rare gravitational events |
| Direct Imaging | ~20 | Young giant planets |

## 🔬 Scientific Significance

### Stellar Neighborhood Mapping
- **Local stellar structure** within 8 kpc revealed through exoplanet hosts
- **Galactic disk** organization visible in host star distribution
- **Stellar associations** identified through spatial clustering
- **Planetary system demographics** across stellar populations

### Observational Selection Effects
- **Distance bias** towards nearby systems (<100 pc heavily sampled)
- **Magnitude bias** towards bright host stars
- **Survey bias** towards specific sky regions
- **Detection bias** towards larger/closer-in planets

### Exoplanet Demographics
- **Spatial bias** towards nearby bright stars (transit surveys)
- **Metallicity gradient** effects on planet occurrence
- **Stellar age effects** on planetary system architecture
- **Galactic location** correlation with planet properties

## 🛠️ CLI Commands

```bash
# Process exoplanet data with cosmic web analysis
uv run python -m astro_lab.cli preprocess cosmic-web exoplanet --max-samples 5000 --scales 10.0 25.0 50.0 --output results/

# Process with verbose logging
uv run python -m astro_lab.cli preprocess cosmic-web exoplanet --max-samples 5000 --verbose

# Process all surveys including exoplanet
uv run python -m astro_lab.cli preprocess all-surveys --max-samples 500 --output results/
```

## 📁 Data Products

### Output Files: `results/exoplanet_cosmic_web/`
```
exoplanet_coords_3d_pc.pt        # 3D stellar coordinates in pc
exoplanet_properties.pt          # Planet & host star properties
exoplanet_cosmic_web_summary.txt # Multi-scale analysis summary
```

### Data Structure
```python
# Load exoplanet cosmic web results
import torch
coords = torch.load('results/exoplanet_cosmic_web/exoplanet_coords_3d_pc.pt')
props = torch.load('results/exoplanet_cosmic_web/exoplanet_properties.pt')

print(f"Exoplanet systems: {len(coords):,}")      # 5,798
print(f"Coordinates: {coords.shape}")              # [5798, 3]
print(f"Distance range: {coords.max():.0f} pc")    # ~8,240 pc
print(f"Planet names: {len(props['planet_names'])}") # Individual planets
```

## 💡 Usage Examples

### Basic Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

# Load and analyze exoplanet cosmic web
results = create_cosmic_web_loader(
    survey="exoplanet",
    max_samples=5000,
    scales_mpc=[10.0, 25.0, 50.0]
)

print(f"Found {results['n_objects']} exoplanet systems")
print(f"Volume: {results['total_volume']:.0f} Mpc³")
print(f"Clusters: {len(results['clusters'])}")
```

### Interactive Visualization
```python
from astro_lab.utils.viz import CosmographBridge

# Create interactive 3D visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="exoplanet",
    radius=2.0,
    background_color='#000011',
    point_color='#ff00ff'  # Magenta for exoplanets
)
```

### Advanced Analysis
```python
# Analyze distance distribution
import numpy as np
distances = np.linalg.norm(coords, axis=1)
print(f"Median distance: {np.median(distances):.1f} pc")
print(f"90th percentile: {np.percentile(distances, 90):.1f} pc")

# Analyze clustering efficiency by scale
for scale, result in results["results_by_scale"].items():
    efficiency = result['grouped_fraction'] * 100
    print(f"{scale} pc scale: {efficiency:.1f}% grouped")
```

## 📚 Related Documentation

### Core Documentation
- **[Data Loaders](DATA_LOADERS.md)** - Data processing and loading
- **[Cosmic Web Analysis](COSMIC_WEB_ANALYSIS.md)** - Complete analysis framework
- **[Cosmograph Integration](COSMOGRAPH_INTEGRATION.md)** - Interactive visualization

### Survey-Specific Guides
- **[Gaia Cosmic Web](GAIA_COSMIC_WEB.md)** - Stellar structure analysis
- **[SDSS/NSA Analysis](NSA_COSMIC_WEB.md)** - Galaxy survey analysis
- **[Exoplanet Pipeline](EXOPLANET_PIPELINE.md)** - Exoplanet detection workflows

### Main Documentation
- **[Main README](../README.md)** - Complete framework overview
- **[Complete Analysis](../COSMIC_WEB_COMPLETE_ANALYSIS.md)** - Multi-survey comparison

---

**Ready to explore exoplanet distributions?** Start with [Data Loading](DATA_LOADERS.md) or dive into [Cosmic Web Analysis](COSMIC_WEB_ANALYSIS.md)! 