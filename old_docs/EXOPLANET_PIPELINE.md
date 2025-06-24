# 🪐 Exoplanet Pipeline Documentation

This document describes the exoplanet data processing pipeline for the AstroLab framework.

## Overview

The exoplanet pipeline processes confirmed exoplanet data from various surveys and provides
cosmic web analysis capabilities for stellar neighborhood studies.

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

## CLI Usage

```bash
# Process exoplanet data with cosmic web analysis
astro-lab preprocess cosmic-web exoplanet --max-samples 5000 --scales 10.0 25.0 50.0 --output results/

# Process with verbose logging
astro-lab preprocess cosmic-web exoplanet --max-samples 5000 --verbose

# Process all surveys including exoplanet
astro-lab preprocess all-surveys --max-samples 500 --output results/
```

## Dependencies

The pipeline requires the following main dependencies:
- numpy
- torch  
- polars
- astropy
- pyvista
- cosmograph

## Usage

```python
import numpy as np
import torch
import polars as pl
import astropy.units as u
from astropy.coordinates import SkyCoord

from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.tensors.spatial_3d import Spatial3DTensor
```

## Dataset: NASA Exoplanet Archive

### Raw Data Properties
```
Total Exoplanets:    5,798 (with distance data)
Distance Range:      1.3 - 8,240 pc
Sky Coverage:        All-sky (multiple surveys)  
Data Size:           ~247 KB parquet
Planet Types:        All confirmed discovery methods
Host Star Types:     Main sequence to giants
```

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

## Multi-Scale Structure Discovery

### Small Scale (10 pc radius) - Stellar associations:
```
Star Groups:         396
Grouped Systems:     42.7%
Isolated Systems:    57.3%
Structure Type:      Local stellar associations
Physical Scale:      Stellar neighborhoods
```

### Medium Scale (25-50 pc radius) - Open clusters:
```
Star Groups:         158-65
Grouped Systems:     73-86%
Isolated Systems:    27-14%
Structure Type:      Open clusters & moving groups
Physical Scale:      Galactic disk structure
```

### Large Scale (100-200 pc radius) - Stellar populations:
```
Star Groups:         31-17
Grouped Systems:     93-98%
Isolated Systems:    7-2%
Structure Type:      Galactic stellar populations
Physical Scale:      Disk-halo structure
```

## Exoplanet Properties

### Planet Size Distribution
```
Earth-size (0.5-1.5 R⊕):     878 planets (20.3%)
Super-Earths (1.5-4 R⊕):    2,383 planets (55.1%)
Mini-Neptunes (4-8 R⊕):     265 planets (6.1%)
Gas Giants (>8 R⊕):         792 planets (18.3%)
```

### Discovery Methods
```
Transit:                    ~4,000 planets (Kepler/TESS)
Radial Velocity:           ~1,000 planets (ground-based)
Microlensing:              245 planets (rare events)
Direct Imaging:            ~20 planets (young giants)
```

## Scientific Significance

### Stellar Neighborhood Mapping
- **Local stellar structure** within 8 kpc revealed through exoplanet hosts
- **Galactic disk** organization visible in host star distribution
- **Stellar associations** identified through spatial clustering
- **Planetary system demographics** across stellar populations

### Exoplanet Demographics
- **Spatial bias** towards nearby bright stars (transit surveys)
- **Metallicity gradient** effects on planet occurrence
- **Stellar age effects** on planetary system architecture
- **Galactic location** correlation with planet properties

### Observational Selection Effects
- **Distance bias** towards nearby systems (<100 pc heavily sampled)
- **Magnitude bias** towards bright host stars
- **Survey bias** towards specific sky regions
- **Detection bias** towards larger/closer-in planets

## Data Products

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

## Usage Examples

### Basic Exoplanet Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

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

## Related Documentation

- **[Cosmic Web Analysis](COSMIC_WEB_ANALYSIS.md)**: General cosmic web analysis guide
- **[Data Loaders](DATA_LOADERS.md)**: Comprehensive data loading guide
- **[Cosmograph Integration](COSMOGRAPH_INTEGRATION.md)**: Interactive visualization