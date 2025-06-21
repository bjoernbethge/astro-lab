# 🪐 Exoplanet Cosmic Web Analysis

Complete 3D cosmic web analysis of confirmed exoplanet host stars using advanced spatial clustering techniques and interactive visualization.

## Overview

The exoplanet cosmic web analysis processes **5,798 confirmed exoplanet systems** with 3D stellar coordinates, revealing the distribution of planetary systems across the local stellar neighborhood up to 8,200 pc distance.

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

## Usage

### CLI Processing
```bash
# Process exoplanet data with cosmic web analysis
astro-lab preprocess cosmic-web exoplanet --max-samples 5000 --scales 10.0 25.0 50.0 --output results/

# Process with verbose logging
astro-lab preprocess cosmic-web exoplanet --max-samples 5000 --verbose

# Process all surveys including exoplanet
astro-lab preprocess all-surveys --max-samples 500 --output results/
```

### Results
- Multi-scale clustering: 10-200 pc hierarchy
- Stellar associations to stellar neighborhoods
- Exoplanet host star distribution mapped
- Planetary system properties analyzed
- Interactive 3D visualization with CosmographBridge

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

### Advanced Analysis
```python
from astro_lab.tensors import Spatial3DTensor
import torch
import numpy as np

# Create spatial tensor from results
coords_3d = torch.tensor(results['coordinates'], dtype=torch.float32)
spatial_tensor = Spatial3DTensor(coords_3d, unit='pc')

# Multi-scale stellar clustering
for scale in [10, 25, 50, 100, 200]:  # pc
    cluster_results = spatial_tensor.cosmic_web_clustering(
        eps_pc=scale, min_samples=3
    )
    print(f"{scale} pc: {cluster_results['n_clusters']} stellar groups")
    
# Analyze planetary properties by stellar group
props = torch.load('results/exoplanet_cosmic_web/exoplanet_properties.pt')
planet_radii = props['planet_radius']
valid_radii = planet_radii[~np.isnan(planet_radii)]

print(f"Planet size statistics:")
print(f"  Median: {np.median(valid_radii):.2f} R⊕")
print(f"  Earth-size: {np.sum((valid_radii >= 0.8) & (valid_radii <= 1.25)):,}")
```

### Advanced Host Star Analysis
```python
# Analyze host star properties by spatial groups
distances = props['distance_pc']
discovery_years = props['discovery_year']

# Distance-limited samples
nearby_mask = distances < 100  # pc
distant_mask = distances > 100

print(f"Nearby systems (<100 pc): {nearby_mask.sum():,}")
print(f"Distant systems (>100 pc): {distant_mask.sum():,}")

# Discovery timeline analysis
if discovery_years is not None:
    valid_years = discovery_years[~np.isnan(discovery_years)]
    print(f"Discovery peak: {np.median(valid_years):.0f}")
    print(f"Recent discoveries (>2015): {np.sum(valid_years > 2015):,}")
```

### Multi-Survey Comparison
```python
# Compare exoplanet distribution with stellar surveys
surveys = ["exoplanet", "gaia", "sdss"]
comparison = {}

for survey in surveys:
    comparison[survey] = create_cosmic_web_loader(
        survey=survey,
        max_samples=1000,
        scales_mpc=[25.0]
    )
    print(f"{survey}: {comparison[survey]['n_objects']} objects, "
          f"{comparison[survey]['n_clusters']} clusters")

# Visualize comparison
bridge = CosmographBridge()
widgets = []

for survey in surveys:
    widget = bridge.from_cosmic_web_results(
        comparison[survey],
        survey_name=survey,
        radius=1.5
    )
    widgets.append(widget)
```

## 🎨 Visualization Features

### CosmographBridge Integration
- **Magenta color scheme** for exoplanet data
- **Real-time physics simulation** with gravity and repulsion
- **Interactive 3D navigation** with click and drag
- **Hover information** for individual planetary systems
- **Cluster highlighting** for stellar associations

### Survey-Specific Colors
```python
color_map = {
    'exoplanet': '#ff00ff',  # Magenta for exoplanets
    'gaia': '#ffd700',       # Gold for stars
    'sdss': '#4a90e2',       # Blue for galaxies
    'nsa': '#e24a4a',        # Red for NSA
    'tng50': '#00ff00',      # Green for simulation
    'linear': '#ff8800'      # Orange for asteroids
}
```

## Comparison with Other Surveys

### Exoplanet vs Stellar Surveys
```
Survey           Objects    Distance   Density      Scale
Gaia DR3         3M stars  <1 kpc     3k/pc³       Galactic disk
Exoplanets       6k hosts  <8 kpc     4e-8/pc³     Planetary systems
NSA galaxies     640k gal  <640 Mpc   4e-4/Mpc³    Cosmic web
```

### Key Differences
- **Exoplanets**: Highly biased towards nearby bright stars
- **Gaia**: Complete stellar census of local neighborhood
- **SDSS**: Extragalactic survey with redshift information
- **NSA**: Galaxy catalog with distance measurements

## 🔬 Advanced Analysis

### Planetary System Demographics
```python
# Analyze planet properties by spatial location
from astro_lab.tensors.spatial_3d import analyze_spatial_properties

properties = {
    'planet_radius': props['planet_radius'],
    'orbital_period': props['orbital_period'],
    'host_star_mass': props['host_star_mass']
}

spatial_analysis = analyze_spatial_properties(
    coordinates=results['coordinates'],
    properties=properties,
    clusters=results['clusters']
)

print("Planetary system demographics by spatial location:")
for cluster_id, cluster_data in spatial_analysis.items():
    print(f"Cluster {cluster_id}: {cluster_data['n_systems']} systems")
    print(f"  Mean planet radius: {cluster_data['mean_radius']:.2f} R⊕")
    print(f"  Mean orbital period: {cluster_data['mean_period']:.1f} days")
```

### Discovery Bias Analysis
```python
# Analyze discovery bias in exoplanet surveys
discovery_methods = props['discovery_method']
distances = props['distance_pc']

# Group by discovery method
methods = np.unique(discovery_methods)
for method in methods:
    method_mask = discovery_methods == method
    method_distances = distances[method_mask]
    print(f"{method}: {method_mask.sum()} planets, "
          f"mean distance: {np.mean(method_distances):.0f} pc")
```

## 📊 Export and Sharing

### Save Results
```python
import json

# Save exoplanet cosmic web results
with open('exoplanet_cosmic_web_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save visualization settings
viz_config = {
    'survey_name': 'exoplanet',
    'radius': 2.0,
    'background_color': '#000011',
    'point_color': '#ff00ff'
}

with open('exoplanet_viz_config.json', 'w') as f:
    json.dump(viz_config, f, indent=2)
```

### Load and Continue
```python
# Load saved results
with open('exoplanet_cosmic_web_results.json', 'r') as f:
    results = json.load(f)

# Create visualization from saved results
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name='exoplanet',
    radius=2.0
)
```

## 🌟 Key Insights

1. **Spatial Distribution**: Exoplanet host stars reveal stellar associations and galactic structure
2. **Discovery Bias**: Strong bias towards nearby bright stars in current surveys
3. **Multi-scale Structure**: Planetary systems cluster at multiple scales from 10-200 pc
4. **Interactive Exploration**: CosmographBridge enables 3D exploration of planetary system distribution
5. **Comparative Analysis**: Framework supports cross-survey analysis with stellar and galactic data

This analysis provides unique insight into the spatial distribution of planetary systems and their relationship to stellar structure in our galactic neighborhood! 🪐✨ 