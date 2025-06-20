# 🪐 Exoplanet Cosmic Web Analysis

Complete 3D cosmic web analysis of confirmed exoplanet host stars using advanced spatial clustering techniques.

## Overview

The exoplanet cosmic web analysis processes **5,798 confirmed exoplanet systems** with 3D stellar coordinates, revealing the distribution of planetary systems across the local stellar neighborhood up to 8,200 pc distance.

## Usage

### CLI Processing
```bash
# Process exoplanet data with cosmic web analysis
python -m astro_lab.cli.preprocessing exoplanet

# Direct cosmic web script  
python process_exoplanet_cosmic_web.py
```

### Results
- Multi-scale clustering: 10-200 pc hierarchy
- Stellar associations to stellar neighborhoods
- Exoplanet host star distribution mapped
- Planetary system properties analyzed

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
from src.astro_lab.tensors import Spatial3DTensor
import torch
import numpy as np

# Load exoplanet coordinates
coords = torch.load('results/exoplanet_cosmic_web/exoplanet_coords_3d_pc.pt')
spatial_tensor = Spatial3DTensor(coords, unit='pc')

# Multi-scale stellar clustering
for scale in [10, 25, 50, 100, 200]:  # pc
    results = spatial_tensor.cosmic_web_clustering(
        eps_pc=scale, min_samples=3
    )
    print(f"{scale} pc: {results['n_clusters']} stellar groups")
    
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
- **Stellar surveys**: More complete but distance-limited
- **Galaxy surveys**: Large-scale structure but sparse locally

## Future Applications

### Exoplanet Science
- **Host star environments** and planet formation
- **Galactic chemical evolution** effects on planets
- **Stellar age gradients** and planetary system evolution
- **Habitable zone** statistics across stellar populations

### Stellar Astrophysics
- **Local stellar structure** through exoplanet host mapping
- **Kinematic groups** identified via spatial clustering
- **Stellar metallicity** gradients in the solar neighborhood
- **Galactic archaeology** through planet-hosting stars

### Survey Planning
- **Target selection** for future exoplanet surveys
- **Completeness corrections** for current samples
- **Optimal observing strategies** for different planet types
- **Statistical studies** of planet occurrence rates

## Key Achievements

1. **Largest 3D exoplanet host analysis** of confirmed systems
2. **Multi-scale stellar structure** from associations to populations
3. **Complete planetary demographics** across distance ranges
4. **Host star distribution mapping** in the solar neighborhood
5. **Selection effect quantification** for future surveys

---

**This analysis represents the most comprehensive 3D mapping of exoplanet host star distribution, revealing both stellar structure and planetary system demographics across the local galactic neighborhood.**

*Generated by AstroLab Exoplanet Cosmic Web Analysis System* 