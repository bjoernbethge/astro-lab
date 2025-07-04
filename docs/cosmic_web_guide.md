# 🌌 Cosmic Web Analysis Guide

Complete guide to using AstroLab's cosmic web analysis functionality for studying large-scale astronomical structures.

## Overview

The cosmic web module provides tools for analyzing the large-scale structure of the universe across multiple scales:

- **Stellar Scale**: Local galactic disk structure (parsecs)
- **Galactic Scale**: Galaxy clusters and superclusters (Megaparsecs)  
- **Exoplanet Scale**: Stellar neighborhoods and associations

## Quick Start

### 1. Command Line Interface

```bash
# Analyze Gaia stellar data
astro-lab cosmic-web gaia --max-samples 100000 --clustering-scales 5 10 25

# Analyze NSA galaxy data  
astro-lab cosmic-web nsa --clustering-scales 5 10 20 50 --visualize

# Analyze exoplanet host stars
astro-lab cosmic-web exoplanet --clustering-scales 10 25 50 100 200
```

### 2. Python API

```python
from astro_lab.data import analyze_gaia_cosmic_web
from astro_lab.widgets import CosmicWebVisualizer

# Run analysis
results = analyze_gaia_cosmic_web(
    max_samples=100000,
    clustering_scales=[5.0, 10.0, 25.0, 50.0],
)

# Visualize results
viz = CosmicWebVisualizer()
fig = viz.plot_3d_structure(spatial_tensor, cluster_labels)
fig.show()
```

### 3. Tensor Methods

```python
from astro_lab.tensors import SpatialTensorDict

# Create spatial tensor
spatial = SpatialTensorDict(coordinates, unit="parsec")

# Cosmic web clustering
labels = spatial.cosmic_web_clustering(eps_pc=10.0, min_samples=5)

# Local density analysis
density = spatial.analyze_local_density(radius_pc=50.0)

# Grid structure analysis
structure = spatial.cosmic_web_structure(grid_size_pc=100.0)
```

## Features

### Multi-Scale Clustering

Analyze structure at different physical scales:

```python
scales = [5.0, 10.0, 25.0, 50.0]  # parsecs
results = {}

for scale in scales:
    labels = spatial.cosmic_web_clustering(eps_pc=scale)
    results[f"{scale}_pc"] = {
        "n_clusters": len(np.unique(labels[labels >= 0])),
        "labels": labels
    }
```

### Density Analysis

Compute local density around each point:

```python
# Count neighbors within radius
density_counts = spatial.analyze_local_density(radius_pc=50.0)

# Find high-density regions
high_density_mask = density_counts > density_counts.quantile(0.9)
```

### Grid-Based Structure

Analyze structure using spatial grids:

```python
# Analyze with 100 pc grid cells
structure = spatial.cosmic_web_structure(grid_size_pc=100.0)

print(f"Grid dimensions: {structure['grid_dims']}")
print(f"Occupied fraction: {structure['occupied_fraction']:.1%}")
```

## Visualization

### 3D Structure Plot

```python
fig = viz.plot_3d_structure(
    spatial_tensor,
    cluster_labels,
    title="Cosmic Web Structure",
    point_size=2,
    show_clusters=True
)
```

### Density Map

```python
fig = viz.plot_density_map(
    spatial_tensor,
    density_counts,
    radius_pc=50.0
)
```

### Multi-Scale Comparison

```python
fig = viz.plot_clustering_comparison(
    spatial_tensor,
    clustering_results
)
```

## Analysis Examples

### Gaia Stellar Analysis

```python
from astro_lab.data import analyze_gaia_cosmic_web

# Analyze local stellar structure
results = analyze_gaia_cosmic_web(
    max_samples=1000000,
    magnitude_limit=12.0,
    clustering_scales=[5.0, 10.0, 25.0, 50.0],
    min_samples=5
)

# Results include:
# - n_stars: Total number of stars analyzed
# - clustering_results: Results for each scale
# - density_stats: Density statistics
```

### NSA Galaxy Analysis  

```python
from astro_lab.data import analyze_nsa_cosmic_web

# Analyze large-scale structure
results = analyze_nsa_cosmic_web(
    redshift_limit=0.15,
    clustering_scales=[5.0, 10.0, 20.0, 50.0],  # Mpc
    min_samples=5
)
```

### Exoplanet Host Analysis

```python
from astro_lab.data import analyze_exoplanet_cosmic_web

# Analyze exoplanet host star distribution
results = analyze_exoplanet_cosmic_web(
    clustering_scales=[10.0, 25.0, 50.0, 100.0, 200.0],
    min_samples=3
)
```

## Usage

### Custom Analysis Pipeline

```python
class CustomCosmicWebAnalyzer:
    def __init__(self):
        self.analyzer = CosmicWebAnalyzer()
        
    def analyze_with_properties(self, catalog_df, properties):
        """Analyze cosmic web with additional properties."""
        # Convert to spatial tensor
        spatial = self._create_spatial_tensor(catalog_df)
        
        # Run clustering
        labels = spatial.cosmic_web_clustering(eps_pc=10.0)
        
        # Analyze properties per cluster
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_properties = properties[mask]
            # Analyze cluster properties...
```

### Integration with Graph Neural Networks

```python
from torch_geometric.nn import knn_graph

# Create k-NN graph from cosmic web
edge_index = knn_graph(spatial["coordinates"], k=8)

# Add cosmic web features
node_features = torch.stack([
    density_counts,
    cluster_labels,
    spatial["coordinates"].norm(dim=1),  # Distance
], dim=1)
```

## Performance Tips

1. **Use appropriate scales** for your data:
   - Stars: 5-100 parsecs
   - Galaxies: 5-100 Megaparsecs
   - Exoplanets: 10-500 parsecs

2. **Adjust min_samples** based on density:
   - Dense regions: higher min_samples (10-20)
   - Sparse regions: lower min_samples (3-5)

3. **Memory optimization** for large datasets:
   ```python
   # Process in chunks
   chunk_size = 100000
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       # Process chunk...
   ```

## Scientific Applications

### 1. Stellar Associations
- Identify stellar clusters and moving groups
- Study galactic disk structure
- Find coeval stellar populations

### 2. Galaxy Clusters
- Map large-scale structure
- Identify superclusters and voids
- Study dark matter distribution

### 3. Exoplanet Demographics
- Correlate planet occurrence with stellar environment
- Study planetary system architectures
- Identify stellar neighborhoods

## References

- [Gaia DR3 Documentation](https://www.cosmos.esa.int/web/gaia/dr3)
- [NASA-Sloan Atlas](http://nsatlas.org/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

## See Also

- [Data Module Guide](data_module_guide.md)
- [Tensor System Guide](tensor_guide.md)
- [Visualization Guide](visualization_guide.md)
