# 🌌 Cosmic Web Analysis

Comprehensive guide to cosmic web analysis using AstroLab's advanced data processing and visualization capabilities.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📊 Available Surveys](#-available-surveys)
- [🔧 Analysis Pipeline](#-analysis-pipeline)
- [🎨 Visualization](#-visualization)
- [📈 Results & Statistics](#-results--statistics)
- [🛠️ CLI Commands](#️-cli-commands)
- [⚡ Performance Optimization](#-performance-optimization)
- [🔬 Scientific Applications](#-scientific-applications)
- [📚 Related Documentation](#-related-documentation)

## 🚀 Quick Start

```python
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

# Load and analyze Gaia stellar data
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0, 20.0]
)

# Create interactive visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")
```

## 📊 Available Surveys

| Survey | Type | Description | Optimal Scales (Mpc) |
|--------|------|-------------|----------------------|
| **Gaia** | Stellar | Stellar catalogs with proper motions | 1.0, 2.0, 5.0, 10.0 |
| **SDSS** | Galaxy | Galaxy photometry and spectroscopy | 5.0, 10.0, 20.0, 50.0 |
| **NSA** | Galaxy | Galaxy catalogs with distances | 5.0, 10.0, 20.0, 50.0 |
| **TNG50** | Simulation | Cosmological simulation particles | 5.0, 10.0, 20.0, 50.0 |
| **LINEAR** | Solar System | Asteroid light curves | 5.0, 10.0, 20.0, 50.0 |
| **Exoplanet** | Planetary | NASA Exoplanet Archive | 10.0, 25.0, 50.0, 100.0 |

## 🔧 Analysis Pipeline

### 1. Data Loading
```python
from astro_lab.data.core import create_cosmic_web_loader

# Single survey analysis
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[1.0, 2.0, 5.0, 10.0]
)

print(f"Found {results['n_objects']} objects")
print(f"Volume: {results['total_volume']:.0f} Mpc³")
```

### 2. Multi-Survey Comparison
```python
# Compare different surveys
surveys = ["gaia", "sdss", "nsa", "tng50"]
all_results = {}

for survey in surveys:
    all_results[survey] = create_cosmic_web_loader(
        survey=survey,
        max_samples=500,
        scales_mpc=[5.0, 10.0, 20.0]
    )
```

### 3. Tensor Creation
```python
from astro_lab.tensors import Spatial3DTensor

# Create spatial tensor from cosmic web results
spatial_tensor = Spatial3DTensor(results["coordinates"])
```

## 🎨 Visualization

### Interactive 3D Visualization
```python
from astro_lab.utils.viz import CosmographBridge

# Create interactive visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0,
    background_color='#000011'
)
```

### Multi-Survey Comparison
```python
# Compare different surveys visually
surveys = ["gaia", "sdss", "nsa", "tng50"]
widgets = []

for survey in surveys:
    results = create_cosmic_web_loader(survey=survey, max_samples=500)
    widget = bridge.from_cosmic_web_results(results, survey_name=survey)
    widgets.append(widget)
```

## 📈 Results & Statistics

### Statistical Analysis
```python
# Analyze cosmic web structure
print(f"Total objects: {results['n_objects']:,}")
print(f"Total volume: {results['total_volume']:.0f} Mpc³")
print(f"Global density: {results['global_density']:.2e} obj/Mpc³")

# Multi-scale clustering results
for scale, result in results["results_by_scale"].items():
    print(f"\n{scale} Mpc scale:")
    print(f"  Groups: {result['n_clusters']}")
    print(f"  Grouped fraction: {result['grouped_fraction']*100:.1f}%")
    print(f"  Mean local density: {result['mean_local_density']:.2e} obj/pc³")
    print(f"  Processing time: {result['time_s']:.1f}s")
```

### Density Analysis
```python
# Local density statistics
for scale, result in results["results_by_scale"].items():
    density_stats = result['local_density_stats']
    print(f"\n{scale} Mpc scale density:")
    print(f"  Range: {density_stats['min']:.2e} - {density_stats['max']:.2e}")
    print(f"  Median: {density_stats['median']:.2e} obj/pc³")
    print(f"  Variation: {result['density_variation']:.2e}")
```

## 🛠️ CLI Commands

### Single Survey Processing
```bash
# Process Gaia data with cosmic web analysis
astro-lab preprocess cosmic-web gaia --max-samples 1000 --scales 5.0 10.0 20.0 --output results/

# Process SDSS data
astro-lab preprocess cosmic-web sdss --max-samples 500 --scales 10.0 20.0 50.0 --output results/

# Process TNG50 data
astro-lab preprocess cosmic-web tng50 --max-samples 1000 --scales 5.0 10.0 20.0 50.0 --output results/
```

### Batch Processing
```bash
# Process all available surveys at once
astro-lab preprocess all-surveys --max-samples 500 --scales 5.0 10.0 20.0 50.0 --output results/

# List available surveys
astro-lab preprocess surveys

# Enable detailed logging
astro-lab preprocess cosmic-web gaia --max-samples 1000 --verbose
```

## ⚡ Performance Optimization

### Memory Management
```python
# Use tensor operations for efficiency
from astro_lab.tensors import optimize_memory_usage

optimized_data = optimize_memory_usage(
    data,
    precision="float16",
    chunk_size=1000
)
```

### Batch Processing
```python
# Process large datasets in batches
for batch in data_batches:
    results = create_cosmic_web_loader(
        survey="gaia",
        data=batch,
        scales_mpc=[5.0, 10.0, 20.0]
    )
    # Process batch results
```

### GPU Acceleration
```python
# Automatic GPU detection and usage
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    device="cuda"  # Automatically detected if available
)
```

## 🔬 Scientific Applications

### Stellar Cluster Analysis
```python
# Analyze stellar clusters in Gaia data
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=2000,
    scales_mpc=[1.0, 2.0, 5.0]  # Small scales for clusters
)
```

### Galaxy Distribution Analysis
```python
# Analyze large-scale galaxy distribution
results = create_cosmic_web_loader(
    survey="sdss",
    max_samples=1000,
    scales_mpc=[10.0, 20.0, 50.0]  # Large scales for galaxies
)
```

### Cosmological Simulation Analysis
```python
# Analyze TNG50 simulation data
results = create_cosmic_web_loader(
    survey="tng50",
    max_samples=5000,
    scales_mpc=[5.0, 10.0, 20.0, 50.0]
)
```

### Multi-Survey Comparison Study
```python
# Compare clustering across different surveys
surveys = ["gaia", "sdss", "nsa", "tng50"]
all_results = {}

for survey in surveys:
    all_results[survey] = create_cosmic_web_loader(
        survey=survey,
        max_samples=500,
        scales_mpc=[5.0, 10.0, 20.0]
    )

# Compare clustering efficiency
for scale in [5.0, 10.0, 20.0]:
    print(f"\nScale: {scale} Mpc")
    for survey, results in all_results.items():
        efficiency = results["results_by_scale"][scale]['grouped_fraction'] * 100
        print(f"  {survey}: {efficiency:.1f}% grouped")
```

## 🌟  Features

### Adaptive Clustering Parameters
The cosmic web analysis automatically adapts clustering parameters based on local density:

- **Adaptive eps**: Clustering radius adjusts to density variation
- **Adaptive min_samples**: Minimum group size based on local density
- **Multi-scale analysis**: Different spatial scales for comprehensive analysis

### Output Structure
```
results/
├── gaia_coords_3d_mpc.pt          # 3D coordinates tensor
├── gaia_cosmic_web_summary.txt    # Detailed analysis summary
├── sdss_coords_3d_mpc.pt
├── sdss_cosmic_web_summary.txt
└── ...
```

## 📚 Related Documentation

### Core Documentation
- **[Data Loaders](DATA_LOADERS.md)** - Comprehensive data loading guide
- **[Cosmograph Integration](COSMOGRAPH_INTEGRATION.md)** - Interactive visualization
- **[Development Guide](DEVGUIDE.md)** - Contributing guidelines

### Survey-Specific Guides
- **[Gaia Cosmic Web](GAIA_COSMIC_WEB.md)** - Stellar structure analysis
- **[SDSS/NSA Analysis](NSA_COSMIC_WEB.md)** - Galaxy survey analysis
- **[Exoplanet Pipeline](EXOPLANET_PIPELINE.md)** - Exoplanet detection workflows

### Main Documentation
- **[Main README](../README.md)** - Complete framework overview
- **[Examples](../examples/README.md)** - Ready-to-run examples

## 🎯 Use Cases

### Research Applications
- **Galaxy Formation**: Study galaxy clustering and evolution
- **Dark Matter**: Analyze dark matter distribution
- **Cosmology**: Large-scale structure analysis
- **Stellar Dynamics**: Stellar cluster analysis

### Educational Applications
- **Interactive Learning**: 3D visualization of cosmic structures
- **Data Science**: Real-world astronomical data analysis
- **Scientific Computing**: High-performance data processing

---

**Ready to explore the cosmic web?** Start with [Data Loading](DATA_LOADERS.md) or dive into [Interactive Visualization](COSMOGRAPH_INTEGRATION.md)! 