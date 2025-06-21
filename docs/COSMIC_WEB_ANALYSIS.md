# 🌌 Cosmic Web Analysis with AstroLab

Comprehensive guide to cosmic web analysis using AstroLab's advanced data processing and visualization capabilities.

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

### 🌟 Stellar Surveys
- **Gaia DR3**: Stellar catalogs with proper motions and photometry
- **LINEAR**: Asteroid light curves and variable stars

### 🌌 Galaxy Surveys
- **SDSS DR17**: Galaxy photometry and spectroscopy
- **NSA**: Galaxy catalogs with distance measurements

### 🪐 Exoplanet Data
- **NASA Exoplanet Archive**: Confirmed exoplanet systems

### 🌌 Cosmological Simulations
- **TNG50**: IllustrisTNG cosmological simulation particles

## 🔧 Data Processing Pipeline

### 1. Raw Data Loading
```python
from astro_lab.data.core import load_gaia_data

# Load raw survey data
gaia_tensor = load_gaia_data(
    max_samples=10000,
    return_tensor=True
)
```

### 2. Cosmic Web Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

# Perform cosmic web analysis
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[1.0, 2.0, 5.0, 10.0]
)

print(f"Found {results['n_objects']} objects")
print(f"Volume: {results['total_volume']:.0f} Mpc³")
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
# Compare different surveys
surveys = ["gaia", "sdss", "nsa", "tng50"]
widgets = []

for survey in surveys:
    results = create_cosmic_web_loader(survey=survey, max_samples=500)
    widget = bridge.from_cosmic_web_results(results, survey_name=survey)
    widgets.append(widget)
```

## 📈 Analysis Results

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

## 🛠️ CLI Usage

### Process Single Survey
```bash
# Process Gaia data with cosmic web analysis
astro-lab preprocess cosmic-web gaia --max-samples 1000 --scales 5.0 10.0 20.0 --output results/

# Process SDSS data
astro-lab preprocess cosmic-web sdss --max-samples 500 --scales 10.0 20.0 50.0 --output results/

# Process TNG50 data
astro-lab preprocess cosmic-web tng50 --max-samples 1000 --scales 5.0 10.0 20.0 50.0 --output results/
```

### Process All Surveys
```bash
# Process all available surveys at once
astro-lab preprocess all-surveys --max-samples 500 --scales 5.0 10.0 20.0 50.0 --output results/
```

### List Available Surveys
```bash
# Show all available surveys and their descriptions
astro-lab preprocess surveys
```

### Verbose Logging
```bash
# Enable detailed logging for debugging
astro-lab preprocess cosmic-web gaia --max-samples 1000 --verbose
```

### Interactive Analysis
```bash
# Start interactive session
python -c "from astro_lab.data.core import create_cosmic_web_loader; create_cosmic_web_loader('gaia', max_samples=1000)"
```

## 📊 Performance Optimization

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

## 🌟 Advanced Features

### Adaptive Clustering Parameters
The cosmic web analysis automatically adapts clustering parameters based on local density:

- **Adaptive eps**: Clustering radius adjusts to density variation
- **Adaptive min_samples**: Minimum group size based on local density
- **Multi-scale analysis**: Different spatial scales for comprehensive analysis

### Logging and Monitoring
```bash
# Standard logging (INFO level)
astro-lab preprocess cosmic-web gaia
# Output: 14:30:15 - astro_lab_cli - INFO - 🌌 Cosmic Web Analysis for survey: gaia

# Verbose logging (DEBUG level)
astro-lab preprocess cosmic-web gaia --verbose
# Output: 14:30:15 - astro_lab_cli - DEBUG - Loading coordinates...
```

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

- **[Data Loaders](DATA_LOADERS.md)**: Comprehensive data loading guide
- **[Cosmograph Integration](COSMOGRAPH_INTEGRATION.md)**: Interactive visualization
- **[CLI Usage](CLI_USAGE.md)**: Command-line interface guide
- **[Development Guide](DEVGUIDE.md)**: Contributing guidelines

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