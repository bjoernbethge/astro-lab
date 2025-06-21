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
- **Gaia DR3**: Stellar catalogs with proper motions
- **LINEAR**: Asteroid light curves and variable stars

### 🌌 Galaxy Surveys
- **SDSS DR17**: Galaxy photometry and spectra
- **NSA**: Galaxy catalogs with distances

### 🪐 Exoplanet Data
- **NASA Exoplanet Archive**: Confirmed exoplanets

### 🌌 Cosmological Simulations
- **TNG50**: IllustrisTNG cosmological simulations

## 🔧 Data Processing Pipeline

### 1. Raw Data Loading
```python
from astro_lab.data.core import load_survey_data

# Load raw survey data
raw_data = load_survey_data(
    survey="gaia",
    magnitude_limit=12.0,
    max_samples=10000
)
```

### 2. Cosmic Web Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

# Perform cosmic web analysis
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[1.0, 2.0, 5.0, 10.0],
    return_tensor=True
)

print(f"Found {results['n_objects']} objects")
print(f"Volume: {results['total_volume']:.0f} Mpc³")
```

### 3. Tensor Creation
```python
from astro_lab.tensors import Spatial3DTensor

# Create spatial tensor from cosmic web results
spatial_tensor = Spatial3DTensor.from_catalog_data({
    "RA": results["ra"],
    "DEC": results["dec"],
    "DISTANCE": results["distance"]
})
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
```

### Density Analysis
```python
# Local density statistics
density_stats = results['local_density_stats']
print(f"Density range: {density_stats['min']:.2e} - {density_stats['max']:.2e}")
print(f"Median density: {density_stats['median']:.2e} obj/pc³")
```

## 🛠️ CLI Usage

### Process Survey Data
```bash
# Process Gaia data
python scripts/process_gaia_cosmic_web.py

# Process SDSS data
python scripts/process_sdss_cosmic_web.py

# Process TNG50 data
python scripts/process_tng_cosmic_web.py
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

## 📚 Related Documentation

- **[Data Loaders](docs/DATA_LOADERS.md)**: Comprehensive data loading guide
- **[Cosmograph Integration](docs/COSMOGRAPH_INTEGRATION.md)**: Interactive visualization
- **[Development Guide](docs/DEVGUIDE.md)**: Contributing guidelines

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

**Ready to explore the cosmic web?** Start with [Data Loading](docs/DATA_LOADERS.md) or dive into [Interactive Visualization](docs/COSMOGRAPH_INTEGRATION.md)! 