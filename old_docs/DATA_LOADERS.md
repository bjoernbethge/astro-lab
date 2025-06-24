# 🌌 Data Loaders Guide

Comprehensive guide to loading and processing astronomical data with AstroLab.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📊 Available Surveys](#-available-surveys)
- [🔧 Core Functions](#-core-functions)
- [🌌 Cosmic Web Analysis](#-cosmic-web-analysis)
- [📈 Common Workflows](#-common-workflows)
- [🛠️ CLI Commands](#️-cli-commands)
- [⚡ Performance Tips](#-performance-tips)
- [🐛 Troubleshooting](#-troubleshooting)
- [📚 Related Documentation](#-related-documentation)

## 🚀 Quick Start

### Basic Data Loading
```python
from astro_lab.data.core import create_cosmic_web_loader

# Load and analyze Gaia stellar data
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0, 20.0]
)

print(f"Found {results['n_objects']} objects")
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
    survey_name="gaia",
    radius=3.0
)
```

### Training Data
```python
from astro_lab.data import create_astro_datamodule

# Create data module for machine learning
datamodule = create_astro_datamodule(
    dataset="gaia",
    batch_size=32,
    max_samples=5000,
    return_tensor=True
)

train_loader = datamodule.train_dataloader()
```

## 📊 Available Surveys

| Survey | Type | Description | Features | Guide |
|--------|------|-------------|----------|-------|
| **Gaia** | Stellar | Gaia DR3 stellar catalog | Position, brightness, motion | [Gaia Guide](GAIA_COSMIC_WEB.md) |
| **SDSS** | Galaxy | SDSS DR17 galaxies | Colors, redshift, morphology | [NSA Guide](NSA_COSMIC_WEB.md) |
| **NSA** | Galaxy | NASA Sloan Atlas | Sérsic profiles, distances | [NSA Guide](NSA_COSMIC_WEB.md) |
| **TNG50** | Simulation | Cosmological simulation | 3D positions, masses, velocities | [Cosmic Web](COSMIC_WEB_ANALYSIS.md) |
| **LINEAR** | Solar System | Asteroid light curves | Periods, variability | - |
| **Exoplanet** | Planetary | NASA Exoplanet Archive | Orbital parameters | [Exoplanet Guide](EXOPLANET_PIPELINE.md) |

## 🔧 Core Functions

### Cosmic Web Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

# Single survey analysis
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0, 20.0]
)

# Multi-survey comparison
surveys = ["gaia", "nsa", "tng50"]
all_results = {}

for survey in surveys:
    all_results[survey] = create_cosmic_web_loader(
        survey=survey,
        max_samples=500,
        scales_mpc=[5.0, 10.0]
    )
```

### Direct Data Loading
```python
from astro_lab.data import load_gaia_data, load_sdss_data, load_tng50_data

# Load specific surveys
stars = load_gaia_data(max_samples=5000)
galaxies = load_sdss_data(max_samples=2000)
simulation = load_tng50_data(max_samples=10000, particle_type="PartType0")
```

### Graph Creation
```python
from astro_lab.data import AstroDataset

# Create graph dataset for GNNs
dataset = AstroDataset(
    survey="gaia",
    max_samples=1000,
    k_neighbors=8
)

graph = dataset[0]
print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
```

## 🌌 Cosmic Web Analysis

### Basic Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

# Analyze cosmic web structure
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=2000,
    scales_mpc=[5.0, 10.0, 20.0]
)

# Access results
print(f"Objects: {results['n_objects']}")
print(f"Volume: {results['total_volume']:.0f} Mpc³")
print(f"Density: {results['global_density']:.2e} obj/Mpc³")
```

### Visualization Pipeline
```python
from astro_lab.utils.viz import CosmographBridge

# Complete cosmic web visualization pipeline
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0,
    background_color='#000011'
)

# Export results
import json
with open('cosmic_web_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## 📈 Common Workflows

### Stellar Classification
```python
# 1. Load Gaia data
from astro_lab.data import load_gaia_data
stars = load_gaia_data(max_samples=10000)

# 2. Extract features (G, BP, RP magnitudes)
magnitudes = stars.data[:, 5:8]
colors = magnitudes[:, 1] - magnitudes[:, 2]  # BP-RP color

# 3. Simple classification
import numpy as np
labels = np.where(colors > 1.0, 2, np.where(colors > 0.5, 1, 0))

# 4. Training/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    magnitudes, labels, test_size=0.2
)
```

### Galaxy Morphology
```python
# 1. Load SDSS galaxies
from astro_lab.data import load_sdss_data
galaxies = load_sdss_data(max_samples=5000)

# 2. Calculate colors (g-r, r-i)
mags = galaxies.data[:, 4:7]
g_r = mags[:, 0] - mags[:, 1]
r_i = mags[:, 1] - mags[:, 2]

# 3. Morphology classification
morphology = (g_r > 0.7).astype(int)  # Elliptical vs Spiral
```

### TNG50 Simulation Analysis
```python
# 1. Load simulation data
from astro_lab.data import load_tng50_data
dark_matter = load_tng50_data(max_samples=15000, particle_type="PartType1")

# 2. 3D spatial analysis
positions = dark_matter.positions  # [N, 3] in ckpc/h
masses = dark_matter.features[:, 3]

# 3. Density calculation
from scipy.spatial import KDTree
tree = KDTree(positions)
densities = []
for pos in positions[:1000]:
    neighbors = tree.query_ball_point(pos, r=500.0)
    density = len(neighbors) / (4/3 * np.pi * 500**3)
    densities.append(density)
```

### Lightning Training
```python
from astro_lab.data import AstroDataModule
import lightning as L

# 1. Create DataModule
datamodule = AstroDataModule(
    survey="gaia",
    batch_size=64,
    max_samples=20000,
    train_ratio=0.7,
    val_ratio=0.15
)

# 2. Training
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule)
```

## 🛠️ CLI Commands

### Data Processing
```bash
# Process all surveys (recommended)
astro-lab process

# Process specific surveys
astro-lab process --surveys gaia nsa --max-samples 10000

# Advanced processing
astro-lab preprocess catalog data/gaia_catalog.parquet --config gaia --splits
```

### Cosmic Web Analysis
```bash
# Analyze Gaia cosmic web
astro-lab preprocess cosmic-web gaia --max-samples 10000

# Multi-scale analysis
astro-lab preprocess cosmic-web nsa --scales 5.0 10.0 20.0 50.0

# Process all surveys
astro-lab preprocess all-surveys --max-samples 500 --output results/
```

### Data Management
```bash
# Show statistics
astro-lab preprocess stats data/gaia_catalog.parquet --verbose

# Browse data
astro-lab preprocess browse --survey gaia --details

# List available surveys
astro-lab config surveys
```

## ⚡ Performance Tips

### Memory Optimization
```python
# Use smaller samples for testing
dataset = load_gaia_data(max_samples=1000)

# Enable tensor conversion for GPU
dataset = load_gaia_data(return_tensor=True)

# Batch processing for large datasets
dataset = load_gaia_data(max_samples=100000, batch_size=1000)
```

### Speed Improvements
- **Start small**: Use `max_samples=1000` for testing
- **Use caching**: Processed data is automatically cached
- **GPU acceleration**: Enable `return_tensor=True`
- **Batch processing**: Use appropriate batch sizes
- **Parallel loading**: Set `num_workers` in DataLoader

### TNG50 Optimization
```python
# Process one particle type at a time
gas_only = load_tng50_data(max_samples=5000, particle_type="PartType0")
dm_only = load_tng50_data(max_samples=15000, particle_type="PartType1")

# Use higher k for dense simulations
tng50_dataset = AstroDataset(
    survey="tng50",
    max_samples=5000,
    k_neighbors=16  # Higher connectivity
)
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Check installation
uv run python -c "import astro_lab; print('astro-lab installed successfully')"

# Reinstall if needed
uv sync
```

**Memory Issues**
```python
# Reduce sample size
dataset = load_gaia_data(max_samples=1000)

# Use smaller batches
dataset = load_gaia_data(batch_size=10)
```

**Slow Performance**
```python
# Enable GPU acceleration
dataset = load_gaia_data(return_tensor=True)

# Use caching
from astro_lab.data.config import data_config
data_config.cache_dir.mkdir(exist_ok=True)
```

**Missing Data**
```python
# Demo data is automatically generated
dataset = load_gaia_data(max_samples=1000)  # Always works
```

### Debugging
```python
# Check data shapes
data = load_gaia_data(max_samples=100)
print(f"Shape: {data.shape}")
print(f"Features: {data.column_mapping}")

# For simulations
sim_data = load_tng50_data(max_samples=100, particle_type="PartType1")
print(f"Positions: {sim_data.positions.shape}")
print(f"Features: {sim_data.features.shape}")
```

## 📚 Related Documentation

### Survey-Specific Guides
- **[Gaia Cosmic Web Analysis](GAIA_COSMIC_WEB.md)** - Stellar structure analysis
- **[SDSS/NSA Analysis](NSA_COSMIC_WEB.md)** - Galaxy survey analysis
- **[Exoplanet Pipeline](EXOPLANET_PIPELINE.md)** - Exoplanet detection workflows
- **[Exoplanet Cosmic Web](EXOPLANET_COSMIC_WEB.md)** - Spatial distribution analysis

### Technical Documentation
- **[Cosmic Web Analysis](COSMIC_WEB_ANALYSIS.md)** - Complete analysis framework
- **[Cosmograph Integration](COSMOGRAPH_INTEGRATION.md)** - Interactive visualization
- **[Development Guide](DEVGUIDE.md)** - Contributing and architecture

### Main Documentation
- **[Main README](../README.md)** - Complete framework overview
- **[Examples](../examples/README.md)** - Ready-to-run examples
- **[AstroLab Widget](../README_astrolab_widget.md)** - Interactive widget guide

---

**Next Steps**: Start with the [Quick Start](#-quick-start) section or explore specific [Survey Guides](#survey-specific-guides) for your use case!

