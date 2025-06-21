# 🔄 AstroLab Migration Guide

This guide helps you migrate from the old individual survey scripts to the new unified AstroLab data processing approach.

## 🗑️ Removed Scripts

The following outdated scripts have been removed and replaced with modern alternatives:

### Deleted Scripts
- `scripts/process_gaia_cosmic_web.py`
- `scripts/process_nsa_cosmic_web.py`
- `scripts/process_exoplanet_cosmic_web.py`
- `scripts/process_linear_cosmic_web.py`
- `scripts/process_tng_cosmic_web.py`
- `scripts/process_satellite_cosmic_web.py`

### Why They Were Removed
- ❌ Used outdated GraphDataset classes that no longer exist
- ❌ Duplicated code across multiple files
- ❌ Inconsistent APIs and error handling
- ❌ No support for modern features like GPU acceleration
- ❌ Limited to single survey processing

## 🆕 New Modern Approach

### 1. Universal Survey Processor
**New**: `scripts/process_all_surveys.py`

**Features**:
- ✅ Single script for all surveys
- ✅ Consistent API across all surveys
- ✅ Command-line interface with options
- ✅ GPU acceleration support
- ✅ Interactive visualization option
- ✅ Comprehensive error handling

**Usage**:
```bash
# Old way (multiple scripts)
python scripts/process_gaia_cosmic_web.py
python scripts/process_nsa_cosmic_web.py
python scripts/process_exoplanet_cosmic_web.py

# New way (single script)
python scripts/process_all_surveys.py --survey gaia
python scripts/process_all_surveys.py --survey nsa
python scripts/process_all_surveys.py --survey exoplanet

# Process all surveys at once
python scripts/process_all_surveys.py --all --max-samples 500
```

### 2. Modern Dataset Checker
**Updated**: `scripts/check_datasets.py`

**Changes**:
- ❌ Old: Used non-existent `GaiaGraphDataset`, `NSAGraphDataset`, etc.
- ✅ New: Uses modern `create_cosmic_web_loader` function
- ✅ Supports all available surveys
- ✅ Better error handling and reporting

### 3. New Examples
**Added**:
- `examples/quick_start.py` - Simple beginner example
- `examples/modern_data_analysis.py` - Complete feature demonstration

**Updated**:
- `examples/README.md` - Modern documentation with new examples

## 🔄 Migration Steps

### Step 1: Update Your Scripts

**Old Code**:
```python
# Old way - individual scripts
from astro_lab.data.datasets import GaiaGraphDataset, NSAGraphDataset

# Load Gaia data
gaia = GaiaGraphDataset()
g = gaia[0]
print(f"Nodes: {g.num_nodes}")

# Load NSA data  
nsa = NSAGraphDataset()
n = nsa[0]
print(f"Nodes: {n.num_nodes}")
```

**New Code**:
```python
# New way - unified approach
from astro_lab.data.core import create_cosmic_web_loader

# Load Gaia data
gaia_results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0]
)
print(f"Objects: {gaia_results['n_objects']}")

# Load NSA data
nsa_results = create_cosmic_web_loader(
    survey="nsa", 
    max_samples=1000,
    scales_mpc=[5.0, 10.0]
)
print(f"Objects: {nsa_results['n_objects']}")
```

### Step 2: Update Command Line Usage

**Old Commands**:
```bash
# Multiple individual scripts
python scripts/process_gaia_cosmic_web.py
python scripts/process_nsa_cosmic_web.py
python scripts/process_exoplanet_cosmic_web.py
```

**New Commands**:
```bash
# Single unified script
python scripts/process_all_surveys.py --survey gaia
python scripts/process_all_surveys.py --survey nsa  
python scripts/process_all_surveys.py --survey exoplanet

# Process all surveys
python scripts/process_all_surveys.py --all

# With custom parameters
python scripts/process_all_surveys.py --survey gaia --max-samples 1000 --scales 5.0 10.0 20.0
```

### Step 3: Update Your Analysis Code

**Old Analysis**:
```python
# Old way - manual cosmic web analysis
from astro_lab.tensors.spatial_3d import Spatial3DTensor
from astro_lab.utils import calculate_volume, calculate_mean_density

# Manual tensor creation and analysis
spatial_tensor = Spatial3DTensor.from_coordinates(coordinates)
volume = calculate_volume(spatial_tensor)
density = calculate_mean_density(spatial_tensor)
```

**New Analysis**:
```python
# New way - integrated cosmic web analysis
from astro_lab.data.core import create_cosmic_web_loader

# Automatic cosmic web analysis
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0, 20.0]
)

# Results include everything
volume = results['total_volume']
density = results['global_density']
clusters = results['results_by_scale']
```

## 🌟 New Features

### 1. Multi-Survey Processing
```python
# Process multiple surveys efficiently
surveys = ["gaia", "nsa", "exoplanet"]
results = {}

for survey in surveys:
    results[survey] = create_cosmic_web_loader(
        survey=survey,
        max_samples=500,
        scales_mpc=[5.0, 10.0]
    )
```

### 2. Interactive Visualization
```python
# Create interactive 3D visualizations
from astro_lab.utils.viz import CosmographBridge

results = create_cosmic_web_loader(survey="gaia", max_samples=200)
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0,
    physics_enabled=True
)
```

### 3. Advanced Command Line Options
```bash
# List available surveys
python scripts/process_all_surveys.py --list-surveys

# Process with visualization
python scripts/process_all_surveys.py --survey gaia --visualize

# Don't save results
python scripts/process_all_surveys.py --survey gaia --no-save
```

## 📊 Available Surveys

| Survey | Type | Description | Default Scales |
|--------|------|-------------|----------------|
| **gaia** | Stellar | Gaia DR3 stellar catalog | 1.0, 2.0, 5.0, 10.0 Mpc |
| **sdss** | Galaxy | SDSS DR17 galaxies | 5.0, 10.0, 20.0, 50.0 Mpc |
| **nsa** | Galaxy | NASA Sloan Atlas | 5.0, 10.0, 20.0, 50.0 Mpc |
| **linear** | Solar System | LINEAR asteroids | 5.0, 10.0, 20.0, 50.0 Mpc |
| **tng** | Simulation | TNG50 cosmological simulation | 5.0, 10.0, 20.0, 50.0 Mpc |
| **exoplanet** | Planetary | NASA Exoplanet Archive | 10.0, 25.0, 50.0, 100.0 Mpc |

## 🎯 Benefits of Migration

### Performance Improvements
- ✅ GPU acceleration support
- ✅ Optimized data loading
- ✅ Memory-efficient processing
- ✅ Parallel processing capabilities

### Developer Experience
- ✅ Single unified API
- ✅ Better error messages
- ✅ Comprehensive documentation
- ✅ Consistent behavior across surveys

### Features
- ✅ Interactive 3D visualization
- ✅ Multi-scale cosmic web analysis
- ✅ Automatic coordinate conversion
- ✅ Survey-specific optimizations

## 🔧 Troubleshooting

### Common Migration Issues

1. **Import Errors**
   ```python
   # Old (no longer works)
   from astro_lab.data.datasets import GaiaGraphDataset
   
   # New
   from astro_lab.data.core import create_cosmic_web_loader
   ```

2. **Missing Data**
   ```bash
   # Check available surveys
   python scripts/process_all_surveys.py --list-surveys
   
   # Check dataset availability
   python scripts/check_datasets.py
   ```

3. **Memory Issues**
   ```python
   # Reduce sample size
   results = create_cosmic_web_loader(
       survey="gaia",
       max_samples=100,  # Start small
       scales_mpc=[5.0]  # Fewer scales
   )
   ```

## 📚 Resources

### Documentation
- **[Data Loaders](docs/DATA_LOADERS.md)** - Complete data loading guide
- **[Cosmic Web Analysis](docs/COSMIC_WEB_ANALYSIS.md)** - Advanced analysis techniques
- **[Examples](examples/README.md)** - Modern examples and tutorials

### Examples
- `examples/quick_start.py` - Simple beginner example
- `examples/modern_data_analysis.py` - Complete feature demonstration

### Scripts
- `scripts/process_all_surveys.py` - Universal survey processor
- `scripts/check_datasets.py` - Dataset availability checker

## 🎉 Migration Complete!

After following this guide, you'll have:
- ✅ Modern, efficient data processing
- ✅ Unified API across all surveys
- ✅ Better performance and features
- ✅ Future-proof codebase

**Ready to explore?** Start with `examples/quick_start.py`! 