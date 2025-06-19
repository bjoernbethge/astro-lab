# Astro-Lab: Astronomical Tensor Library

A specialized Python library for astronomical data processing with PyTorch-based tensors optimized for astronomical coordinate systems, photometry, spectroscopy, and 3D spatial analysis.

## 🌟 Key Features

### 3D Spatial Coordinates
- **Direct Distance Measurements**: Support for precise distance values (e.g., NSA ZDIST)
- **Spatial3DTensor**: Complete 3D coordinate processing (Spherical ↔ Cartesian)
- **Neighbor Search**: Efficient algorithms for galaxy clustering
- **Density Fields**: 3D grid-based structure analysis
- **Volume Calculations**: Cosmological volumes and number densities

### Data Sources
- **NSA (NASA Sloan Atlas)**: Galaxy catalog with distance measurements
- **Exoplanets**: Confirmed exoplanets from NASA Exoplanet Archive
- **TNG50 Simulations**: Batch processing of all particle types (Gas, Dark Matter, Stars, Black Holes)
- **Gaia DR3**: Stellar catalogs with proper motions and parallax
- **AstroML Integration**: LINEAR light curves, SDSS spectra
- **Satellite Data**: TLE-based orbital mechanics

### Tensor-based Processing
- **PyTorch Integration**: Full GPU acceleration support with CUDA optimization
- **Polars Backend**: High-performance data processing (10-100x faster than Pandas)
- **Graph Neural Networks**: Spatial graphs from particle simulations
- **Multiple Tensor Types**: Spatial, Photometric, Spectral, Temporal
- **Batch Processing**: Automated processing of multiple datasets
- **ML-Ready**: Direct usage for deep learning applications

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/bjoernbethge/astro-lab.git
cd astro-lab

# Install with uv (recommended)
uv sync
# Install torch geometric extensions
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

## 🔧 AstroLab CLI

The `astro-lab` command provides powerful tools for data processing and machine learning:

### 📥 Download Data
```bash
# Download Gaia DR3 bright stars (magnitude < 12.0)
astro-lab download gaia --magnitude-limit 12.0

# List all available datasets
astro-lab download list
```

### 🔄 Data Preprocessing
```bash
# Show available preprocessing functions
astro-lab preprocess --show-functions

# Process a catalog with statistics and train/val/test splits
astro-lab preprocess process catalog.parquet --stats --create-splits --output processed/

# Process single TNG50 snapshot
astro-lab preprocess tng50 data/raw/snap_099.0.hdf5 --particle-types PartType4,PartType5 --max-particles 5000

# Process ALL TNG50 snapshots at once (NEW!)
astro-lab preprocess tng50 --all-snapshots --all --max-particles 2000 --stats

# Process specific particle types across all snapshots
astro-lab preprocess tng50 --all-snapshots --particle-types PartType0,PartType1 --max-particles 5000

# List all available TNG50 snapshots with inspection
astro-lab preprocess tng50-list --inspect
```

### 🎯 Machine Learning Training
```bash
# Create a default configuration file
astro-lab train create-config --output my_config.yaml

# Train with configuration file
astro-lab train train --config my_config.yaml

# Quick training without config
astro-lab train train --dataset gaia --model gaia_classifier --epochs 50 --batch-size 64

# Hyperparameter optimization
astro-lab train optimize --config optimization_config.yaml
```

### Basic Data Loading
```bash
# Check available datasets
python scripts/check_datasets.py

# Process NSA catalog
python examples/nsa_processing_example.py
```

### Basic 3D Coordinate Analysis
```python
from astro_lab.data import load_catalog
from astro_lab.tensors.spatial_3d import Spatial3DTensor
import polars as pl

# Load NSA data
df = pl.read_parquet("data/processed/nsa/nsa_catalog.parquet").head(1000)

# Convert to spatial tensor
catalog_data = {
    "RA": df["RA"].to_pandas() if "RA" in df.columns else df["ra"].to_pandas(),
    "DEC": df["DEC"].to_pandas() if "DEC" in df.columns else df["dec"].to_pandas(),
    "DISTANCE": df.get_column("zdist").to_pandas() if "zdist" in df.columns else df.get_column("z").to_pandas() * 3000
}

spatial_tensor = Spatial3DTensor.from_catalog_data(
    catalog_data,
    ra_col="RA", 
    dec_col="DEC",
    distance_col="DISTANCE"
)

# 3D analysis
ra, dec, distance = spatial_tensor.get_coordinates_spherical()
x, y, z = spatial_tensor.get_coordinates_cartesian()

# Neighbor search
neighbors = spatial_tensor.find_neighbors(radius=10.0)  # 10 Mpc radius
```

### TNG50 Simulation Processing

Process cosmological simulation data efficiently:

```bash
# Process all snapshots with all particle types
astro-lab preprocess tng50 --all-snapshots --all --max-particles 10000 --stats

# Focus on dark matter and gas
astro-lab preprocess tng50 --all-snapshots --particle-types PartType0,PartType1 --max-particles 20000

# Single snapshot processing
astro-lab preprocess tng50 snap_099.0.hdf5 --particle-types PartType4,PartType5 --max-particles 5000

# Check available snapshots
astro-lab preprocess tng50-list --inspect
```

**Features:**
- ✅ **Robust Mass Handling**: Automatically handles missing mass fields for dark matter
- ✅ **All Particle Types**: Gas, Dark Matter, Stars, Black Holes
- ✅ **Spatial Graphs**: Creates k-NN graphs for machine learning
- ✅ **GPU Acceleration**: CUDA optimization when available
- ✅ **Batch Processing**: Process 11 snapshots automatically
- ✅ **Error Recovery**: Continues processing even if individual snapshots fail

## 🛠️ Development

### Project Structure
```
astro-lab/
├── src/astro_lab/
│   ├── cli/            # Command Line Interface  
│   ├── data/           # Data processing (Polars)
│   ├── tensors/        # Tensor implementations
│   ├── models/         # ML models
│   ├── training/       # Lightning training
│   └── utils/          # Utilities & visualization
├── examples/           # Usage examples
├── scripts/           # Utility scripts
├── test/             # Test suite
└── docs/             # Documentation
```

### Available Examples
```bash
# NSA galaxy processing
python examples/nsa_processing_example.py

# AstroQuery demonstrations  
python examples/astroquery_demo.py

# FITS optimization
python examples/fits_optimization_demo.py

# Dataset verification
python scripts/check_datasets.py
```

## 📈 Performance

- **Polars Backend**: 10-100x faster than Pandas
- **PyTorch Integration**: GPU acceleration with CUDA support
- **Batch Processing**: Process 11 TNG50 snapshots automatically
- **Robust Error Handling**: Handles missing data fields gracefully
- **Lazy Loading**: Memory-efficient processing
- **Parquet Format**: Optimized data storage
- **Graph Neural Networks**: Scalable spatial analysis with k-NN fallback

## 🎯 Scientific Applications

- **Large-Scale Structure**: 3D galaxy distributions and cosmic web analysis
- **Cosmological Simulations**: TNG50 particle analysis (Gas, Dark Matter, Stars, Black Holes)
- **Exoplanet Analysis**: Classification, habitability, discovery methods
- **Clustering Analysis**: Neighbor-based studies with spatial graphs
- **Machine Learning**: Graph Neural Networks for astronomical objects
- **Simulation Comparison**: Observation vs. theory (TNG50 integration)
- **Multi-Survey Analysis**: Cross-matching different catalogs (NSA, Gaia, etc.)

## 🧪 Testing

```bash
# Run test suite
python scripts/run_tests.py

# Check CUDA availability
python test/test_cuda.py

# Verify datasets
python scripts/check_datasets.py
```

## 🤝 Contributing

Contributions are welcome! Please create issues for bugs or feature requests.

## 📄 License

MIT License - see LICENSE file for details. 