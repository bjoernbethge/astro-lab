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
- **AstroML Integration**: LINEAR light curves, SDSS spectra
- **Satellite Data**: TLE-based orbital mechanics
- **Simulation Bridge**: Connection to cosmological simulations (TNG50)

### Tensor-based Processing
- **PyTorch Integration**: Full GPU acceleration support
- **Polars Backend**: High-performance data processing
- **Multiple Tensor Types**: Spatial, Photometric, Spectral, Temporal
- **ML-Ready**: Direct usage for deep learning applications

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/bjoernbethge/astro-lab.git
cd astro-lab

# Install with uv (recommended)
uv sync
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

# Process TNG50 simulation data
astro-lab preprocess tng50 data/raw/snap_099.0.hdf5 --particle-types PartType4,PartType5 --max-particles 5000

# List TNG50 snapshots with inspection
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
- **PyTorch Integration**: GPU acceleration available
- **Lazy Loading**: Memory-efficient processing
- **Parquet Format**: Optimized data storage
- **Graph Neural Networks**: Scalable spatial analysis

## 🎯 Scientific Applications

- **Large-Scale Structure**: 3D galaxy distributions
- **Exoplanet Analysis**: Classification, habitability, discovery methods
- **Clustering Analysis**: Neighbor-based studies
- **Machine Learning**: Tensor-based features for GNNs
- **Simulation Comparison**: Observation vs. theory (TNG50 integration)
- **Multi-Survey Analysis**: Cross-matching different catalogs

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