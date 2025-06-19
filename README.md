# Astro-Lab: Astronomical Tensor Library

A specialized Python library for astronomical data processing with PyTorch-based tensors optimized for astronomical coordinate systems, photometry, spectroscopy, and 3D spatial analysis.

## 🌟 Key Features

### 3D Spatial Coordinates
- **Direct Distance Measurements**: Uses precise ZDIST values from NSA catalog (not just redshift)
- **Spatial3DTensor**: Complete 3D coordinate processing (Spherical ↔ Cartesian)
- **Neighbor Search**: Efficient algorithms for galaxy clustering
- **Density Fields**: 3D grid-based structure analysis
- **Volume Calculations**: Cosmological volumes and number densities

### Data Sources
- **NSA (NASA Sloan Atlas)**: 145,155 galaxies with direct distance measurements
- **Exoplanets**: 5,921 confirmed exoplanets from NASA Exoplanet Archive
- **AstroML Integration**: LINEAR light curves, SDSS spectra
- **Satellite Data**: TLE-based orbital mechanics
- **Simulation Bridge**: Connection to cosmological simulations

### Tensor-based Processing
- **PyTorch Integration**: Full GPU acceleration support
- **Polars Backend**: High-performance data processing
- **Multiple Tensor Types**: Spatial, Photometric, Spectral, Temporal
- **ML-Ready**: Direct usage for deep learning applications

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/user/astro-lab.git
cd astro-lab

# Install with uv (recommended)
uv pip install -e .
```

### Download Exoplanet Data
```bash
# Download all confirmed exoplanets (5,921 planets)
python scripts/download_all_exoplanets.py

# Analyze the data
python examples/exoplanet_data_summary.py
```

### Basic 3D Coordinate Analysis
```python
from astro_lab.data import load_nsa_data
from astro_lab.tensors.spatial_3d import Spatial3DTensor

# Load NSA data with direct distance measurements
df = load_nsa_data(max_samples=1000)

# Create 3D spatial tensor
catalog_data = {
    "RA": df["RA"].to_pandas(),
    "DEC": df["DEC"].to_pandas(), 
    "ZDIST": df["ZDIST"].to_pandas(),  # Direct distances!
    "Z": df["Z"].to_pandas()
}

spatial_tensor = Spatial3DTensor.from_catalog_data(
    catalog_data,
    ra_col="RA",
    dec_col="DEC",
    distance_col="ZDIST"  # Uses precise measurements
)

# 3D analysis
ra, dec, distance = spatial_tensor.get_coordinates_spherical()
x, y, z = spatial_tensor.get_coordinates_cartesian()

# Neighbor search
neighbors = spatial_tensor.find_neighbors(radius=0.01)  # 0.01 Mpc

# Density field
density_field = spatial_tensor.get_density_field(grid_size=20)
```

## 📊 Data Quality

### NSA Distance Measurements (ZDIST)
- **Range**: 0.002 - 0.055 Mpc (local galaxies)
- **Precision**: Direct measurements, not just Hubble relation
- **Uncertainties**: ZDIST_ERR available
- **Completeness**: 100% of NSA galaxies have ZDIST

## 🛠️ Development

### Project Structure
```
astro-lab/
├── src/astro_lab/
│   ├── data/           # Data processing (Polars)
│   ├── tensors/        # Tensor implementations
│   ├── models/         # ML models
│   └── simulation/     # Simulation bridge
├── examples/           # Usage examples
├── data/              # Data storage
└── scripts/           # Utility scripts
```

### Tests & Demos
```bash
# Quick check of 3D coordinates
python check_3d_coords.py

# Full 3D demo
python examples/working_3d_demo.py
```

## 📈 Performance

- **Polars Backend**: 10-100x faster than Pandas
- **PyTorch Integration**: GPU acceleration available
- **Lazy Loading**: Memory-efficient processing
- **Parquet Format**: Optimized data storage

## 🎯 Scientific Applications

- **Large-Scale Structure**: 3D galaxy distributions
- **Exoplanet Analysis**: Classification, habitability, discovery methods
- **Clustering Analysis**: Neighbor-based studies
- **Machine Learning**: Tensor-based features
- **Simulation Comparison**: Observation vs. theory
- **Multi-Survey Analysis**: Cross-matching different catalogs

## 🤝 Contributing

Contributions are welcome! Please create issues for bugs or feature requests.

## 📄 License

MIT License - see LICENSE file for details. 