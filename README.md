# AstroLab: Comprehensive Astronomical Data Analysis Framework

A modern Python framework for astronomical data analysis, machine learning, and visualization that combines specialized astronomy libraries with cutting-edge ML tools.

## 🚀 Project Overview

AstroLab is designed as a comprehensive ecosystem for astronomical research, featuring:

- **Interactive Development**: Marimo reactive notebooks and Jupyter integration
- **ML Experiment Tracking**: MLflow for reproducible research
- **Specialized Data Types**: Astronomy-specific tensor implementations
- **3D Visualization**: Blender and PyVista integration
- **GPU Acceleration**: CUDA-optimized PyTorch workflows
- **Graph Neural Networks**: For spatial astronomical data structures

## 📦 Architecture

### Core Package Structure

```
astro-lab/
├── astro-viz/          # 3D Visualization & Blender Integration
├── astro-torch/        # PyTorch & ML for Astronomy  
├── astro-pack/         # Astronomy Libraries & Data Access
├── astro-lab/          # Main Framework
└── astro-lab-ml/       # ML-specific Extensions
```

### Key Components

#### 🎯 Interactive Development Stack
- **Marimo v0.14.0**: Reactive notebook system for interactive development
- **MLflow v3.1.0**: Experiment tracking and model management
- **Jupyter**: Traditional notebook support

#### 🔬 Astronomy-Specific Libraries
- **AstroPy v7.1.0**: Core astronomy computations
- **AstroML v1.0.2**: Machine learning for astronomy
- **AstroQuery v0.4.10**: Database queries (Gaia, SDSS, etc.)
- **AstroPhot v0.16.13**: Advanced photometry with PyTorch backend

#### 🧠 Machine Learning Stack
- **PyTorch v2.7.1+cu128**: GPU-accelerated deep learning
- **Lightning v2.5.1**: Training framework
- **Torch Geometric v2.6.1**: Graph neural networks
- **Optuna v4.4.0**: Hyperparameter optimization

#### 📊 Data Processing
- **Polars v1.31.0**: High-performance dataframes (10-100x faster than Pandas)
- **PyArrow v20.0.0**: Columnar data processing
- **NumPy v1.26.4** + **SciPy v1.15.3**: Scientific computing

## 🌟 Key Features

### Specialized Tensor Types
```python
from astro_lab.tensors import (
    Spatial3DTensor,      # 3D coordinates & transformations
    PhotometricTensor,    # Multi-band photometry
    SpectralTensor,       # Spectroscopy data
    LightcurveTensor,     # Time-series observations
    OrbitalTensor         # Satellite & planetary orbits
)
```

### Data Sources Integration
- **Gaia DR3**: Stellar catalogs with proper motions
- **SDSS**: Galaxy surveys and spectra
- **TNG50**: Cosmological simulations
- **NASA Exoplanet Archive**: Confirmed exoplanets
- **LINEAR**: Asteroid light curves
- **NSA**: Galaxy catalogs with distances

### Advanced ML Capabilities
- **Graph Neural Networks**: For spatial astronomical structures
- **3D Point Cloud Models**: Stellar cluster analysis
- **Temporal Models**: Variable star classification
- **Multi-modal Learning**: Combined photometry, spectroscopy, and astrometry

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/astro-lab.git
cd astro-lab

# Install with uv (recommended)
uv sync

# Verify installation
uv run pytest -v
```

### Interactive Development

```bash
# Start Marimo reactive notebook
uv run marimo edit

# Start Jupyter Lab
uv run jupyter lab

# Launch MLflow UI
uv run mlflow ui
```

## 🔧 CLI Tools

### Data Management
```bash
# Download Gaia DR3 data
astro-lab download gaia --magnitude-limit 12.0

# Process TNG50 simulations
astro-lab preprocess tng50 --all-snapshots --all --max-particles 10000

# Create train/validation splits
astro-lab preprocess process catalog.parquet --create-splits
```

### Machine Learning
```bash
# Create configuration
astro-lab train create-config --output config.yaml

# Train models
astro-lab train train --config config.yaml

# Hyperparameter optimization
astro-lab train optimize --config optuna_config.yaml
```

## 💻 Development Examples

### Basic Tensor Usage
```python
import polars as pl
from astro_lab.tensors import Spatial3DTensor

# Load catalog data
df = pl.read_parquet("data/gaia_sample.parquet")

# Create spatial tensor
spatial = Spatial3DTensor.from_catalog_data({
    "RA": df["ra"].to_pandas(),
    "DEC": df["dec"].to_pandas(), 
    "DISTANCE": df["distance"].to_pandas()
})

# 3D analysis
neighbors = spatial.find_neighbors(radius=10.0)  # 10 pc radius
x, y, z = spatial.get_coordinates_cartesian()
```

### Machine Learning Pipeline
```python
from astro_lab.models import create_gaia_classifier
from astro_lab.data import AstroDataModule
from astro_lab.training import AstroTrainer

# Load data
data_module = AstroDataModule("data/processed/gaia/")

# Create model
model = create_gaia_classifier(
    input_features=["bp_rp", "g_mag", "parallax"],
    num_classes=5
)

# Train
trainer = AstroTrainer(max_epochs=100, accelerator="gpu")
trainer.fit(model, data_module)
```

### 3D Visualization
```python
from astro_lab.utils.blender import create_galaxy_visualization

# Create 3D scene
scene = create_galaxy_visualization(
    positions=galaxy_positions,
    colors=galaxy_colors,
    sizes=galaxy_masses
)

# Render
scene.render("galaxy_cluster.png")
```

## 🧪 Testing & Development

```bash
# Run all tests
uv run pytest -v

# Test specific components
uv run pytest test/models/ -v
uv run pytest test/tensors/ -v

# Check dependencies
uv tree
uv tree --package astro-torch
```

## 📁 Project Structure

```
src/astro_lab/
├── cli/                    # Command Line Interface
│   ├── download.py         # Data acquisition
│   ├── preprocessing.py    # Data processing
│   └── train.py           # ML training
├── data/                   # Data Processing & Loading
│   ├── core.py            # Core data structures
│   ├── manager.py         # Data management
│   └── transforms.py      # Data transformations
├── models/                 # ML Models & Architectures
│   ├── base_gnn.py        # Base graph neural networks
│   ├── factory.py         # Model factory
│   ├── encoders.py        # Feature encoders
│   ├── output_heads.py    # Task-specific heads
│   └── point_cloud_models.py  # 3D stellar models
├── tensors/                # Specialized Tensor Types
│   ├── spatial_3d.py      # 3D coordinates
│   ├── photometric.py     # Multi-band photometry
│   ├── spectral.py        # Spectroscopy
│   └── lightcurve.py      # Time series
├── training/               # Training Utilities
│   ├── lightning_module.py # PyTorch Lightning integration
│   ├── mlflow_logger.py   # Experiment tracking
│   └── trainer.py         # Training orchestration
└── utils/                  # Utility Functions
    ├── blender/           # 3D visualization
    ├── graph.py           # Graph utilities
    └── tensor.py          # Tensor operations
```

## 🎯 Research Applications

- **Galaxy Classification**: Multi-band photometry analysis
- **Variable Star Detection**: Time-series classification
- **Exoplanet Discovery**: Transit detection and characterization
- **Stellar Cluster Analysis**: 3D spatial clustering
- **Cosmological Simulations**: Large-scale structure analysis
- **Asteroid Tracking**: Orbital mechanics and light curves

## 📚 Documentation

- [Development Guide](docs/DEVGUIDE.md) - Detailed setup and architecture
- [Data Loaders](docs/DATA_LOADERS.md) - Data processing documentation
- [Exoplanet Pipeline](docs/EXOPLANET_PIPELINE.md) - Specialized workflows
- [Examples](examples/) - Practical usage examples

## 🤝 Contributing

1. **Setup Development Environment**: `uv sync`
2. **Run Tests**: `uv run pytest -v`
3. **Follow Code Style**: Use provided pre-commit hooks
4. **Add Tests**: Ensure new features have test coverage
5. **Update Documentation**: Keep docs current with changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with modern astronomy and machine learning libraries:
- AstroPy Collaboration
- PyTorch Team
- Polars Development Team
- MLflow Community
- Marimo Developers 