# 🌌 AstroLab - Astronomical Machine Learning Framework

A comprehensive framework for astronomical data analysis, machine learning, and interactive 3D visualization with advanced cosmic web analysis capabilities.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/bjoernbethge/astro-lab.git
cd astro-lab
uv sync
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### First Steps
```bash
# Process data (recommended first step)
astro-lab process --surveys gaia --max-samples 1000

# Start interactive development environment
marimo edit
```

## 🌟 Key Features

### 🔬 **Multi-Survey Data Integration**
- **Gaia DR3**: Stellar catalogs with proper motions
- **SDSS**: Galaxy surveys and spectra  
- **NSA**: Galaxy catalogs with distances
- **TNG50**: Cosmological simulations
- **NASA Exoplanet Archive**: Confirmed exoplanets with Gaia crossmatching
- **LINEAR**: Asteroid light curves

### 🧠 **Advanced Machine Learning**
- **Graph Neural Networks**: For spatial astronomical structures
- **Automatic Class Detection**: From training data
- **Hyperparameter Optimization**: With Optuna integration
- **Experiment Tracking**: MLflow for reproducible research
- **GPU Acceleration**: CUDA-optimized PyTorch workflows

### 🌌 **Cosmic Web Analysis**
- **Multi-scale clustering** across stellar and cosmological scales
- **Adaptive density-based analysis** for all survey types
- **Real-time cosmic web visualization** with CosmographBridge
- **Survey-specific color mapping** and physics simulation

### 🎨 **Interactive 3D Visualization**
- **CosmographBridge**: Seamless integration with cosmic web analysis
- **Survey-specific colors**: Gold for stars, blue for galaxies, green for simulations
- **Real-time physics**: Gravity and repulsion simulation
- **Blender Integration**: Advanced 3D rendering capabilities

## 📚 Documentation & Examples

### Core Documentation
- **[Data Loaders](docs/DATA_LOADERS.md)** - Comprehensive guide to loading and processing astronomical data
- **[Development Guide](docs/DEVGUIDE.md)** - Contributing guidelines and development setup
- **[Cosmograph Integration](docs/COSMOGRAPH_INTEGRATION.md)** - Interactive 3D visualization guide

### Survey-Specific Guides
- **[Gaia Cosmic Web Analysis](docs/GAIA_COSMIC_WEB.md)** - Stellar structure analysis with Gaia DR3
- **[SDSS/NSA Analysis](docs/NSA_COSMIC_WEB.md)** - Galaxy survey analysis
- **[Exoplanet Pipeline](docs/EXOPLANET_PIPELINE.md)** - Exoplanet detection and analysis
- **[Exoplanet Cosmic Web](docs/EXOPLANET_COSMIC_WEB.md)** - Exoplanet spatial distribution analysis

### Interactive Widgets
- **[AstroLab Widget](README_astrolab_widget.md)** - Interactive 3D visualization with Polars, Astropy, and PyVista
- **[Examples](examples/README.md)** - Ready-to-run examples and tutorials

## 🛠️ CLI Reference

### Data Processing
```bash
# Process all surveys
astro-lab process

# Process specific surveys
astro-lab process --surveys gaia nsa --k-neighbors 8 --max-samples 10000

# Advanced processing
astro-lab preprocess catalog data/gaia_catalog.parquet --config gaia --splits
astro-lab preprocess stats data/gaia_catalog.parquet
astro-lab preprocess browse --survey gaia --details
```

### Training & Optimization
```bash
# Create configuration
astro-lab config create -o my_experiment.yaml

# Training
astro-lab train -c my_experiment.yaml
astro-lab train --dataset gaia --model gaia_classifier --epochs 50

# Hyperparameter optimization
astro-lab optimize config.yaml --trials 50
```

### Configuration Management
```bash
# Show available configurations
astro-lab config surveys
astro-lab config show gaia
```

## 🏗️ Architecture

### Core Components
```
astro-lab/
├── src/astro_lab/
│   ├── cli/           # Command-line interface
│   ├── data/          # Data loading and processing
│   ├── models/        # Neural network architectures
│   ├── training/      # Training framework
│   ├── tensors/       # Specialized tensor types
│   └── utils/         # Utilities and visualization
├── configs/           # Configuration files
├── docs/              # Documentation
└── examples/          # Example scripts
```

### Key Dependencies
- **PyTorch 2.7.1+cu128**: GPU-accelerated deep learning
- **Lightning 2.5.1**: Training framework with MLflow integration
- **Polars 1.31.0**: High-performance data processing
- **AstroPy 7.1.0**: Astronomical calculations
- **Cosmograph**: Interactive graph visualization
- **Blender** (optional): Advanced 3D rendering

## 🎯 Use Cases

### Stellar Classification
```python
from astro_lab.models.factory import ModelFactory

model = ModelFactory.create_survey_model(
    survey="gaia",
    task="stellar_classification",
    hidden_dim=128,
    num_classes=8
)
```

### Galaxy Morphology Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

results = create_cosmic_web_loader(
    survey="sdss",
    max_samples=1000,
    scales_mpc=[10.0, 20.0, 50.0]
)
```

### Interactive Visualization
```python
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

# Load and analyze Gaia stellar data
results = create_cosmic_web_loader(survey="gaia", max_samples=500)

# Create interactive 3D visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")
```

## 🔧 Development

### Interactive Development
```bash
# Start Marimo reactive notebook
uv run marimo edit

# Start Jupyter Lab
uv run jupyter lab

# Launch MLflow UI
uv run mlflow ui --backend-store-uri ./data/experiments
```

### Testing
```bash
# Run all tests
uv run pytest -v

# Run specific test categories
uv run pytest test/models/ -v
uv run pytest test/tensors/ -v
```

## 🐳 Docker Support

### Quick Start with Docker
```bash
# Build and start the container
docker-compose -f docker/docker-compose.yaml up -d

# Access services
open http://localhost:5000  # MLflow UI
open http://localhost:2718  # Marimo (if started)

# Run CLI commands in container
docker-compose -f docker/docker-compose.yaml exec astro-lab python -m astro_lab.cli process
```

## 📊 Experiment Tracking

All experiments are automatically tracked with MLflow:
- **Metrics**: Training/validation accuracy, loss curves
- **Parameters**: Hyperparameters, model configurations
- **Artifacts**: Model checkpoints, visualizations
- **Reproducibility**: Complete experiment snapshots

## 🤝 Contributing

See our [Development Guide](docs/DEVGUIDE.md) for detailed contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Gaia Collaboration** for stellar data
- **SDSS Collaboration** for galaxy surveys
- **IllustrisTNG** for cosmological simulations
- **NASA Exoplanet Archive** for exoplanet data

---

**Ready to explore the cosmos?** Start with `astro-lab process` or jump into [Interactive Visualization](docs/COSMOGRAPH_INTEGRATION.md)! 