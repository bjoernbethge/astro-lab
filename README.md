# 🌌 AstroLab - Astronomical Machine Learning Framework

A comprehensive framework for astronomical data analysis, machine learning, and interactive 3D visualization with advanced cosmic web analysis capabilities.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/bjoernbethge/astro-lab.git
cd astro-lab
uv sync
uv pip install pyg torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### Basic Usage
```python
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

# Load and analyze Gaia stellar data
results = create_cosmic_web_loader(survey="gaia", max_samples=500)

# Create interactive 3D visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")
```

### Training a Model
```bash
# Create configuration
astro-lab create-config -o my_experiment.yaml

# Run complete ML workflow (optimize + train)
astro-lab run -c my_experiment.yaml --auto-optimize
```

## 🌟 Key Features

### 🔬 **Multi-Survey Data Integration**
- **Gaia DR3**: Stellar catalogs with proper motions
- **SDSS**: Galaxy surveys and spectra  
- **NSA**: Galaxy catalogs with distances
- **TNG50**: Cosmological simulations
- **NASA Exoplanet Archive**: Confirmed exoplanets
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

## 📚 Documentation

### Core Documentation
- **[Data Loaders](docs/DATA_LOADERS.md)** - Comprehensive guide to loading and processing astronomical data
- **[Development Guide](docs/DEVGUIDE.md)** - Contributing guidelines and development setup
- **[Cosmograph Integration](docs/COSMOGRAPH_INTEGRATION.md)** - Interactive 3D visualization guide

### Survey-Specific Guides
- **[Gaia Cosmic Web Analysis](docs/GAIA_COSMIC_WEB.md)** - Stellar structure analysis with Gaia DR3
- **[SDSS/NSA Analysis](docs/NSA_COSMIC_WEB.md)** - Galaxy survey analysis
- **[Exoplanet Pipeline](docs/EXOPLANET_PIPELINE.md)** - Exoplanet detection and analysis
- **[Exoplanet Cosmic Web](docs/EXOPLANET_COSMIC_WEB.md)** - Exoplanet spatial distribution analysis

## 🛠️ CLI Tools

### Training Workflow
```bash
# Complete ML workflow (recommended)
astro-lab run -c config.yaml --auto-optimize

# Training only (for production)
astro-lab train -c config.yaml

# Optimization only (for experiments)
astro-lab optimize -c config.yaml -n 50 --update-config
```

### Configuration Management
```bash
# Create default configuration
astro-lab create-config -o my_experiment.yaml

# List available surveys
astro-lab data list-surveys
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
# Train a model for stellar classification
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
# Analyze galaxy structures
results = create_cosmic_web_loader(
    survey="sdss",
    max_samples=1000,
    scales_mpc=[10.0, 20.0, 50.0]
)
```

### Exoplanet Detection
```python
# Process exoplanet data
from astro_lab.tensors import LightcurveTensor

lightcurves = LightcurveTensor.from_exoplanet_archive()
# Analyze transit signals and detect new exoplanets
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

## 📊 Experiment Tracking

All experiments are automatically tracked with MLflow:
- **Metrics**: Training/validation accuracy, loss curves
- **Parameters**: Hyperparameters, model configurations
- **Artifacts**: Model checkpoints, visualizations
- **Reproducibility**: Complete experiment snapshots

### MLflow Configuration
```yaml
mlflow:
  tracking_uri: ./data/experiments
  experiment_name: my_experiment
  experiment_description: "Detailed experiment description"
  tags:
    survey: Gaia
    task: stellar_classification
    version: v1.0
```

## 🐳 Docker Support

### Quick Start with Docker
```bash
# Build and start the container
docker-compose -f docker/docker-compose.yaml up -d

# Access MLflow UI
open http://localhost:5000

# Access Marimo (if started)
open http://localhost:2718

# Run CLI commands in container
docker-compose -f docker/docker-compose.yaml exec astro-lab astro-lab --help
```

### Docker Configuration
- **MLflow**: Automatically configured with persistent storage in `./data/experiments`
- **Ports**: 2718 (Marimo), 5000 (MLflow)
- **Volumes**: 
  - `./data` → `/app/data` (all data, experiments, artifacts)
  - `./src` → `/app/src` (source code for development)
  - `./configs` → `/app/configs` (configuration files)
  - `./snippets` → `/app/snippets` (development snippets)

### Development with Docker
```bash
# Run training in container with local data
docker-compose -f docker/docker-compose.yaml exec astro-lab astro-lab train -c /app/configs/gaia_optimization.yaml

# Access data from container
docker-compose -f docker/docker-compose.yaml exec astro-lab ls /app/data

# Run interactive development
docker-compose -f docker/docker-compose.yaml exec astro-lab marimo edit
```

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

**Ready to explore the cosmos?** Start with our [Data Loaders Guide](docs/DATA_LOADERS.md) or jump into [Interactive Visualization](docs/COSMOGRAPH_INTEGRATION.md)! 