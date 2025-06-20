# 🌌 AstroLab - Astronomical Data Analysis Framework

AstroLab is a comprehensive framework for astronomical data analysis, visualization, and machine learning with advanced cosmic web analysis and interactive 3D visualization.

## 🚀 Quick Start

```python
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

# Load real survey data with cosmic web analysis
results = create_cosmic_web_loader(survey="gaia", max_samples=500)

# Create interactive 3D visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")

# Explore the cosmic web interactively!
```

## 📦 Main Dependencies

The framework requires the following main dependencies:
- **numpy** - Scientific computing
- **torch** - GPU-accelerated deep learning
- **polars** - High-performance data processing
- **astropy** - Astronomical calculations
- **pyvista** - 3D visualization
- **cosmograph** - Interactive graph visualization
- **blender** (optional) - Advanced 3D rendering

## 🌟 Key Features

### 🌌 Cosmic Web Analysis
- **Multi-scale clustering** across stellar and cosmological scales
- **Adaptive density-based analysis** for all survey types
- **Real-time cosmic web visualization** with CosmographBridge
- **Survey-specific color mapping** and physics simulation

### 🎨 Interactive 3D Visualization
- **CosmographBridge**: Seamless integration with cosmic web analysis
- **Survey-specific colors**: Gold for stars, blue for galaxies, green for simulations
- **Real-time physics**: Gravity and repulsion simulation
- **Multi-survey support**: Gaia, SDSS, NSA, TNG50, LINEAR, Exoplanets

### 🔬 Specialized Tensor Types
```python
from astro_lab.tensors import (
    Spatial3DTensor,      # 3D coordinates & transformations
    PhotometricTensor,    # Multi-band photometry
    SpectralTensor,       # Spectroscopy data
    LightcurveTensor,     # Time-series observations
    OrbitalTensor         # Satellite & planetary orbits
)
```

### 📊 Data Sources Integration
- **Gaia DR3**: Stellar catalogs with proper motions
- **SDSS**: Galaxy surveys and spectra
- **TNG50**: Cosmological simulations
- **NASA Exoplanet Archive**: Confirmed exoplanets
- **LINEAR**: Asteroid light curves
- **NSA**: Galaxy catalogs with distances

### 🧠 Advanced ML Capabilities
- **Graph Neural Networks**: For spatial astronomical structures
- **3D Point Cloud Models**: Stellar cluster analysis
- **Temporal Models**: Variable star classification
- **Multi-modal Learning**: Combined photometry, spectroscopy, and astrometry
- **2025 System Metrics**: Real-time hardware monitoring (CPU, GPU, memory, disk)
- **Automatic Hardware Detection**: GPU optimization, precision selection, device management

## 🚀 Project Overview

AstroLab is designed as a comprehensive ecosystem for astronomical research, featuring:

- **Interactive Development**: Marimo reactive notebooks and Jupyter integration
- **ML Experiment Tracking**: MLflow for reproducible research
- **Specialized Data Types**: Astronomy-specific tensor implementations
- **3D Visualization**: Blender and PyVista integration
- **GPU Acceleration**: CUDA-optimized PyTorch workflows
- **Graph Neural Networks**: For spatial astronomical data structures
- **Cosmic Web Analysis**: Multi-scale structure analysis across all surveys

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

## 🌌 Cosmic Web Analysis Examples

### Basic Cosmic Web Analysis
```python
from astro_lab.data.core import create_cosmic_web_loader

# Analyze Gaia stellar cosmic web
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0, 20.0]
)

print(f"Found {results['n_objects']} objects")
print(f"Volume: {results['total_volume']:.0f} Mpc³")
```

### Interactive Visualization
```python
from astro_lab.utils.viz import CosmographBridge

# Create interactive 3D visualization
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

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/astro-lab.git
cd astro-lab

# Install with uv (recommended)
uv sync
# needed
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
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

AstroLab provides a comprehensive command-line interface for all major operations:

### 📋 Available Commands

```bash
astro-lab download       # Download astronomical datasets
astro-lab preprocess     # Data preprocessing and graph creation  
astro-lab train          # Single ML model training
astro-lab optimize       # Hyperparameter optimization with Optuna
astro-lab config         # Configuration management
astro-lab cosmic-web     # Cosmic web analysis and visualization
```

**Get help for any command:**
```bash
astro-lab --help                    # Main help
astro-lab train --help              # Training options
astro-lab optimize --help           # Optimization options
astro-lab config --help             # Configuration management
astro-lab cosmic-web --help         # Cosmic web analysis
```

### 📥 Data Download & Management

```bash
# Download astronomical survey data
astro-lab download gaia --magnitude-limit 12.0 --output data/gaia/
astro-lab download sdss --survey dr17 --max-objects 100000
astro-lab download nsa --catalog v1_0_1

# Quick data exploration
astro-lab download --list-surveys          # Show available surveys
astro-lab download --status                # Show download progress
```

### 🌌 Cosmic Web Analysis

```bash
# Perform cosmic web analysis
astro-lab cosmic-web gaia --max-samples 1000 --scales 5.0 10.0 20.0
astro-lab cosmic-web sdss --output results/sdss_cosmic_web/
astro-lab cosmic-web nsa --create-visualization

# Multi-survey analysis
astro-lab cosmic-web compare --surveys gaia sdss nsa --output results/comparison/
```

### 🔄 Data Preprocessing

```bash
# Process raw survey data
astro-lab preprocess gaia data/raw/gaia_dr3.csv --create-graphs --k-neighbors 8
astro-lab preprocess sdss data/raw/sdss_dr17.fits --normalize --create-splits

# TNG50 cosmological simulations
astro-lab preprocess tng50 data/tng50/ --all-snapshots --particle-types PartType4
astro-lab preprocess tng50 data/tng50/ --snapshot 99 --max-particles 10000

# Generic data processing
astro-lab preprocess process catalog.parquet --create-splits --train-ratio 0.7
astro-lab preprocess browse data/processed/  # Browse processed data
```

### 🧠 Machine Learning Training

#### Configuration-Based Training (Recommended)

```bash
# Create default configuration
astro-lab config create --output configs/my_experiment.yaml

# Single training run with fixed parameters
astro-lab train --config configs/gaia_classification.yaml

# Hyperparameter optimization with multiple trials
astro-lab optimize configs/gaia_optimization.yaml --trials 50
```

#### 🎯 Train vs Optimize - When to Use What?

**`astro-lab train`** - Single Training Run
- ✅ Train **one** model with **fixed** hyperparameters
- ✅ Use exact parameters from config file
- ✅ Fast and direct (1-2 minutes)
- ✅ Good for: Final models, known parameters, quick tests

**`astro-lab optimize`** - Hyperparameter Search  
- ✅ Run **multiple** trainings with **different** parameters
- ✅ Use Optuna to find optimal hyperparameters
- ✅ Longer but finds best settings (10-60 minutes)
- ✅ Good for: New datasets, unknown parameters, model tuning

```bash
# Examples:
astro-lab train --config gaia.yaml              # Single training
astro-lab optimize gaia.yaml --trials 50        # 50 optimization trials
astro-lab optimize gaia.yaml --trials 10 --experiment-name "quick_test"
```

#### Configuration Management

```bash
# List available survey configurations
astro-lab config surveys

# Show specific survey configuration
astro-lab config show gaia

# Create custom configuration
astro-lab config create --output my_config.yaml
```

### 📊 Experiment Tracking & Results

AstroLab integrates with MLflow for comprehensive experiment tracking:

```bash
# Local MLflow UI
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

# Docker container setup (recommended)
# The marimo-flow container automatically includes MLflow
docker ps  # Check if marimo-flow container is running
# Access at: http://localhost:5000

# Sync local experiments to container
robocopy mlruns "D:\marimo-flow\data\mlflow\mlruns" /E /XO
docker restart marimo-flow
```

#### MLflow Configuration

```yaml
mlflow:
  tracking_uri: ./mlruns                    # Local tracking
  # tracking_uri: D:/marimo-flow/data/mlflow/mlruns  # Container tracking
  experiment_name: my_experiment
  experiment_description: "Detailed experiment description"
  tags:
    survey: Gaia
    task: stellar_classification
    version: v1.0
```

#### Viewing Results

1. **Open MLflow UI**: http://localhost:5000
2. **Navigate to your experiment**: `gaia_optuna_optimization`
3. **Compare trials**: Sort by `val_loss` to see best results
4. **View parameters**: See which hyperparameters worked best
5. **Download models**: Access trained model artifacts
6. **📊 System Metrics**: View real-time hardware performance:
   - `system/cpu/utilization_percent` - CPU usage during training
   - `system/gpu_0/memory_allocated_gb` - GPU memory consumption
   - `system/memory/utilization_percent` - RAM usage
   - `system/disk/utilization_percent` - Storage usage

## ⚙️ Configuration System

AstroLab uses YAML configuration files for reproducible experiments:

### Basic Configuration Structure

```yaml
# configs/gaia_classification.yaml
model:
  type: gaia_classifier
  params:
    hidden_dim: 128
    num_classes: 8
    dropout: 0.1
    use_batch_norm: true

data:
  dataset: gaia
  data_dir: data/processed
  batch_size: 64
  max_samples: 50000
  return_tensor: true
  split_ratios: [0.7, 0.15, 0.15]

training:
  max_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  accelerator: auto
  devices: 1
  precision: 16-mixed

mlflow:
  experiment_name: gaia_stellar_classification
  tracking_uri: ./mlruns
  tags:
    survey: Gaia
    task: stellar_classification
```

### Hyperparameter Optimization Configuration

```yaml
# configs/gaia_optimization.yaml
model:
  type: gaia_classifier
  params:
    hidden_dim: 128  # Will be optimized
    num_classes: 8
    dropout: 0.1     # Will be optimized

data:
  dataset: gaia
  batch_size: 64
  max_samples: 50000

training:
  max_epochs: 50
  learning_rate: 0.001  # Will be optimized
  accelerator: auto
  precision: 16-mixed

optimization:
  n_trials: 50
  timeout: 7200  # 2 hours
  direction: maximize
  study_name: gaia_stellar_classification
  
  search_space:
    learning_rate:
      type: loguniform
      low: 1e-5
      high: 1e-1
    hidden_dim:
      type: categorical
      choices: [64, 128, 256, 512, 768]
    dropout:
      type: uniform
      low: 0.05
      high: 0.3
    weight_decay:
      type: loguniform
      low: 1e-6
      high: 1e-3

mlflow:
  experiment_name: gaia_optuna_optimization
  tracking_uri: ./mlruns
```

### Available Survey Configurations

```bash
# View all available surveys
astro-lab config surveys
```

**Pre-configured Surveys:**
- **Gaia DR3**: `configs/surveys/gaia.yaml` - Stellar astrometry and photometry
- **SDSS DR17**: `configs/surveys/sdss.yaml` - Galaxy photometry and spectra  
- **NSA**: `configs/surveys/nsa.yaml` - Galaxy catalog with distances

### Parameter Distribution System

AstroLab automatically distributes configuration parameters to the correct components:

```python
# Automatic parameter routing:
# training.* → PyTorch Lightning Trainer
# model.* → Model initialization  
# optimization.* → Optuna hyperparameter search
# mlflow.* → Experiment tracking
# data.* → Data loading and processing
```

### CLI Examples by Use Case

#### 🌟 Stellar Classification (Gaia)
```bash
# Download Gaia data
astro-lab download gaia --magnitude-limit 12.0

# Single training run
astro-lab train --config configs/surveys/gaia.yaml

# Hyperparameter optimization
astro-lab optimize configs/gaia_optimization.yaml --trials 50
```

#### 🌌 Galaxy Analysis (SDSS)
```bash
# Download SDSS data
astro-lab download sdss --survey dr17

# Process and create graphs
astro-lab preprocess sdss data/sdss/ --create-graphs --k-neighbors 10

# Train galaxy classifier
astro-lab train --config configs/surveys/sdss.yaml
```

#### 🪐 Exoplanet Detection
```bash
# Download NASA Exoplanet Archive data
astro-lab download exoplanets --confirmed-only

# Train transit detection model
astro-lab train --dataset exoplanets --model transit_detector --epochs 100
```

### Advanced CLI Usage

#### Debugging & Development
```bash
# Verbose logging
astro-lab train --config config.yaml --verbose

# Disable tensor optimizations (for debugging)
astro-lab train --config config.yaml --disable-tensors

# Test configuration without training
astro-lab train --config config.yaml --dry-run
```

#### Resource Management
```bash
# Specify GPU device
astro-lab train --config config.yaml --devices 0

# Use multiple GPUs
astro-lab train --config config.yaml --devices 2 --accelerator gpu

# CPU-only training
astro-lab train --config config.yaml --accelerator cpu
```

## 📋 Quick Reference

### Most Common Commands

```bash
# 1. Download and setup Gaia data
astro-lab download gaia --magnitude-limit 12.0
astro-lab preprocess gaia data/gaia/ --create-graphs

# 2. Train a stellar classifier
astro-lab train --config configs/surveys/gaia.yaml

# 3. Optimize hyperparameters (NEW!)
astro-lab optimize configs/gaia_optimization.yaml --trials 50

# 4. View results with system metrics
mlflow ui --backend-store-uri ./mlruns
# Open: http://localhost:5000
```

### Configuration Templates

| Task | Config File | Description |
|------|-------------|-------------|
| Stellar Classification | `configs/surveys/gaia.yaml` | Gaia DR3 stellar classification |
| Galaxy Analysis | `configs/surveys/sdss.yaml` | SDSS galaxy photometry |
| Hyperparameter Tuning | `configs/gaia_optimization.yaml` | Optuna-based optimization |
| Custom Experiment | `configs/default.yaml` | Basic template |

### 🖥️ System Monitoring (NEW in 2025!)

AstroLab automatically logs comprehensive system metrics during training:

```bash
# System metrics logged every 30 seconds:
# ├── system/cpu/utilization_percent      # CPU usage
# ├── system/memory/utilization_percent   # RAM usage  
# ├── system/gpu_0/memory_allocated_gb    # GPU VRAM
# ├── system/gpu_0/utilization_percent    # GPU usage
# ├── system/disk/utilization_percent     # Storage usage
# └── system/network/bytes_sent_mb        # Network I/O

# View in MLflow UI under "Metrics" tab with hierarchical grouping
# Automatic hardware detection: GPU, CUDA, precision optimization
```

**System Requirements Monitoring:**
- **CPU**: Multi-core utilization tracking
- **GPU**: VRAM, temperature, utilization (NVIDIA RTX/Tesla)
- **Memory**: RAM consumption and availability
- **Storage**: Disk usage and I/O patterns
- **Network**: Data transfer monitoring

### Troubleshooting

```bash
# Check system status
astro-lab --version
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Debug training issues
astro-lab train --config config.yaml --verbose --dry-run

# Reset MLflow experiments
rm -rf mlruns/
# Or on Windows: rmdir /s mlruns

# Container issues
docker ps
docker logs marimo-flow
docker restart marimo-flow
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