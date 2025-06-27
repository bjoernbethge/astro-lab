# 🌌 AstroLab - Astronomical Machine Learning Framework

A comprehensive framework for astronomical data analysis, machine learning, and interactive 3D visualization with advanced **cosmic web analysis** capabilities across multiple astronomical scales.

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

# Analyze cosmic web structure
astro-lab cosmic-web gaia --max-samples 10000 --clustering-scales 5 10 25 --visualize

# Start interactive development environment
marimo run src/astro_lab/ui/app.py
```

## 🌟 Key Features

### 🔬 **Multi-Survey Data Integration**
- **Gaia DR3**: Stellar catalogs with proper motions and cosmic web clustering
- **SDSS**: Galaxy surveys and spectra with large-scale structure analysis
- **NSA**: Galaxy catalogs with distances and cosmic web visualization  
- **TNG50**: Cosmological simulations with filament detection
- **NASA Exoplanet Archive**: Confirmed exoplanets with host star clustering
- **LINEAR**: Asteroid light curves with orbital family analysis

### 🧠 **Machine Learning**
- **Graph Neural Networks**: Specialized for spatial astronomical structures
- **Automatic Class Detection**: From training data with cosmic web features
- **Hyperparameter Optimization**: With Optuna integration
- **Experiment Tracking**: MLflow for reproducible research
- **GPU Acceleration**: CUDA-optimized PyTorch workflows

### 🌌 **Advanced Cosmic Web Analysis**

#### **Multi-Scale Structure Detection**
- **Stellar Scale**: Local galactic disk structure (1-100 parsecs)
- **Galactic Scale**: Galaxy clusters and superclusters (1-100 Megaparsecs)  
- **Exoplanet Scale**: Stellar neighborhoods and associations (10-500 parsecs)
- **Adaptive Clustering**: DBSCAN, K-means, Hierarchical, and Spectral methods
- **Filament Detection**: MST, Morse theory, and Hessian eigenvalue analysis

#### **Interactive 3D Visualization**
- **CosmographBridge**: Real-time cosmic web visualization with physics simulation
- **Survey-specific colors**: Gold for stars, blue for galaxies, green for simulations
- **Multi-backend support**: PyVista, Open3D, Blender, and Plotly integration
- **Live tensor sync**: Real-time updates between analysis and visualization

#### **Specialized Tensor Operations**
```python
from astro_lab.tensors import SpatialTensorDict

# Create spatial tensor with coordinate system support
spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="parsec")

# Multi-scale cosmic web clustering  
labels = spatial.cosmic_web_clustering(eps_pc=10.0, min_samples=5)

# Grid-based structure analysis
structure = spatial.cosmic_web_structure(grid_size_pc=100.0)

# Local density computation
density = spatial.analyze_local_density(radius_pc=50.0)
```

## 📚 Documentation & API Reference

The complete, up-to-date documentation is available as a modern website:

- **[API Reference](./docs/api/astro_lab/)**
- **[Cosmic Web Guide](./docs/cosmic_web_guide.md)**
- **[User Guide & Examples](./examples/README.md)**

All code is fully documented with mkdocstrings and includes automatic class inheritance diagrams, usage examples, and configuration options.

## 🛠️ CLI Reference

AstroLab provides a comprehensive command-line interface for all aspects of astronomical machine learning and cosmic web analysis.

### Core Commands

```bash
# Show all available commands
astro-lab --help

# Get help for specific commands
astro-lab <command> --help
```

### Data Download
```bash
# Download survey data (standalone module)
python -m astro_lab.cli.download --survey gaia --magnitude-limit 12.0 --region all_sky
python -m astro_lab.cli.download --survey sdss --verbose
python -m astro_lab.cli.download --survey 2mass --region lmc

# List available datasets
python -m astro_lab.cli.download --list
```

### Data Processing
```bash
# Process all surveys with cosmic web features
astro-lab process --surveys gaia nsa sdss --max-samples 10000

# Process specific surveys with spatial indexing
astro-lab process --surveys gaia nsa --k-neighbors 8 --max-samples 10000

# Preprocess raw data files
astro-lab preprocess --surveys gaia sdss --max-samples 5000 --output-dir ./processed_data
```

### Configuration Management
```bash
# Create new configuration file
astro-lab config create -o my_experiment.yaml --template gaia

# Show available survey configurations
astro-lab config surveys

# Show specific survey configuration details
astro-lab config show gaia
astro-lab config show nsa
```

### Model Training
```bash
# Train with configuration file
astro-lab train -c my_experiment.yaml --verbose

# Train with command-line parameters
astro-lab train --dataset gaia --model astro_graph_gnn --epochs 50 --batch-size 32
astro-lab train --dataset nsa --model astro_node_gnn --learning-rate 0.001 --devices 2

# Resume from checkpoint
astro-lab train -c config.yaml --checkpoint path/to/checkpoint.ckpt

# Debug training with small dataset
astro-lab train --dataset gaia --max-samples 1000 --overfit-batches 10
```

### Hyperparameter Optimization
```bash
# Optimize hyperparameters
astro-lab optimize config.yaml --trials 50 --timeout 3600
astro-lab optimize config.yaml --algorithm optuna --trials 100

# Quick optimization for debugging
astro-lab optimize config.yaml --trials 10 --max-samples 1000
```

### Cosmic Web Analysis
```bash
# Multi-scale stellar structure analysis
astro-lab cosmic-web gaia --max-samples 100000 --clustering-scales 5 10 25 50 --visualize

# Large-scale galaxy structure  
astro-lab cosmic-web nsa --clustering-scales 5 10 20 50 --redshift-limit 0.15

# Exoplanet host star clustering
astro-lab cosmic-web exoplanet --clustering-scales 10 25 50 100 200 --min-samples 3

# Custom analysis with output directory
astro-lab cosmic-web gaia --catalog-path ./my_catalog.fits --output-dir ./results --verbose
```

### Supported Surveys
All commands support these astronomical surveys:
- `gaia`: Gaia DR3 stellar catalog
- `sdss`: Sloan Digital Sky Survey  
- `nsa`: NASA-Sloan Atlas galaxy catalog
- `tng50`: TNG50 cosmological simulation
- `exoplanet`: NASA Exoplanet Archive
- `rrlyrae`: RR Lyrae variable stars
- `linear`: LINEAR asteroid survey

## 🔧 Setup Scripts

AstroLab provides automated setup scripts for easy installation across different platforms.

### Linux/macOS Setup (setup.sh)

The `setup.sh` script automates the entire installation process on Linux and macOS systems:

```bash
# Make the script executable and run it
chmod +x setup.sh
./setup.sh
```

**What the script does:**
1. **Installs uv package manager** if not already present
2. **Runs `uv sync`** to install all dependencies from `pyproject.toml`
3. **Installs PyTorch Geometric extensions** for CUDA support
4. **Activates the virtual environment** automatically
5. **Provides instructions** for future activation

**Manual equivalent:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Install dependencies
uv sync
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Activate environment
source .venv/bin/activate
```

### Windows Setup (setup.ps1)

The `setup.ps1` script provides the same functionality for Windows PowerShell:

```powershell
# Run the PowerShell setup script
.\setup.ps1
```

**What the script does:**
1. **Installs uv package manager** via PowerShell
2. **Runs `uv sync`** to install dependencies
3. **Installs PyTorch Geometric extensions** with CUDA support
4. **Activates the virtual environment**
5. **Provides activation instructions** for future use

**Manual equivalent (PowerShell):**
```powershell
# Install uv
irm https://astral.sh/uv/install.ps1 | iex
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# Install dependencies
uv sync
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Activate environment
.\.venv\Scripts\Activate.ps1
```

### Environment Activation

After setup, activate the environment for future sessions:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

### Verification

Test your installation:
```bash
# Check CLI availability
astro-lab --help

# Verify CUDA support (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test cosmic web analysis
astro-lab cosmic-web gaia --max-samples 100 --clustering-scales 5 10
```

## 📖 Documentation Generation

AstroLab includes automated documentation generation and management tools.

### Documentation Scripts

The `docs/generate_docs.py` script provides comprehensive documentation management:

```bash
# Generate/update all documentation
python docs/generate_docs.py update

# Start local documentation server
python docs/generate_docs.py serve

# Deploy documentation to GitHub Pages
python docs/generate_docs.py deploy
```

**What the documentation script does:**
1. **Scans source code** for all Python modules
2. **Generates API documentation** using mkdocstrings
3. **Creates navigation structure** automatically
4. **Builds documentation** with MkDocs
5. **Serves locally** for development
6. **Deploys to GitHub Pages** for production

### Manual Documentation Commands

You can also run documentation commands directly:

```bash
# Install documentation dependencies
uv run pip install mkdocs mkdocstrings[python] mkdocs-material

# Build documentation
uv run mkdocs build --clean

# Serve documentation locally (http://127.0.0.1:8000)
uv run mkdocs serve

# Deploy to GitHub Pages
uv run mkdocs gh-deploy --force
```

### Documentation Structure

The documentation system automatically generates:
- **API Reference**: Complete code documentation with inheritance diagrams
- **Cosmic Web Guide**: Comprehensive analysis tutorials
- **User Guide**: Examples and tutorials
- **Configuration Reference**: All survey and model configurations

## 🤖 Automation and Fabric Scripts

**Note**: This repository does not currently use Fabric (Python remote execution library) for automation. Instead, automation is handled through:

1. **Setup Scripts**: `setup.sh` and `setup.ps1` for environment setup
2. **Documentation Scripts**: `docs/generate_docs.py` for documentation management  
3. **UI Launch Script**: `run_ui.py` for starting the interactive dashboard
4. **Docker Compose**: `docker/docker-compose.yaml` for containerized deployment
5. **CLI Commands**: Built-in automation through the `astro-lab` CLI

### UI Launch Script

The `run_ui.py` script provides an easy way to start the AstroLab interactive dashboard:

```bash
# Start the AstroLab UI dashboard
python run_ui.py

# The dashboard will be available at http://localhost:2718
```

**What the UI script does:**
- Launches the Marimo reactive notebook interface
- Provides access to cosmic web analysis tools
- Enables interactive data visualization  
- Runs on port 2718 by default

For remote deployment and automation needs, the Docker Compose setup provides:
```bash
# Start all services
docker-compose -f docker/docker-compose.yaml up -d

# Execute commands in containers
docker-compose -f docker/docker-compose.yaml exec astro-lab astro-lab cosmic-web gaia --max-samples 1000
```

## 🏗️ Architecture

### Core Components
```
astro-lab/
├── src/astro_lab/
│   ├── cli/
│   │   └── cosmic_web.py      # Cosmic web CLI interface
│   ├── data/
│   │   └── cosmic_web.py      # Core cosmic web analysis
│   ├── tensors/
│   │   └── tensordict_astro.py # Spatial tensor operations  
│   ├── widgets/
│   │   ├── cosmograph_bridge.py   # Interactive visualization
│   │   ├── graph.py               # Graph analysis functions
│   │   └── plotly_bridge.py       # 3D plotting
│   ├── ui/modules/
│   │   ├── cosmic_web.py      # UI for cosmic web analysis
│   │   ├── analysis.py        # Interactive analysis tools
│   │   └── visualization.py   # Visualization interface
│   ├── models/core/           # GNN models for spatial data
│   └── training/              # Training framework
├── configs/                   # Configuration files
├── docs/
│   └── cosmic_web_guide.md    # Comprehensive cosmic web guide
└── examples/                  # Example scripts
```

### Key Dependencies
- **PyTorch 2.7.1+cu128**: GPU-accelerated deep learning with geometric extensions
- **PyTorch Geometric**: Graph neural networks for cosmic web analysis
- **Lightning 2.5.1**: Training framework with MLflow integration
- **Polars 1.31.0**: High-performance data processing
- **AstroPy 7.1.0**: Astronomical calculations and coordinate systems
- **Cosmograph**: Interactive graph visualization with physics simulation
- **Marimo**: Reactive notebooks for interactive analysis
- **scikit-learn**: Clustering algorithms (DBSCAN, K-means, etc.)

## 🎯 Use Cases

### Stellar Structure Analysis
```python
from astro_lab.data.cosmic_web import analyze_gaia_cosmic_web

# Analyze local stellar neighborhoods
results = analyze_gaia_cosmic_web(
    max_samples=100000,
    magnitude_limit=12.0,
    clustering_scales=[5.0, 10.0, 25.0, 50.0],  # parsecs
    min_samples=5
)

print(f"Found {results['n_stars']} stars")
for scale, stats in results['clustering_results'].items():
    print(f"{scale}: {stats['n_clusters']} clusters, {stats['grouped_fraction']:.1%} grouped")
```

### Galaxy Cluster Analysis
```python
from astro_lab.data.cosmic_web import analyze_nsa_cosmic_web

# Large-scale structure analysis
results = analyze_nsa_cosmic_web(
    redshift_limit=0.15,
    clustering_scales=[5.0, 10.0, 20.0, 50.0],  # Mpc
    min_samples=5
)
```

### Interactive Cosmic Web Visualization
```python
from astro_lab.widgets.cosmograph_bridge import CosmographBridge
from astro_lab.data.cosmic_web import CosmicWebAnalyzer

# Load and analyze data
analyzer = CosmicWebAnalyzer()
results = analyzer.analyze_gaia_cosmic_web(max_samples=10000)

# Create interactive 3D visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")

# Display with physics simulation and survey-specific colors
widget.show()  # Gold points for Gaia stars with real-time clustering
```

### Advanced Filament Detection
```python
from astro_lab.tensors import SpatialTensorDict
from astro_lab.data.cosmic_web import CosmicWebAnalyzer

# Create spatial tensor
spatial = SpatialTensorDict(coordinates, coordinate_system="icrs", unit="parsec")

# Detect filamentary structures
analyzer = CosmicWebAnalyzer()
filaments = analyzer.detect_filaments(
    spatial, 
    method="mst",  # or "morse_theory", "hessian"
    n_neighbors=20,
    distance_threshold=10.0
)

print(f"Detected {filaments['n_filament_segments']} filament segments")
print(f"Total filament length: {filaments['total_filament_length']:.1f} pc")
```

### Multi-Backend Visualization
```python
from astro_lab.widgets.tensor_bridge import create_tensor_bridge

# Create visualization bridge
bridge = create_tensor_bridge(backend="cosmograph")  # or "pyvista", "blender"

# Visualize cosmic web with clustering
viz = bridge.cosmic_web_to_backend(
    spatial_tensor=spatial,
    cluster_labels=labels,
    point_size=2.0,
    show_filaments=True
)
```

## 🔧 Development

### Interactive Development
```bash
# Start Marimo reactive notebook with cosmic web UI
uv run marimo run src/astro_lab/ui/app.py

# Launch MLflow UI for experiment tracking
uv run mlflow ui --backend-store-uri ./data/experiments
```

### Testing Cosmic Web Features
```bash
# Test cosmic web analysis
python test_cosmic_web.py

# Run full test suite
uv run pytest -v

# Test specific cosmic web components
uv run pytest test/test_cosmic_web.py -v
uv run pytest src/astro_lab/tensors/ -v -k cosmic_web
```

## 🐳 Docker Support

### Quick Start with Docker
```bash
# Build and start the container with cosmic web support
docker-compose -f docker/docker-compose.yaml up -d

# Access services
open http://localhost:5000  # MLflow UI
open http://localhost:2718  # Marimo UI with cosmic web analysis

# Run cosmic web analysis in container
docker-compose -f docker/docker-compose.yaml exec astro-lab python -m astro_lab.cli cosmic-web gaia --max-samples 1000
```

## 📊 Experiment Tracking

All cosmic web analyses are automatically tracked with MLflow:

```python
# Results are logged with cosmic web metadata
- clustering_scales: [5.0, 10.0, 25.0, 50.0]
- survey_type: "gaia" 
- n_clusters_per_scale: {5.0: 125, 10.0: 89, 25.0: 45, 50.0: 12}
- filament_detection_method: "mst"
- visualization_backend: "cosmograph"
```

## 🎨 Visualization Gallery

### Supported Backends
- **Cosmograph**: Interactive 3D with physics simulation and survey-specific colors
- **PyVista**: High-quality 3D rendering with filament visualization  
- **Plotly**: Web-based interactive plots with multi-scale clustering
- **Blender**: Professional 3D rendering and animation via albpy integration
- **Open3D**: Real-time point cloud visualization with octree support

### Example Visualizations
- **Gaia stellar neighborhoods**: Gold points with gravitational clustering
- **NSA galaxy superclusters**: Blue points with large-scale structure
- **TNG50 cosmic web**: Green points with dark matter filaments
- **Exoplanet host clusters**: Magenta points with stellar associations

## 🤝 Contributing

We welcome contributions to cosmic web analysis features! See our [contribution guidelines](CONTRIBUTING.md) for details on:

- Adding new filament detection algorithms
- Implementing additional clustering methods  
- Creating visualization backends
- Extending survey support
- Improving performance with GPU acceleration

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**AstroLab** - Bridging astronomy and machine learning with advanced cosmic web analysis across all scales of the universe. 🌌✨ 