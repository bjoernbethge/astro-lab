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

### Cosmic Web Analysis
```bash
# Multi-scale stellar structure analysis
astro-lab cosmic-web gaia --max-samples 100000 --clustering-scales 5 10 25 50 --visualize

# Large-scale galaxy structure  
astro-lab cosmic-web nsa --clustering-scales 5 10 20 50 --redshift-limit 0.15

# Exoplanet host star clustering
astro-lab cosmic-web exoplanet --clustering-scales 10 25 50 100 200 --min-samples 3
```

### Data Processing
```bash
# Process all surveys with cosmic web features
astro-lab process

# Process specific surveys with spatial indexing
astro-lab process --surveys gaia nsa --k-neighbors 8 --max-samples 10000

# Preprocess with cosmic web metadata
astro-lab preprocess catalog data/gaia_catalog.parquet --config gaia --spatial-index
```

### Training & Optimization
```bash
# Create configuration with cosmic web features
astro-lab config create -o my_experiment.yaml --features cosmic-web

# Training with spatial features
astro-lab train -c my_experiment.yaml --spatial-features
astro-lab train --dataset gaia --model astro_graph_gnn --epochs 50

# Hyperparameter optimization for cosmic web models
astro-lab optimize config.yaml --trials 50 --spatial-aware
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