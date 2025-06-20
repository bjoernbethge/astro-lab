# 🌌 Loading & Processing Astronomical Data

## 🎯 What does this module do?

The `astro_lab.data` module loads astronomical catalogs (stars, galaxies) **and cosmological simulations** and prepares them for machine learning and cosmic web analysis. **Simple, fast, without complexity.**

## 📋 Complete Workflow

### 1️⃣ Load Data (30 seconds)

```python
from astro_lab.data import load_gaia_data, load_sdss_data, load_tng50_data

# Load Gaia stars
stars = load_gaia_data(max_samples=5000)
print(f"✅ {stars.shape[0]} stars loaded")

# Load SDSS galaxies  
galaxies = load_sdss_data(max_samples=2000)
print(f"✅ {galaxies.shape[0]} galaxies loaded")

# Load TNG50 simulation particles
simulation = load_tng50_data(max_samples=10000, particle_type="PartType0")
print(f"✅ {len(simulation):,} gas particles loaded")
```

### 2️⃣ Cosmic Web Analysis

```python
from astro_lab.data.core import create_cosmic_web_loader

# Analyze cosmic web structure
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=1000,
    scales_mpc=[5.0, 10.0, 20.0]
)

print(f"🌌 Found {results['n_objects']} objects in cosmic web")
print(f"📊 Volume: {results['total_volume']:.0f} Mpc³")
print(f"🔗 Clusters: {len(results['clusters'])}")
```

### 3️⃣ Interactive Visualization

```python
from astro_lab.utils.viz import CosmographBridge

# Create interactive 3D visualization
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0
)
```

### 4️⃣ Inspect Data

```python
# What's inside?
print(f"Gaia Features: {stars.photometric_bands}")  # ['G', 'BP', 'RP']
print(f"SDSS Features: {galaxies.photometric_bands}")  # ['u', 'g', 'r', 'i', 'z']
print(f"TNG50 Features: {simulation.feature_names}")  # ['x', 'y', 'z', 'masses', 'velocities_*', 'density']

# First rows
print(stars.data[:3])  # First 3 stars
print(galaxies.data[:3])  # First 3 galaxies
print(simulation.positions[:3])  # First 3 particle positions
```

### 5️⃣ Processing for ML

```python
# For PyTorch training
import torch
from torch.utils.data import DataLoader

# As regular PyTorch tensor
X = stars.data  # [5000, 8] - 5000 stars, 8 features
y = torch.randint(0, 3, (5000,))  # Dummy labels

# Simulation data
sim_positions = simulation.positions  # [10000, 3] - 3D coordinates
sim_features = simulation.features    # [10000, 8] - all features

# DataLoader for training
loader = DataLoader(
    list(zip(X, y)), 
    batch_size=32, 
    shuffle=True
)

# Training loop
for batch_x, batch_y in loader:
    # Your ML model goes here
    pass
```

### 6️⃣ For Graph Neural Networks

```python
from astro_lab.data import AstroDataset

# Create graph dataset
dataset = AstroDataset(
    survey="gaia",
    max_samples=1000,
    k_neighbors=8  # 8 nearest neighbors
)

# TNG50 3D simulation graphs
tng50_dataset = AstroDataset(
    survey="tng50",
    max_samples=5000,
    k_neighbors=16  # Higher connectivity for dense simulation
)

# Inspect graph
graph = dataset[0]
print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

# For PyTorch Geometric
from torch_geometric.loader import DataLoader
graph_loader = DataLoader([graph], batch_size=1)
```

## 🗂️ Available Data

| Survey | What | Count | Features | Usage |
|--------|-----|--------|----------|------------|
| **Gaia** | Stars | 5k-50k | Position, brightness, motion | Stellar Classification, Cosmic Web |
| **SDSS** | Galaxies | 1k-10k | Colors, redshift, morphology | Galaxy Classification, Cosmic Web |
| **NSA** | Galaxies | 1k-5k | Sérsic profiles, mass | Galaxy Evolution, Cosmic Web |
| **TNG50** | Simulation | 1k-20k per type | 3D positions, masses, velocities | Cosmic Web, Dark Matter |
| **LINEAR** | Asteroids | 500-2k | Light curves, periods | Variable Stars |
| **Exoplanet** | Planets | 500-1k | Orbital parameters | Planetary Systems |

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
print(f"Clusters: {len(results['clusters'])}")
```

### Multi-Survey Cosmic Web Comparison
```python
# Compare different surveys
surveys = ["gaia", "sdss", "nsa", "tng50"]
results = {}

for survey in surveys:
    results[survey] = create_cosmic_web_loader(
        survey=survey,
        max_samples=500,
        scales_mpc=[5.0, 10.0]
    )
    print(f"{survey}: {results[survey]['n_objects']} objects")
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

## 🚀 Quick Start Recipes

### Recipe 1: Stellar Classification

```python
# 1. Load data
from astro_lab.data import load_gaia_data
stars = load_gaia_data(max_samples=10000)

# 2. Extract features (G, BP, RP magnitudes)
magnitudes = stars.data[:, 5:8]  # Columns 5,6,7 are G,BP,RP
colors = magnitudes[:, 1] - magnitudes[:, 2]  # BP-RP color

# 3. Simple classification
import numpy as np
# Red giants (BP-RP > 1.0), main sequence (0.5-1.0), blue stars (<0.5)
labels = np.where(colors > 1.0, 2, np.where(colors > 0.5, 1, 0))

# 4. Training/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    magnitudes, labels, test_size=0.2
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")
```

### Recipe 2: Galaxy Morphology

```python
# 1. Load galaxies
from astro_lab.data import load_sdss_data
galaxies = load_sdss_data(max_samples=5000)

# 2. Calculate colors (g-r, r-i)
mags = galaxies.data[:, 4:7]  # g,r,i magnitudes
g_r = mags[:, 0] - mags[:, 1]  # g-r
r_i = mags[:, 1] - mags[:, 2]  # r-i

# 3. Morphology features
features = np.column_stack([g_r, r_i, galaxies.data[:, 2]])  # + redshift

# 4. Elliptical (red), spiral (blue) separation
# Ellipticals: g-r > 0.7, Spirals: g-r < 0.7
morphology = (g_r > 0.7).astype(int)

print(f"Ellipticals: {np.sum(morphology)}, Spirals: {len(morphology) - np.sum(morphology)}")
```

### Recipe 3: TNG50 Cosmic Web Analysis

```python
# 1. Load simulation data
from astro_lab.data import load_tng50_data
import numpy as np

# Load different particle types
dark_matter = load_tng50_data(max_samples=15000, particle_type="PartType1")
gas = load_tng50_data(max_samples=10000, particle_type="PartType0") 
stars = load_tng50_data(max_samples=5000, particle_type="PartType4")

# 2. 3D spatial analysis
dm_positions = dark_matter.positions  # [N, 3] in ckpc/h
dm_masses = dark_matter.features[:, 3]  # Mass column

# 3. Calculate density field
from scipy.spatial import KDTree
tree = KDTree(dm_positions)
densities = []

for pos in dm_positions[:1000]:  # Sample subset
    neighbors = tree.query_ball_point(pos, r=500.0)  # 500 ckpc/h
    density = len(neighbors) / (4/3 * np.pi * 500**3)  # Number density
    densities.append(density)

print(f"Mean density: {np.mean(densities):.2e} particles/ckpc³")

# 4. Cosmic web classification
high_density = np.array(densities) > np.percentile(densities, 80)
print(f"High-density regions: {np.sum(high_density)} / {len(densities)}")
```

### Recipe 4: Lightning Training

```python
# 1. Create DataModule
from astro_lab.data import AstroDataModule
import lightning as L

datamodule = AstroDataModule(
    survey="gaia",
    batch_size=64,
    max_samples=20000,
    train_ratio=0.7,
    val_ratio=0.15
)

# 2. Simple model
class StellarClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64),  # 8 Gaia features
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)   # 3 stellar classes
        )
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # Your loss function here
        pass

# 3. Training
model = StellarClassifier()
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule)
```

## 🖥️ CLI Preprocessing

For large-scale processing, use the command-line interface:

```bash
# Process Gaia data
python -m astro_lab.cli.preprocessing gaia --max-samples 50000

# Process SDSS galaxies  
python -m astro_lab.cli.preprocessing sdss --max-samples 10000

# Process NSA catalog
python -m astro_lab.cli.preprocessing nsa --enable-clustering

# Process TNG50 simulation (all particle types)
python -m astro_lab.cli.preprocessing tng50 --max-samples 20000

# Advanced TNG50 with features
python -m astro_lab.cli.preprocessing tng50 \
  --max-samples 50000 \
  --enable-clustering \
  --enable-statistics \
  --output-dir results/tng50_full
```

## 🌌 Cosmic Web CLI

```bash
# Perform cosmic web analysis
python -m astro_lab.cli.cosmic_web gaia --max-samples 1000 --scales 5.0 10.0 20.0
python -m astro_lab.cli.cosmic_web sdss --output results/sdss_cosmic_web/
python -m astro_lab.cli.cosmic_web nsa --create-visualization

# Multi-survey analysis
python -m astro_lab.cli.cosmic_web compare --surveys gaia sdss nsa --output results/comparison/
```

## 🔧 Common Problems & Solutions

### Problem: "No real data found"
```python
# Solution: Demo data is automatically generated
dataset = load_gaia_data(max_samples=1000)
# ✅ Always works, even without internet
```

### Problem: "Too slow with large datasets"
```python
# Solution: Use fewer samples
quick_data = load_gaia_data(max_samples=1000)  # Instead of 50000

# For TNG50: Process one particle type at a time
gas_only = load_tng50_data(max_samples=5000, particle_type="PartType0")
```

### Problem: "Wrong feature dimensions"
```python
# Solution: Check shape
data = load_gaia_data(max_samples=100)
print(f"Shape: {data.shape}")  # [100, 8]
print(f"Features: {data.column_mapping}")  # Which column is what

# For simulations
sim_data = load_tng50_data(max_samples=100, particle_type="PartType1")
print(f"Positions: {sim_data.positions.shape}")  # [100, 3]
print(f"Features: {sim_data.features.shape}")   # [100, N]
```

### Problem: "TNG50 particle types"
```python
# Solution: Available particle types
particle_types = {
    "PartType0": "Gas particles",
    "PartType1": "Dark matter", 
    "PartType4": "Star particles",
    "PartType5": "Black holes"
}

# Load specific type
gas = load_tng50_data(particle_type="PartType0", max_samples=10000)
dark_matter = load_tng50_data(particle_type="PartType1", max_samples=15000)
```

## 📊 What happens internally?

1. **Auto-Download**: If no data exists, synthetic astronomical data is generated
2. **Preprocessing**: Coordinates are converted, missing values handled
3. **Graph Creation**: k-NN graphs are built for spatial relationships
4. **Cosmic Web Analysis**: Multi-scale clustering and structure detection
5. **Tensor Integration**: Native AstroLab tensor support for advanced workflows
6. **Caching**: Processed data is cached for faster subsequent loads
7. **Simulation Loading**: TNG50 Parquet files are loaded with simulation metadata

## 🎓 Next Steps

Once you have your data loaded:

```python
# → Create PyTorch datasets
dataset = torch.utils.data.TensorDataset(X, y)

# → Build k-NN graphs
from astro_lab.data import create_knn_graph
graph = create_knn_graph(coordinates, k=8)

# → Edge list for PyTorch Geometric
edge_index = graph.edge_index

# → For simulations: 3D cosmic web graphs
sim_graph = create_knn_graph(simulation.positions, k=16)

# → Cosmic web analysis
from astro_lab.data.core import create_cosmic_web_loader
results = create_cosmic_web_loader(survey="gaia", max_samples=1000)

# → Interactive visualization
from astro_lab.utils.viz import CosmographBridge
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")
```

## 🔄 Data Flow Architecture

```
Raw Catalogs → Preprocessing → Feature Engineering → Graph Building → Cosmic Web → ML Ready
     ↓              ↓              ↓                ↓               ↓            ↓
   Parquet     Polars/Pandas    AstroLab Tensors   PyG Data    Cosmograph    Training

Simulations → HDF5/Parquet → SimulationTensor → 3D Graphs → Cosmic Web ML → Visualization
     ↓              ↓              ↓               ↓            ↓              ↓
   TNG50       Load Particles   Positions+Features  Spatial NN   Analysis      Cosmograph
```

## 💡 Pro Tips

- **Start small**: Use `max_samples=1000` for testing
- **Check dimensions**: Always verify tensor shapes before training  
- **Use caching**: Processed data is automatically cached
- **Graph building**: k-NN graphs work best with k=8-16 for astronomical data
- **Memory usage**: Large datasets (>100k objects) may need chunked processing
- **TNG50 efficiency**: Process one particle type at a time for large simulations
- **3D coordinates**: TNG50 uses comoving coordinates in ckpc/h units
- **Cosmic web**: Use multiple scales for comprehensive structure analysis
- **Visualization**: CosmographBridge provides interactive 3D exploration

## 🧪 Advanced Usage

### Custom Data Loading

```python
# Load with custom parameters
data = load_gaia_data(
    max_samples=50000,
    magnitude_limit=18.0,  # Fainter limit
    coordinate_range={"ra": [0, 90], "dec": [-30, 30]},  # Sky region
    cache_dir="./my_cache"
)

# TNG50 with specific settings
simulation = load_tng50_data(
    max_samples=25000,
    particle_type="PartType1",  # Dark matter
    return_tensor=True,  # SimulationTensor format
)
```

### Tensor Integration

```python
# Use native AstroLab tensors
from astro_lab.tensors import SurveyTensor, SimulationTensor

# Survey data
survey_data = SurveyTensor.from_survey_data(
    data=data,
    survey_name="gaia",
    coordinate_system="icrs"
)

# Simulation data (automatically creates SimulationTensor)
sim_data = load_tng50_data(particle_type="PartType0", return_tensor=True)

# → Access simulation metadata
print(f"Box size: {sim_data.box_size} Mpc/h")
print(f"Particle type: {sim_data.particle_type}")
print(f"Cosmology: {sim_data.cosmology}")

# → Automatically compatible model for different data types
model = create_gaia_classifier(survey_data)  # For surveys
model = create_cosmic_web_model(sim_data)    # For simulations
```

### Multi-Survey Analysis

```python
# Load and combine multiple datasets
gaia_stars = load_gaia_data(max_samples=10000)
sdss_galaxies = load_sdss_data(max_samples=5000) 
tng50_dm = load_tng50_data(max_samples=15000, particle_type="PartType1")

# Create unified analysis
all_positions = [
    gaia_stars.coordinates,      # 2D sky coordinates
    sdss_galaxies.coordinates,   # 2D sky coordinates  
    tng50_dm.positions          # 3D simulation coordinates
]

print("Multi-survey dataset ready for cross-correlation analysis!")
```

### Cosmic Web Pipeline

```python
# Complete cosmic web analysis pipeline
from astro_lab.data.core import create_cosmic_web_loader
from astro_lab.utils.viz import CosmographBridge

# 1. Load and analyze
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=2000,
    scales_mpc=[5.0, 10.0, 20.0]
)

# 2. Visualize interactively
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0,
    background_color='#000011'
)

# 3. Export results
import json
with open('cosmic_web_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("🌌 Complete cosmic web analysis pipeline!")
```

# AstroLab Data Loaders

Modern astronomical data management with clean, efficient APIs for all major surveys.

## Directory Structure Policy

AstroLab follows a **lazy directory creation** policy to avoid cluttering the filesystem:

- **Core directories** (`data/raw`, `data/processed`, `data/cache`, etc.) are only created when explicitly requested
- **Survey-specific directories** are only created when actually working with that survey
- **No automatic directory creation** on import - directories are created only when needed

### Directory Structure

```
data/
├── raw/                    # Original survey data
│   ├── gaia/              # Gaia DR3 catalogs
│   ├── sdss/              # SDSS DR17 data
│   ├── nsa/               # NASA Sloan Atlas
│   └── tng50/             # TNG50 simulation data
├── processed/             # Cleaned, ML-ready data
│   ├── gaia/              # Processed Gaia data
│   ├── sdss/              # Processed SDSS data
│   ├── nsa/               # Processed NSA data
│   └── tng50/             # Processed TNG50 data
├── cache/                 # Temporary cache files
├── experiments/           # MLflow and checkpoints
│   ├── mlruns/           # MLflow tracking
│   └── checkpoints/      # Lightning checkpoints
├── results/              # Organized model outputs
└── configs/              # Configuration files
```

## Preprocessing with Automatic Survey Organization

The preprocessing CLI now automatically detects survey types and saves processed data to the appropriate survey subdirectories:

```bash
# Process any catalog - automatically saves to survey-specific directory
astro-lab preprocess process data/raw/gaia_catalog.parquet

# This automatically:
# 1. Detects the survey type (Gaia)
# 2. Creates data/processed/gaia/ if needed
# 3. Saves processed data to data/processed/gaia/gaia_catalog_processed.parquet
# 4. Creates graphs in data/processed/gaia/
```

### Manual Directory Creation

If you need to create directories explicitly:

```python
from astro_lab.data.config import data_config

# Create core structure
data_config.setup_directories()

# Create survey-specific directories
data_config.ensure_survey_directories("gaia")
data_config.ensure_survey_directories("nsa")
```

## Quick Start

### Load Gaia Data

```python
from astro_lab.data import load_gaia_data

# Load Gaia data with automatic tensor conversion
dataset = load_gaia_data(max_samples=5000, return_tensor=True)
print(f"Loaded {len(dataset)} Gaia sources")
```

### Load SDSS Data

```python
from astro_lab.data import load_sdss_data

# Load SDSS galaxy data
dataset = load_sdss_data(max_samples=10000)
print(f"Loaded {len(dataset)} SDSS galaxies")
```

### Load NSA Data

```python
from astro_lab.data import load_nsa_data

# Load NSA galaxy catalog
dataset = load_nsa_data(max_samples=5000)
print(f"Loaded {len(dataset)} NSA galaxies")
```

### Load TNG50 Simulation Data

```python
from astro_lab.data import load_tng50_data, load_tng50_temporal_data

# Load static TNG50 data
static_data = load_tng50_data(max_particles=100000)

# Load temporal TNG50 data
temporal_data = load_tng50_temporal_data(
    max_particles=50000,
    time_steps=[0, 1, 2, 3, 4]
)
```

## Advanced Usage

### Custom Data Loading

```python
from astro_lab.data import AstroDataset

# Create custom dataset
dataset = AstroDataset(
    survey="gaia",
    data_path="path/to/custom/gaia_data.parquet",
    k_neighbors=12,
    max_samples=10000,
    return_tensor=True
)
```

### Graph Creation

```python
from astro_lab.data import create_graph_from_dataframe
import polars as pl

# Create sample data
df = pl.DataFrame({
    "ra": [0.0, 1.0, 2.0],
    "dec": [0.0, 1.0, 2.0],
    "mag": [10.0, 11.0, 12.0]
})

# Create graph
graph = create_graph_from_dataframe(
    df, 
    survey_type="gaia",
    k_neighbors=8,
    distance_threshold=50.0
)
```

### Training Splits

```python
from astro_lab.data import create_training_splits, save_splits_to_parquet

# Create splits
train_df, val_df, test_df = create_training_splits(
    df, 
    test_size=0.2, 
    val_size=0.1
)

# Save splits
save_splits_to_parquet(
    train_df, val_df, test_df,
    output_dir="data/processed/gaia",
    dataset_name="gaia_stellar"
)
```

## Survey-Specific Features

### Gaia DR3

- **Astrometry**: Proper motion, parallax, position
- **Photometry**: G, BP, RP magnitudes
- **Spectroscopy**: Teff, logg from GSP-Phot
- **Graph Features**: Spatial clustering, proper motion groups

### SDSS DR17

- **Photometry**: u, g, r, i, z magnitudes
- **Spectroscopy**: Redshift, spectral features
- **Morphology**: Galaxy classification
- **Graph Features**: Redshift-space clustering

### NSA

- **Photometry**: Multi-band photometry
- **Distances**: Redshift-independent distances
- **Morphology**: Galaxy properties
- **Graph Features**: 3D spatial clustering

### TNG50

- **Particle Data**: Dark matter, gas, stars
- **Temporal**: Multiple snapshots
- **Physics**: Hydrodynamics, feedback
- **Graph Features**: Particle clustering, merger trees

## Performance Optimizations

### Memory Management

```python
# Use smaller batches for large datasets
dataset = load_gaia_data(max_samples=100000, batch_size=1000)

# Enable tensor conversion for GPU acceleration
dataset = load_gaia_data(return_tensor=True)
```

### Caching

```python
# Enable caching for repeated access
from astro_lab.data.config import data_config
data_config.cache_dir.mkdir(exist_ok=True)
```

## CLI Commands

### Preprocessing

```bash
# Process catalog with automatic survey detection
astro-lab preprocess process data/raw/gaia_catalog.parquet

# Process with specific output location
astro-lab preprocess process data/raw/gaia_catalog.parquet --output data/processed/gaia/

# Create training splits
astro-lab preprocess process data/raw/gaia_catalog.parquet --create-splits

# Show statistics
astro-lab preprocess stats data/raw/gaia_catalog.parquet --verbose
```

### Data Management

```bash
# List available catalogs
astro-lab preprocess list

# Browse data directory
astro-lab preprocess browse data/processed

# Load and display splits
astro-lab preprocess splits data/processed/gaia gaia_stellar
```

### Cosmic Web Analysis

```bash
# Analyze Gaia cosmic web
astro-lab preprocess cosmic-web gaia --max-samples 10000

# Analyze with custom scales
astro-lab preprocess cosmic-web nsa --scales 5.0 10.0 20.0 50.0

# Save results
astro-lab preprocess cosmic-web linear --output results/cosmic_web/
```

## Integration with Training

### Lightning DataModule

```python
from astro_lab.data import create_astro_datamodule

# Create Lightning DataModule
datamodule = create_astro_datamodule(
    survey="gaia",
    batch_size=64,
    max_samples=50000,
    return_tensor=True
)

# Use with Lightning Trainer
trainer.fit(datamodule=datamodule)
```

### Custom Training Loop

```python
from astro_lab.data import create_astro_dataloader

# Create data loaders
train_loader, val_loader, test_loader = create_astro_dataloader(
    survey="gaia",
    batch_size=32,
    max_samples=10000
)

# Custom training loop
for batch in train_loader:
    # Process batch
    pass
```

## Error Handling

### Missing Data

```python
try:
    dataset = load_gaia_data()
except FileNotFoundError:
    print("Gaia data not found. Download first:")
    print("astro-lab download gaia")
```

### Memory Issues

```python
# Reduce batch size for large datasets
dataset = load_gaia_data(max_samples=1000000, batch_size=100)

# Use tensor conversion for GPU
dataset = load_gaia_data(return_tensor=True)
```

## Best Practices

1. **Use survey-specific loaders** for optimal performance
2. **Enable tensor conversion** for GPU acceleration
3. **Create training splits** early in the pipeline
4. **Save processed data** to avoid reprocessing
5. **Use appropriate batch sizes** for your hardware
6. **Monitor memory usage** with large datasets
7. **Cache frequently used data** in the cache directory

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed
```bash
uv sync
```

**Memory Issues**: Reduce batch size or max_samples
```python
dataset = load_gaia_data(max_samples=1000, batch_size=10)
```

**Slow Loading**: Enable caching and use tensor conversion
```python
dataset = load_gaia_data(return_tensor=True)
```

**Missing Data**: Download survey data first
```bash
astro-lab download gaia
```

### Performance Tips

1. Use `return_tensor=True` for GPU acceleration
2. Set appropriate `max_samples` for your use case
3. Use `batch_size` that fits in memory
4. Enable caching for repeated access
5. Use survey-specific optimizations

