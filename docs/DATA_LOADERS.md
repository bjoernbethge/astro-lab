# Loading & Processing Astronomical Data

## 🎯 What does this module do?

The `astro_lab.data` module loads astronomical catalogs (stars, galaxies) **and cosmological simulations** and prepares them for machine learning. **Simple, fast, without complexity.**

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

### 2️⃣ Inspect Data

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

### 3️⃣ Processing for ML

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

### 4️⃣ For Graph Neural Networks

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
| **Gaia** | Stars | 5k-50k | Position, brightness, motion | Stellar Classification |
| **SDSS** | Galaxies | 1k-10k | Colors, redshift, morphology | Galaxy Classification |
| **NSA** | Galaxies | 1k-5k | Sérsic profiles, mass | Galaxy Evolution |
| **TNG50** | Simulation | 1k-20k per type | 3D positions, masses, velocities | Cosmic Web, Dark Matter |
| **LINEAR** | Asteroids | 500-2k | Light curves, periods | Variable Stars |

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
4. **Tensor Integration**: Native AstroLab tensor support for advanced workflows
5. **Caching**: Processed data is cached for faster subsequent loads
6. **Simulation Loading**: TNG50 Parquet files are loaded with simulation metadata

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
```

## 🔄 Data Flow Architecture

```
Raw Catalogs → Preprocessing → Feature Engineering → Graph Building → ML Ready
     ↓              ↓              ↓                ↓            ↓
   Parquet     Polars/Pandas    AstroLab Tensors   PyG Data    Training

Simulations → HDF5/Parquet → SimulationTensor → 3D Graphs → Cosmic Web ML
     ↓              ↓              ↓               ↓            ↓
   TNG50       Load Particles   Positions+Features  Spatial NN   Training
```

## 💡 Pro Tips

- **Start small**: Use `max_samples=1000` for testing
- **Check dimensions**: Always verify tensor shapes before training  
- **Use caching**: Processed data is automatically cached
- **Graph building**: k-NN graphs work best with k=8-16 for astronomical data
- **Memory usage**: Large datasets (>100k objects) may need chunked processing
- **TNG50 efficiency**: Process one particle type at a time for large simulations
- **3D coordinates**: TNG50 uses comoving coordinates in ckpc/h units

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

