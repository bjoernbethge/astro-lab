# Loading & Processing Astronomical Data

## 🎯 What does this module do?

The `astro_lab.data` module loads astronomical catalogs (stars, galaxies) and prepares them for machine learning. **Simple, fast, without complexity.**

## 📋 Complete Workflow

### 1️⃣ Load Data (30 seconds)

```python
from astro_lab.data import load_gaia_data, load_sdss_data

# Load Gaia stars
stars = load_gaia_data(max_samples=5000)
print(f"✅ {stars.shape[0]} stars loaded")

# Load SDSS galaxies  
galaxies = load_sdss_data(max_samples=2000)
print(f"✅ {galaxies.shape[0]} galaxies loaded")
```

### 2️⃣ Inspect Data

```python
# What's inside?
print(f"Gaia Features: {stars.photometric_bands}")  # ['G', 'BP', 'RP']
print(f"SDSS Features: {galaxies.photometric_bands}")  # ['u', 'g', 'r', 'i', 'z']

# First rows
print(stars.data[:3])  # First 3 stars
print(galaxies.data[:3])  # First 3 galaxies
```

### 3️⃣ Processing for ML

```python
# For PyTorch training
import torch
from torch.utils.data import DataLoader

# As regular PyTorch tensor
X = stars.data  # [5000, 8] - 5000 stars, 8 features
y = torch.randint(0, 3, (5000,))  # Dummy labels

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

### Recipe 3: Lightning Training

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
```

### Problem: "Wrong feature dimensions"
```python
# Solution: Check shape
data = load_gaia_data(max_samples=100)
print(f"Shape: {data.shape}")  # [100, 8]
print(f"Features: {data.column_mapping}")  # Which column is what
```

## 📊 What happens internally?

1. **Auto-Download**: If no data exists, synthetic astronomical data is generated
2. **Preprocessing**: Coordinates are converted, missing values handled
3. **Graph Creation**: k-NN graphs are built for spatial relationships
4. **Tensor Integration**: Native AstroLab tensor support for advanced workflows
5. **Caching**: Processed data is cached for faster subsequent loads

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
```

## 🔄 Data Flow Architecture

```
Raw Catalogs → Preprocessing → Feature Engineering → Graph Building → ML Ready
     ↓              ↓              ↓                ↓            ↓
   Parquet     Polars/Pandas    AstroLab Tensors   PyG Data    Training
```

## 💡 Pro Tips

- **Start small**: Use `max_samples=1000` for testing
- **Check dimensions**: Always verify tensor shapes before training
- **Use caching**: Processed data is automatically cached
- **Graph building**: k-NN graphs work best with k=8-16 for astronomical data
- **Memory usage**: Large datasets (>100k objects) may need chunked processing

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
```

### Tensor Integration

```python
# Use native AstroLab tensors
from astro_lab.tensors import SurveyTensor

survey_data = SurveyTensor.from_survey_data(
    data=data,
    survey_name="gaia",
    coordinate_system="icrs"
)

# → Automatically compatible model for Gaia data
model = create_gaia_classifier(survey_data)
```

## 📖 Related Documentation

- [Training Guide](TRAINING.md) - Model training workflows
- [Tensor System](TENSORS.md) - Advanced tensor operations
- [Graph Networks](GRAPHS.md) - Graph neural network specifics 