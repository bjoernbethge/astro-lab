# Exoplanet Graph Dataset Pipeline

## Overview

The ExoplanetGraphDataset provides a PyTorch Geometric-based interface for NASA Exoplanet Archive data. It automatically downloads confirmed exoplanets and creates spatial graphs based on 3D stellar system positions for graph neural network training.

## 🌟 Key Features

### PyTorch Geometric Integration
- **InMemoryDataset** implementation for efficient graph processing
- **Automatic k-NN graph construction** based on 3D stellar positions
- **GPU-optimized** data loading and processing
- **Spatial relationships** between exoplanet host stars

### Data Source
- **NASA Exoplanet Archive** via astroquery
- **Table:** `ps` (Planetary Systems)
- **Filter:** `default_flag=1` (confirmed planets only)
- **Automatic download** with timeout protection

### Graph Construction
- **3D Cartesian coordinates** from RA/Dec/Distance
- **k-nearest neighbor** connections (default k=5)
- **Distance-based filtering** (default max 100 parsecs)
- **Edge weights** based on 3D distances

## 🚀 Usage

### Basic Dataset Creation

```python
from astro_lab.data import ExoplanetGraphDataset

# Create dataset with default settings
dataset = ExoplanetGraphDataset()

# Access the graph
graph = dataset[0]
print(f"Exoplanet graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
print(f"Node features: {graph.x.shape}")
print(f"3D positions: {graph.pos.shape}")
```

### Custom Configuration

```python
# Custom graph parameters
dataset = ExoplanetGraphDataset(
    k_neighbors=8,           # More connections
    max_distance=50.0,       # Closer systems only (parsecs)
    root="custom/path"       # Custom storage location
)
```

### DataLoader Integration

```python
from astro_lab.data import create_exoplanet_dataloader

# Create optimized DataLoader
loader = create_exoplanet_dataloader(
    k_neighbors=5,
    max_distance=100.0,
    batch_size=1,
    use_exoplanet_transforms=True
)

# Process graphs
for data in loader:
    # data.x: node features [N, F]
    # data.edge_index: graph connectivity [2, E]
    # data.pos: 3D positions [N, 3]
    # data.edge_attr: edge weights [E, 1]
    pass
```

### SurveyTensor Conversion

```python
# Convert to specialized tensor format (if available)
dataset = ExoplanetGraphDataset()
survey_tensor = dataset.to_survey_tensor()
spatial_tensor = dataset.get_spatial_tensor()
```

## 📊 Data Structure

### Node Features
The graph nodes represent exoplanet host stars with features:

```python
# Feature columns (in order):
features = [
    "ra",          # Right Ascension (degrees)
    "dec",         # Declination (degrees) 
    "sy_dist",     # System distance (parsecs)
    "pl_rade",     # Planet radius (Earth radii)
    "pl_masse",    # Planet mass (Earth masses)
    "disc_year",   # Discovery year
    "x",           # 3D Cartesian X coordinate
    "y",           # 3D Cartesian Y coordinate
    "z"            # 3D Cartesian Z coordinate
]
```

### Graph Properties
- **Nodes**: Each confirmed exoplanet system
- **Edges**: k-NN connections between nearby stellar systems
- **Edge weights**: 3D Euclidean distances in parsecs
- **Positions**: 3D Cartesian coordinates for spatial analysis

## 🔧 Technical Implementation

### Automatic Download

```python
# The dataset automatically downloads data on first use
dataset = ExoplanetGraphDataset()

# Output:
# 🪐 Downloading exoplanet data from NASA Exoplanet Archive...
# 📡 Querying NASA Exoplanet Archive...
# ✅ Downloaded X,XXX confirmed exoplanets
# 📁 Saved to: data/processed/exoplanet_graphs/raw/confirmed_exoplanets.parquet
```

### Graph Processing

```python
# Automatic processing creates graph structure
# 🔄 Processing Exoplanet catalog: confirmed_exoplanets.parquet
# 📊 Using all X,XXX available planets
# Creating graph with X,XXX exoplanets...
# Graph created: X,XXX nodes, X,XXX edges
```

### Storage Structure

```
data/processed/exoplanet_graphs/
├── raw/
│   └── confirmed_exoplanets.parquet     # Raw NASA data
└── processed/
    └── exoplanet_graph_k5_all.pt       # Processed graph
```

## 🎯 Machine Learning Applications

### Graph Neural Networks

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ExoplanetGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return self.classifier(x)

# Train on exoplanet graph
dataset = ExoplanetGraphDataset()
model = ExoplanetGNN(dataset[0].x.shape[1])
```

### Use Cases

1. **Discovery Method Prediction**
   - Predict how exoplanets were discovered based on system properties
   - Classification task using host star features

2. **System Clustering**
   - Group similar exoplanet systems
   - Identify patterns in stellar neighborhoods

3. **Planet Property Estimation**
   - Predict planet properties from stellar environment
   - Regression on planet radius/mass

4. **Habitability Assessment**
   - Classify potentially habitable systems
   - Use spatial and physical features

## 🔍 Advanced Usage

### Custom Transforms

```python
from astro_lab.data import get_exoplanet_transforms

# Apply specialized transforms
transforms = get_exoplanet_transforms()
dataset = ExoplanetGraphDataset(transform=transforms)
```

### Data Filtering

```python
# Filter by distance
def close_systems_filter(data):
    # Only keep systems within 50 parsecs
    return data.pos.norm(dim=1) < 50.0

dataset = ExoplanetGraphDataset(pre_filter=close_systems_filter)
```

### Batch Processing

```python
from torch_geometric.loader import DataLoader

# Process multiple subgraphs
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    # batch contains multiple graphs
    # Use torch_geometric.data.Batch methods
    pass
```

## 📈 Performance Characteristics

### Download Performance
- **Data Source**: NASA Exoplanet Archive
- **Download Time**: ~10-15 seconds
- **File Size**: ~1-2 MB (compressed Parquet)
- **Update Frequency**: On-demand (cached locally)

### Graph Processing
- **Processing Time**: ~5-10 seconds
- **Memory Usage**: ~50-100 MB
- **Graph Size**: ~5,000-6,000 nodes (confirmed planets)
- **Edge Count**: Variable (depends on k_neighbors and max_distance)

### Storage Requirements
- **Raw Data**: ~2 MB
- **Processed Graph**: ~20 MB
- **Total**: ~25 MB

## ⚠️ Known Issues

### NASA API Limitations
- **Timeout Issues**: NASA Exoplanet Archive occasionally times out
- **Solution**: Automatic retry with exponential backoff
- **Fallback**: Cached data used if download fails

### Data Quality
- **Missing Values**: Filled with 0.0 (using `np.nan_to_num`)
- **Distance Estimates**: Some systems lack distance measurements
- **Coordinate Precision**: Limited by catalog accuracy

## 🛠️ Development Commands

```bash
# Test dataset creation
uv run python -c "from astro_lab.data import ExoplanetGraphDataset; d = ExoplanetGraphDataset(); print(f'Graph: {d[0].num_nodes} nodes')"

# Test DataLoader
uv run python -c "from astro_lab.data import create_exoplanet_dataloader; l = create_exoplanet_dataloader(); print('DataLoader created')"

# Check dataset info
uv run python scripts/check_datasets.py

# Test with astroquery
uv run python examples/astroquery_demo.py
```

## 🧪 Testing

The exoplanet dataset is covered by the test suite:

```bash
# Run dataset tests
uv run pytest test/test_*.py -v -k exoplanet

# Test data loading
uv run pytest test/test_data_module.py -v
```

## 🔗 Integration

### With Other Datasets

```python
# Combine with other astronomical datasets
from astro_lab.data import GaiaGraphDataset, NSAGraphDataset

gaia_data = GaiaGraphDataset()
nsa_data = NSAGraphDataset()
exo_data = ExoplanetGraphDataset()

# Cross-reference stellar systems
```

### With Training Pipeline

```python
# Use in training loop
from astro_lab.training import LightningModule

class ExoplanetModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.dataset = ExoplanetGraphDataset()
        # ... model definition
```

## 🔮 Future Extensions

### Planned Features
- **Multi-planet systems**: Better representation of systems with multiple planets
- **Temporal evolution**: Track discovery trends over time
- **Cross-matching**: Link with Gaia stellar data
- **Advanced filtering**: More sophisticated data quality filters

### Research Directions
- **Exoplanet population synthesis**
- **Discovery bias analysis**
- **Habitability zone modeling**
- **System architecture studies**

## 📚 References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [astroquery Documentation](https://astroquery.readthedocs.io/)

---

**Status:** ✅ Production Ready  
**Implementation:** PyTorch Geometric InMemoryDataset  
**Data Source:** NASA Exoplanet Archive via astroquery  
**Graph Type:** 3D spatial k-NN graph  
**Testing:** ✅ Integrated with test suite  
**GPU Support:** ✅ Automatic optimization 