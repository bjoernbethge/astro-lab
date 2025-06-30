# AstroLab Data Module Structure

## 📁 Directory Structure

```
src/astro_lab/data/
├── datamodules/           # Lightning DataModules
│   ├── __init__.py       # Unified API with create_datamodule()
│   ├── lightning.py      # PyG Lightning wrappers (recommended)
│   └── legacy.py         # Old DataModule (deprecated)
├── datasets/             # PyG Dataset implementations
│   ├── __init__.py
│   ├── base.py          # Base dataset class
│   ├── survey_graph_dataset.py    # Single large graph
│   └── point_cloud_dataset.py     # Multiple point clouds
├── preprocessors/        # Survey-specific preprocessing
│   ├── __init__.py
│   ├── base.py
│   └── [survey_name].py
├── utils/               # Utility functions
│   ├── __init__.py
│   └── clustering.py    # PyG-native clustering
├── analysis/            # Data analysis tools
├── collectors/          # Data collection helpers
├── graphs/             # Graph construction
└── loaders/            # Custom data loaders
```

## 🚀 Quick Start

### Unified API

```python
from astro_lab.data import create_datamodule

# Graph-level tasks (point clouds)
dm = create_datamodule(
    survey="gaia",
    task="graph",
    dataset_type="point_cloud",
    batch_size=32
)

# Node-level tasks (single graph)
dm = create_datamodule(
    survey="gaia",
    task="node",
    num_neighbors=[25, 10],
    batch_size=128
)
```

### Direct Dataset Usage

```python
from astro_lab.data.datasets import AstroPointCloudDataset

dataset = AstroPointCloudDataset(
    root="./data",
    survey="gaia",
    num_subgraphs=1000,
    points_per_subgraph=500
)
```

## 🔄 Migration Status

| Component | Status | Notes |
|-----------|--------|-------|
| **PyG Lightning Wrappers** | ✅ Active | Recommended approach |
| **Legacy DataModule** | ⚠️ Deprecated | Will be removed in v0.3.0 |
| **sklearn Dependencies** | ❌ Removed | Replaced with PyG utilities |
| **AstroSplitter** | ⚠️ Deprecated | Integrated into Lightning wrappers |

## 📊 Key Components

### DataModules

- **`create_datamodule()`**: Unified factory function
  - `backend="lightning"` (default): PyG Lightning wrappers
  - `backend="legacy"`: Old implementation (deprecated)

### Datasets

- **`SurveyGraphDataset`**: Single large graph for entire survey
- **`AstroPointCloudDataset`**: Multiple smaller graphs for batching

### Utilities

- **`clustering.py`**: PyG-native clustering functions
  - `create_pyg_kmeans()`: K-means without sklearn
  - `spatial_clustering_fps()`: FPS-based clustering
  - `hierarchical_fps_clustering()`: Multi-level clustering

## 🎯 Best Practices

1. **Always use the unified API**: `create_datamodule()`
2. **Prefer point cloud datasets** for graph-level tasks (better batching)
3. **Use neighbor sampling** for node-level tasks on large graphs
4. **No external dependencies**: Only PyTorch + PyG libraries

## ⚡ Performance Tips

- Use `persistent_workers=True` for DataLoaders
- Enable `pin_memory=True` for GPU training
- Adjust `num_subgraphs` and `points_per_subgraph` based on GPU memory
- Use FPS clustering for faster spatial splits

## 🔮 Future Plans

- [ ] Add link prediction support
- [ ] Implement temporal graph support
- [ ] Add heterogeneous graph handling
- [ ] Support for streaming/online learning
