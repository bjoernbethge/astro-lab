"""
Astronomical Graph Construction - PyTorch Geometric
==============================================================

graph building tools using latest PyG transforms and eliminating
legacy torch_cluster dependencies for astronomical data processing.
"""

from .base import AstronomicalGraphBuilder
from .pointcloud import (
    PointCloudGraphBuilder,
    create_multiscale_pointcloud_graph,
    create_pointcloud_graph,
)

__all__ = [
    # graph building
    "AstronomicalGraphBuilder",
    # point cloud processing
    "PointCloudGraphBuilder",
    "create_pointcloud_graph",
    "create_multiscale_pointcloud_graph",
]

# Version and metadata
__version__ = "2.0.0"
__author__ = "AstroLab Team"

# Module documentation
if __doc__ is None:
    __doc__ = ""
__doc__ += """

Graph Construction Features (2025)
========================================

The graphs module provides cutting-edge tools for building graphs from astronomical
data using PyTorch Geometric  transforms:

## 🚀 Features:

1. **Native PyG Transform Integration**:
   - KNNGraph and RadiusGraph transforms
   - GridSampling for hierarchical clustering
   - ToUndirected and AddSelfLoops transforms
   - Compose for chaining multiple transforms

2. **Connectivity Methods**:
   - Native PyG k-NN graphs with KNNGraph transform
   - Radius graphs with RadiusGraph transform
   - Adaptive graphs using transform combinations
   - Cosmic web graphs with multi-scale processing

3. **Point Cloud Processing**:
   - GridSampling for structure-preserving downsampling
   - RandomSample for uniform sampling
   - density-based feature extraction
   - Multi-scale structure analysis

4. **Transform-Based Pipeline**:
   - Compose multiple transforms seamlessly
   - Standard PyG transform interface
   - Memory-efficient batch processing
   - Device-aware computations

## ✨ Key Improvements in v2.0.0:

- ✅ **Zero torch_cluster dependencies**
- ✅ **Native PyG transform usage**
- ✅ **GridSampling** for hierarchical processing
- ✅ **KNNGraph/RadiusGraph transforms**
- ✅ **Compose pipeline** for complex workflows
- ✅ **Memory-efficient** processing
- ✅ **PyTorch Geometric ** compatibility
"""
