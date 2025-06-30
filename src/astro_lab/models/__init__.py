"""
AstroLab Models - Astronomical Neural Networks
===================================================

Core model implementations for astronomical data analysis.
"""

from .components import (
    ClassificationHead,
    MultiModalFusion,
    PeriodDetectionHead,
    PhotometricEncoder,
    RegressionHead,
    ShapeModelingHead,
    SpectralEncoder,
    TemporalEncoder,
)
from .core import (
    AstroBaseModel,
    AstroCosmicWebGNN,
    AstroGraphGNN,
    AstroNodeGNN,
    AstroPointNet,
    AstroTemporalGNN,
)

__all__ = [
    # Core models
    "AstroBaseModel",
    "AstroGraphGNN",
    "AstroNodeGNN",
    "AstroPointNet",
    "AstroTemporalGNN",
    "AstroCosmicWebGNN",
    # Components
    "ClassificationHead",
    "RegressionHead",
    "PeriodDetectionHead",
    "ShapeModelingHead",
    "PhotometricEncoder",
    "SpectralEncoder",
    "TemporalEncoder",
    "MultiModalFusion",
]

# Module metadata
__version__ = "3.0.0-tensordict"
if __doc__ is None:
    __doc__ = ""
__doc__ += """

Major Changes in v3.0 - TensorDict Native Integration:
=====================================================

1. **Native TensorDict Support**:
   - All models now work seamlessly with AstroTensorDict classes
   - TensorDictModule wrappers for existing PyTorch models
   - AstroTensorDictModule for astronomical-specific functionality
   - Direct integration with SpatialTensorDict, PhotometricTensorDict, etc.

2. **PyTorch Geometric Integration**:
   - Optimized for PyTorch Geometric 2.6+ features
   - Native support for torch.compile and TorchScript
   - Efficient batching and memory management
   - graph operations for cosmic web analysis

3. **Multi-Modal Model Support**:
   - MultiModalFusion for combining spatial, photometric, spectral data
   - Cross-modal attention mechanisms
   - Unified interfaces for different data modalities
   - Seamless integration with analysis workflows

4. **Cosmic Web Models**:
   - AstroCosmicWebGNN for large-scale structure analysis
   - SpatialAttention for astronomical coordinate systems
   - Scale-aware architectures for different cosmic scales
   - Proper astronomical distance and coordinate handling

5. **2025 Framework Integration**:
   - Full torch.compile support for performance
   - TensorDictSequential for complex pipelines
   - Memory-mapped tensor support for large datasets
   - GPU-optimized operations with CUDA 12.x support

Performance Features:
====================

1. **Memory Efficiency**: TensorDict enables zero-copy operations and memory sharing
2. **GPU Acceleration**: Native CUDA support with optimized kernels
3. **Compilation**: Full torch.compile support for 2x performance gains
4. **Batching**: Efficient batching of heterogeneous astronomical data
5. **Modularity**: Reusable components that can be combined flexibly

The enhanced models module provides state-of-the-art neural networks specifically
optimized for astronomical data while maintaining full backward compatibility
with existing AstroLab workflows.
"""
