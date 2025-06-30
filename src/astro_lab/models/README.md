# AstroLab Models - Astronomical Neural Networks

Core model implementations for astronomical data analysis with TensorDict integration.

## Core Components

### Base Components
- `EnhancedMLPBlock` - MLP with modern features
- `ModernGraphEncoder` - graph convolution encoder
- `AdvancedTemporalEncoder` - LSTM/GRU encoder with attention
- `ModernPointNetEncoder` - PointNet encoder with set attention

### TensorDict-Native Components
- `AstroTensorDictModule` - TensorDict wrapper for astronomical models
- `PhotometricEncoder` - Encoder for photometric data
- `SpectralEncoder` - Encoder for spectral data
- `SpatialAttention` - Attention mechanism for spatial data
- `MultiModalFusion` - Fusion of multiple data modalities
- `CosmicWebEncoder` - Specialized encoder for cosmic web analysis

### Output Heads
- `ClassificationHead` - Classification output head
- `RegressionHead` - Regression output head
- `create_output_head` - Factory function for output heads

### Factory Functions
- `create_tensordict_model` - Create TensorDict-native models
- `create_multimodal_model` - Create multi-modal models
- `create_cosmic_web_model` - Create cosmic web analysis models
- `create_survey_specific_model` - Create survey-specific models
- `create_analysis_pipeline` - Create comprehensive analysis pipelines

## Overview

The AstroLab models module has been refactored for simplicity and efficiency, following DRY principles and avoiding over-engineering.

## Key Features

### 1. Model Hierarchy

```
AstroBaseModel (Lightning + Mixins)
├── AstroGraphGNN     - Graph-level tasks
├── AstroNodeGNN      - Node-level tasks  
├── AstroTemporalGNN  - Time series tasks
├── AstroPointNet     - Point cloud tasks
└── AstroCosmicWebGNN - Cosmic web analysis
```

### 2. Unified Mixin System

All models use a consistent set of mixins from `components/mixins.py`:

**Core Functionality:**
- `MetricsMixin` - Pure PyTorch metrics computation
- `OptimizationMixin` - Advanced optimizer configurations
- `VisualizationMixin` - Model visualization capabilities

**HPO Optimization:**
- `HPOResetMixin` - Efficient parameter reset for model recycling
- `HPOMemoryMixin` - Memory optimization strategies
- `EfficientTrainingMixin` - Fast training configurations
- `ArchitectureSearchMixin` - Architecture configuration management

**Astronomical Domain:**
- `AstronomicalAugmentationMixin` - Domain-specific data augmentation
- `AstronomicalLossMixin` - Specialized loss functions

**Combined Mixins:**
- `StandardModelMixin` - Standard combination for basic models
- `HPOModelMixin` - Complete set for HPO-enabled models
- `AstronomicalModelMixin` - Domain-specific model features

### 3. Factory Pattern

```python
# Create by model type
model = create_model('graph', num_features=128, num_classes=7)

# Create by task
model = create_model_for_task('node_classification', num_features=128, num_classes=7)

# Create TensorDict model
model = create_tensordict_model(
    model_type='graph',
    in_keys=['coordinates', 'features'],
    out_keys=['predictions'],
    num_features=128,
    num_classes=7
)
```

## Usage Examples

### Basic Training

```python
from astro_lab.training import train_model

config = {
    'model_type': 'graph',
    'dataset': 'gaia',
    'batch_size': 32,
    'max_epochs': 100,
}

train_model(config)
```

### Hyperparameter Optimization

```python
from astro_lab.training import run_hpo

config = {
    'model_type': 'graph',
    'dataset': 'gaia',
    'task': 'graph_classification',
}

# Efficient HPO with model recycling
results = run_hpo(config, n_trials=100)
best_model = results['best_model']
```

### Manual Model Creation

```python
from astro_lab.models import AstroGraphGNN

model = AstroGraphGNN(
    num_features=128,
    num_classes=7,
    hidden_dim=256,
    num_layers=3,
    conv_type='gat',
    heads=4,
    pooling='mean',
    dropout=0.2
)
```

### Using Mixins Directly

```python
from astro_lab.models.core import AstroBaseModel
from astro_lab.models.components.mixins import (
    HPOModelMixin,
    AstronomicalModelMixin
)

class CustomAstroModel(AstroBaseModel, HPOModelMixin, AstronomicalModelMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
    
    def forward(self, batch):
        # Custom forward logic
        pass
```

## Model Types

1. **AstroGraphGNN**
   - Tasks: graph_classification, graph_regression, anomaly_detection
   - Features: Multiple conv types (GCN, GAT, SAGE, GIN)
   - Pooling: mean, max, sum, attention

2. **AstroNodeGNN**
   - Tasks: node_classification, node_regression, node_segmentation
   - Features: Same conv types as GraphGNN
   - Output: Per-node predictions

3. **AstroTemporalGNN**
   - Tasks: time_series_classification, forecasting
   - Features: LSTM/GRU/Transformer options
   - Input: Sequential data with graph structure

4. **AstroPointNet**
   - Tasks: point_classification, point_segmentation, point_registration
   - Features: PointNet++ architecture
   - Input: 3D point clouds

5. **AstroCosmicWebGNN**
   - Tasks: cosmic_web_classification, filament_detection
   - Features: Multi-scale analysis
   - Input: Large-scale spatial data

## Design Principles

1. **Simplicity**: Each model does one thing well
2. **DRY**: Shared functionality in unified mixin system
3. **Efficiency**: Model recycling for HPO saves memory and time
4. **Flexibility**: Easy to extend with new model types or mixins
5. **Lightning Integration**: All models are Lightning modules
6. **Pure PyTorch**: No external dependencies like sklearn

## Performance Tips

1. **HPO**: Use `run_hpo()` for automatic hyperparameter tuning
2. **GPU**: Models automatically use GPU if available
3. **Model Recycling**: Use `HPOResetMixin` for efficient HPO
4. **Mixed Precision**: Enable with `precision="16-mixed"`
5. **Gradient Checkpointing**: Use `HPOMemoryMixin.enable_gradient_checkpointing()`

## Architecture Details

### Mixin Organization

```
mixins.py
├── Core Functionality
│   ├── MetricsMixin       - Metrics computation
│   ├── OptimizationMixin  - Optimizer configs
│   └── VisualizationMixin - Visualization tools
├── HPO Optimization
│   ├── HPOResetMixin      - Parameter reset
│   ├── HPOMemoryMixin     - Memory optimization
│   ├── EfficientTrainingMixin - Fast training
│   └── ArchitectureSearchMixin - Architecture management
├── Astronomical Domain
│   ├── AstronomicalAugmentationMixin - Data augmentation
│   └── AstronomicalLossMixin - Domain losses
└── Combined Mixins
    ├── StandardModelMixin - Basic combination
    ├── HPOModelMixin      - HPO combination
    └── AstronomicalModelMixin - Domain combination
```

### Adding New Models

To add a new model type:

1. Create new class inheriting from `AstroBaseModel`
2. Select appropriate mixins
3. Implement `forward()` method
4. Register in `factory.py`
5. Add task mappings if needed

Example:
```python
from astro_lab.models.core import AstroBaseModel
from astro_lab.models.components.mixins import StandardModelMixin

class AstroNewModel(AstroBaseModel, StandardModelMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Model-specific layers
    
    def forward(self, batch):
        # Forward implementation
        pass
```

## Memory Optimization

For large-scale astronomical datasets:

1. Use `HPOMemoryMixin.optimize_memory_usage('aggressive')`
2. Enable gradient checkpointing
3. Use smaller batch sizes with gradient accumulation
4. Leverage model recycling during HPO
5. Clear CUDA cache between trials

## Future Extensions

The refactored architecture makes it easy to:
- Add new model architectures
- Create custom mixins for specific domains
- Integrate with new astronomical surveys
- Implement novel loss functions
- Add explainability features

The unified mixin system ensures consistency while allowing flexibility for astronomical-specific requirements.
