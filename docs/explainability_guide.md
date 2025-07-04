# Explainability Guide for AstroLab Models

## Overview

AstroLab provides comprehensive explainability and interpretability features for astronomical models, particularly for Graph Neural Networks (GNNs) using PyTorch Geometric. This guide covers the available methods and how to use them.

## Available Explainability Methods

### 1. Attention Weight Analysis

**Method**: `extract_attention_weights()`
**Use Case**: Understanding which nodes/edges are most important in GAT/Transformer models

```python
from astro_lab.models.components.mixins import ExplainabilityMixin

class MyAstronomicalModel(ExplainabilityMixin, ...):
    pass

model = MyAstronomicalModel()
attention_data = model.extract_attention_weights(batch, layer_idx=-1)

# Access attention information
node_importance = attention_data["node_importance"]
edge_attention = attention_data["edge_attention"]
```

**Supported Layers**:
- `GATConv` (Graph Attention Networks)
- `TransformerConv` (Transformer Convolution)

### 2. Gradient-Based Feature Importance

**Method**: `get_feature_importance()`
**Use Case**: Understanding which input features most influence predictions

```python
feature_importance = model.get_feature_importance(batch)
# Returns tensor of importance scores for each feature
```

### 3. Saliency Maps

**Method**: `saliency_map()`
**Use Case**: Pixel-level attribution for understanding model decisions

```python
saliency_data = model.saliency_map(batch, target_class=0)
saliency_map = saliency_data["saliency"]
feature_importance = saliency_data["feature_importance"]
```

### 4. Integrated Gradients

**Method**: `integrated_gradients()`
**Use Case**: More robust feature attribution using path integration

```python
ig_data = model.integrated_gradients(
    batch, 
    target_class=0, 
    steps=50,
    baseline=None  # Optional baseline tensor
)

attributions = ig_data["attributions"]
feature_importance = ig_data["feature_importance"]
```

### 5. Astronomical Feature Importance

**Method**: `astronomical_feature_importance()`
**Use Case**: Domain-specific importance analysis for astronomical data

```python
astro_importance = model.astronomical_feature_importance(batch)

# Access specific astronomical features
coordinate_importance = astro_importance["astronomical_breakdown"]["coordinates"]
photometry_importance = astro_importance["astronomical_breakdown"]["photometry"]

# Most important features
most_important_coord = astro_importance["most_important_coordinate"]
most_important_mag = astro_importance["most_important_magnitude"]
```

### 6. Comprehensive Prediction Explanation

**Method**: `explain_prediction()`
**Use Case**: Get all explanation methods in one call

```python
explanation = model.explain_prediction(batch, target_class=0)

# Access all explanation data
prediction = explanation["prediction"]
confidence = explanation["confidence"]
feature_importance = explanation["feature_importance"]
attention_weights = explanation["attention_weights"]
saliency = explanation["saliency"]
integrated_gradients = explanation["integrated_gradients"]
astronomical_importance = explanation["astronomical_importance"]
```

## Model Zoo Integration

### Using Explainability Mixins

AstroLab provides several pre-configured mixin combinations:

```python
from astro_lab.models.components.mixins import (
    ExplainabilityMixin,
    ExplainableModelMixin,
    ResearchModelMixin,
    FullAstronomicalModelMixin
)

# For explainability-focused models
class MyExplainableModel(ExplainableModelMixin, ...):
    pass

# For research models with explainability
class MyResearchModel(ResearchModelMixin, ...):
    pass

# For full-featured astronomical models
class MyFullModel(FullAstronomicalModelMixin, ...):
    pass
```

### Supported Model Types

The explainability features work with all AstroLab models that:
- Inherit from the explainability mixins
- Use PyTorch Geometric data structures (`Data`, `Batch`, `HeteroData`)
- Have attention mechanisms (GAT, Transformer) for attention analysis

## Astronomical Domain Specifics

### Coordinate System Analysis

AstroLab automatically interprets the first 3 features as astronomical coordinates:
- Feature 0: Right Ascension (RA)
- Feature 1: Declination (Dec)  
- Feature 2: Distance/Parallax

### Photometric Analysis

Features 3-7 are typically interpreted as photometric magnitudes:
- Feature 3: First magnitude band
- Feature 4: Second magnitude band
- etc.

### Custom Feature Mapping

You can customize feature interpretation by overriding the `astronomical_feature_importance()` method:

```python
def astronomical_feature_importance(self, batch):
    # Custom implementation for your specific feature layout
    pass
```

## Visualization Integration

### Creating Visualization Data

```python
viz_data = model.create_explanation_visualization_data(batch)

# Access visualization-ready data
predictions = viz_data["predictions"]
feature_importance = viz_data["feature_importance"]
attention = viz_data["attention"]
attributions = viz_data["attributions"]
```

### Integration with AstroLab Widgets

The explainability data can be directly used with AstroLab's visualization widgets:

```python
from astro_lab.widgets import alpv, alo3d

# Create plots with explanation data
plotter = alpv.AstronomicalPlotter()
plotter.plot_feature_importance(feature_importance)
plotter.plot_attention_weights(attention_data)

# 3D visualization with node importance
visualizer = alo3d.AstronomicalVisualizer()
visualizer.visualize_with_importance(
    coordinates, 
    node_importance=attention_data["node_importance"]
)
```

## Best Practices

### 1. Model Architecture Considerations

- Use attention mechanisms (GAT, Transformer) for attention analysis
- Ensure gradients are enabled for gradient-based methods
- Consider model complexity vs. explainability trade-offs

### 2. Performance Optimization

- Use `torch.no_grad()` for inference-only operations
- Batch explanations when possible
- Cache attention weights for repeated analysis

### 3. Astronomical Context

- Validate coordinate ranges (RA: 0-360°, Dec: -90° to 90°)
- Check magnitude reasonableness (-5 to 50 mag)
- Consider redshift effects for cosmological data

### 4. Interpretation Guidelines

- **Attention Weights**: Higher weights indicate more important relationships
- **Feature Importance**: Higher values indicate more influential features
- **Saliency Maps**: Brighter regions indicate more important input areas
- **Integrated Gradients**: More robust than simple gradients

## Example Usage

```python
import torch
from astro_lab.models.components.mixins import ExplainableModelMixin
from astro_lab.data import create_sample_batch

class MyAstronomicalGNN(ExplainableModelMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Your model architecture here
        pass
    
    def forward(self, batch):
        # Your forward pass here
        pass

# Create model and sample data
model = MyAstronomicalGNN()
batch = create_sample_batch()

# Get comprehensive explanation
explanation = model.explain_prediction(batch, target_class=0)

# Analyze astronomical features
astro_importance = explanation["astronomical_importance"]
print(f"Most important coordinate: {astro_importance['most_important_coordinate']}")
print(f"Most important magnitude: {astro_importance['most_important_magnitude']}")

# Visualize attention
attention_data = explanation["attention_weights"]
if attention_data["node_importance"] is not None:
    print("Node importance scores:", attention_data["node_importance"])
```

## Troubleshooting

### Common Issues

1. **No attention layers found**: Ensure your model uses GAT or Transformer layers
2. **Gradient computation errors**: Check that inputs require gradients
3. **Memory issues**: Reduce batch size or use gradient checkpointing
4. **Feature interpretation errors**: Verify your feature layout matches expectations

### Debugging Tips

```python
# Check if model has attention layers
attention_layers = [name for name, module in model.named_modules() 
                   if isinstance(module, (GATConv, TransformerConv))]
print("Attention layers:", attention_layers)

# Verify gradient computation
batch.x.requires_grad_(True)
print("Gradients enabled:", batch.x.requires_grad)

# Check feature dimensions
print("Feature dimensions:", batch.x.shape)
```

## Future Enhancements

Planned improvements include:
- SHAP integration for model-agnostic explanations
- LIME-style local explanations
- Counterfactual analysis
- Astronomical-specific baseline generation
- Real-time explanation streaming
- Integration with external explainability libraries 