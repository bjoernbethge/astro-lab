# 🎨 Cosmograph Integration Guide

Interactive 3D visualization for astronomical data using CosmographBridge with Polars for efficient data handling.

## 🚀 Quick Start

```python
from astro_lab.utils.viz import CosmographBridge
import polars as pl

# Create bridge
bridge = CosmographBridge()

# Create visualization from Polars DataFrame
df = pl.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [1, 2, 3, 4, 5],
    'z': [1, 2, 3, 4, 5]
})

viz = bridge.from_polars_dataframe(df, radius=2.0)
```

## 🌟 Key Features

### Interactive 3D Visualization
- **Real-time physics**: Gravity and repulsion simulation
- **Survey-specific colors**: Gold for stars, blue for galaxies, green for simulations
- **Multi-scale exploration**: Zoom from stellar to cosmological scales
- **Interactive controls**: Rotate, zoom, and explore data in 3D

### Seamless Integration
- **Direct from cosmic web analysis**: No intermediate steps required
- **Automatic color mapping**: Survey-specific visual themes
- **Real-time updates**: Dynamic visualization as data changes
- **Jupyter compatibility**: Works in notebooks and lab environments

### Polars Integration
- **Efficient data handling**: Use Polars DataFrames for data processing
- **Automatic graph creation**: Neighbor graphs based on spatial proximity

## 🔧 Basic Usage

### Single Survey Visualization
```python
from astro_lab.utils.viz import CosmographBridge

# Create bridge instance
bridge = CosmographBridge()

# Load and visualize Gaia data
results = create_cosmic_web_loader(survey="gaia", max_samples=1000)
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=3.0,
    background_color='#000011'
)
```

### Multi-Survey Comparison
```python
# Compare different surveys
surveys = ["gaia", "sdss", "nsa", "tng50"]
widgets = []

for survey in surveys:
    results = create_cosmic_web_loader(survey=survey, max_samples=500)
    widget = bridge.from_cosmic_web_results(results, survey_name=survey)
    widgets.append(widget)

# Display all widgets
for i, widget in enumerate(widgets):
    print(f"Survey: {surveys[i]}")
    display(widget)
```

## 🎨 Customization Options

### Visual Styling
```python
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=5.0,                    # Node radius
    background_color='#000011',    # Dark blue background
    node_color='#FFD700',          # Gold nodes for stars
    edge_color='#444444',          # Gray edges
    edge_width=0.5                 # Edge thickness
)
```

### Physics Simulation
```python
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    physics_enabled=True,          # Enable physics simulation
    gravity_strength=0.1,          # Gravity force
    repulsion_strength=0.05,       # Repulsion force
    damping_factor=0.8             # Motion damping
)
```

### Layout Options
```python
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    layout_type="force_directed",  # Force-directed layout
    # layout_type="circular",      # Circular layout
    # layout_type="random",        # Random layout
    layout_iterations=100          # Layout optimization iterations
)
```

## 🌌 Survey-Specific Visualizations

### Stellar Data (Gaia)
```python
# Gold theme for stellar data
results = create_cosmic_web_loader(survey="gaia", max_samples=1000)
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    node_color='#FFD700',          # Gold stars
    background_color='#000011',    # Dark space
    radius=2.0                     # Smaller nodes for stars
)
```

### Galaxy Data (SDSS/NSA)
```python
# Blue theme for galaxy data
results = create_cosmic_web_loader(survey="sdss", max_samples=500)
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="sdss",
    node_color='#4169E1',          # Royal blue galaxies
    background_color='#000011',    # Dark space
    radius=4.0                     # Larger nodes for galaxies
)
```

### Simulation Data (TNG50)
```python
# Green theme for simulation data
results = create_cosmic_web_loader(survey="tng50", max_samples=300)
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="tng50",
    node_color='#32CD32',          # Lime green simulation
    background_color='#000011',    # Dark space
    radius=3.0                     # Medium nodes
)
```

## 🔄 Advanced Usage

### Custom Data Integration
```python
from astro_lab.tensors import Spatial3DTensor

# Create custom spatial tensor
spatial_tensor = Spatial3DTensor.from_catalog_data({
    "RA": ra_data,
    "DEC": dec_data,
    "DISTANCE": distance_data
})

# Convert to cosmic web format
results = {
    "positions": spatial_tensor.get_coordinates_cartesian(),
    "n_objects": len(ra_data),
    "total_volume": spatial_tensor.calculate_volume(),
    "clusters": spatial_tensor.find_clusters(radius=10.0)
}

# Visualize custom data
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="custom",
    node_color='#FF6B6B',          # Custom color
    radius=2.5
)
```

### Real-time Updates
```python
import time

# Create initial visualization
results = create_cosmic_web_loader(survey="gaia", max_samples=100)
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")

# Update data in real-time
for i in range(10):
    # Update cosmic web analysis
    new_results = create_cosmic_web_loader(
        survey="gaia", 
        max_samples=100 + i * 50
    )
    
    # Update visualization
    widget.update_data(new_results)
    time.sleep(1)  # Wait 1 second between updates
```

## 🎯 Use Cases

### Stellar Cluster Analysis
```python
# Analyze stellar clusters in Gaia data
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=2000,
    scales_mpc=[1.0, 2.0, 5.0]  # Small scales for clusters
)

widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    radius=1.5,
    physics_enabled=True,
    gravity_strength=0.2
)
```

### Galaxy Distribution Analysis
```python
# Analyze large-scale galaxy distribution
results = create_cosmic_web_loader(
    survey="sdss",
    max_samples=1000,
    scales_mpc=[10.0, 20.0, 50.0]  # Large scales for galaxies
)

widget = bridge.from_cosmic_web_results(
    results,
    survey_name="sdss",
    radius=5.0,
    layout_type="force_directed",
    layout_iterations=200
)
```

### Exoplanet Spatial Analysis
```python
# Analyze exoplanet spatial distribution
results = create_cosmic_web_loader(
    survey="exoplanets",
    max_samples=500,
    scales_mpc=[0.1, 0.5, 1.0]  # Very small scales for exoplanets
)

widget = bridge.from_cosmic_web_results(
    results,
    survey_name="exoplanets",
    node_color='#FF4500',          # Orange for exoplanets
    radius=2.0,
    physics_enabled=True
)
```

## 🔧 Configuration

### Bridge Configuration
```python
# Configure bridge with custom settings
bridge = CosmographBridge(
    default_radius=3.0,
    default_background_color='#000011',
    physics_enabled=True,
    layout_type="force_directed"
)
```

### Widget Configuration
```python
# Configure widget behavior
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    auto_play=True,                # Auto-start physics
    show_controls=True,            # Show control panel
    show_info=True,                # Show data info
    responsive=True                # Responsive to window size
)
```

## 🔗 Integration with Other Tools

### Jupyter Integration
```python
# In Jupyter notebook
from IPython.display import display

widget = bridge.from_cosmic_web_results(results, survey_name="gaia")
display(widget)

# Save as HTML
widget.save_html("cosmic_web_visualization.html")
```

### Blender Integration
```python
from astro_lab.utils.blender import export_to_blender

# Export cosmic web data to Blender
export_to_blender(
    results,
    output_file="cosmic_web.blend",
    node_material="emission",
    edge_material="wire"
)
```

## 🐛 Troubleshooting

### Common Issues
```python
# Check if cosmograph is available
try:
    import cosmograph
    print("Cosmograph available")
except ImportError:
    print("Install cosmograph: pip install cosmograph")

# Check data format
if "positions" not in results:
    print("Results must contain 'positions' key")

# Check widget creation
try:
    widget = bridge.from_cosmic_web_results(results, survey_name="gaia")
    print("Widget created successfully")
except Exception as e:
    print(f"Error creating widget: {e}")
```

### Performance Optimization
```python
# Reduce data size for better performance
results = create_cosmic_web_loader(
    survey="gaia",
    max_samples=500,  # Reduce for better performance
    return_tensor=False  # Disable tensor operations
)

# Use simpler physics for large datasets
widget = bridge.from_cosmic_web_results(
    results,
    survey_name="gaia",
    physics_enabled=False,  # Disable for large datasets
    layout_iterations=50    # Reduce iterations
)
```

## 🔗 Related Documentation

- **[Data Loaders](docs/DATA_LOADERS.md)**: Loading astronomical data
- **[Gaia Cosmic Web](docs/GAIA_COSMIC_WEB.md)**: Stellar structure analysis
- **[Development Guide](docs/DEVGUIDE.md)**: Contributing to the project

## 🎯 Examples

### Complete Workflow
```python
# 1. Load data
from astro_lab.data.core import create_cosmic_web_loader
results = create_cosmic_web_loader(survey="gaia", max_samples=1000)

# 2. Create visualization
from astro_lab.utils.viz import CosmographBridge
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")

# 3. Customize appearance
widget.set_node_color('#FFD700')  # Gold stars
widget.set_background_color('#000011')  # Dark space

# 4. Enable physics
widget.enable_physics(gravity_strength=0.1, repulsion_strength=0.05)

# 5. Display
display(widget)
```

---

**Ready to explore the cosmos?** Start with [Data Loading](docs/DATA_LOADERS.md) or dive into [Machine Learning Training](../README.md#training-workflow)! 