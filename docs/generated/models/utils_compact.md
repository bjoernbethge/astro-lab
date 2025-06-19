# Utils Module

Auto-generated documentation for `models.utils`

## Functions

### compile_astro_model(model: Any, mode: str = 'default', dynamic: bool = True) -> Any

Compile astronomical model for PyTorch 2.x.

Args:
    model: Model to compile
    mode: Compilation mode
    dynamic: Enable dynamic shapes

Returns:
    Compiled model

### count_parameters(model: torch.nn.modules.module.Module) -> int

Count trainable parameters in a model.

Args:
    model: PyTorch model

Returns:
    Number of trainable parameters

### create_asteroid_period_detector(hidden_dim: int = 96, **kwargs) -> Any

Create asteroid rotation period detector using lightcurve data.

### create_astrophot_model(model_type: str = 'sersic+disk', hidden_dim: int = 128, **kwargs) -> Any

Create AstroPhot-integrated model for galaxy modeling.

### create_gaia_classifier(hidden_dim: int = 128, num_classes: int = 7, **kwargs) -> Any

Create Gaia stellar classifier.

### create_lightcurve_classifier(hidden_dim: int = 128, output_dim: int = 1, **kwargs) -> Any

Create lightcurve/ALCDEF classifier with LightcurveEncoder.

### create_lsst_transient_detector(hidden_dim: int = 96, **kwargs) -> Any

Create LSST transient detection model.

### create_multi_survey_model(surveys: List[str], hidden_dim: int = 256, output_dim: int = 1, **kwargs) -> Any

Create model for multiple surveys.

### create_nsa_galaxy_modeler(hidden_dim: int = 128, **kwargs) -> Any

Create NSA galaxy modeler with full component set.

### create_sdss_galaxy_classifier(hidden_dim: int = 128, output_dim: int = 1, **kwargs) -> Any

Create SDSS galaxy property predictor.

### get_activation(name: str) -> torch.nn.modules.module.Module

Get activation function by name.

Args:
    name: Activation function name

Returns:
    PyTorch activation module

### get_pooling(name: str, hidden_dim: Optional[int] = None) -> Callable

Get pooling function by name.

Args:
    name: Pooling function name
    hidden_dim: Hidden dimension (needed for attention pooling)

Returns:
    Pooling function

### initialize_weights(module: torch.nn.modules.module.Module)

Initialize model weights using Xavier/Kaiming initialization.

Args:
    module: PyTorch module to initialize

### model_summary(model: torch.nn.modules.module.Module) -> dict

Get model summary information.

Args:
    model: PyTorch model

Returns:
    Dictionary with model information

## Classes

### AttentionPooling

Attention-based global pooling layer for graphs.

#### Methods

**`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**

Apply attention pooling.

Args:
x: Node features [num_nodes, hidden_dim]
batch: Batch assignment [num_nodes]

Returns:
Pooled features [batch_size, hidden_dim]
