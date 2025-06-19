# Tgnn Module

Auto-generated documentation for `models.tgnn`

## LightcurveTensor

Tensor for astronomical time series and lightcurve data.

Handles time-dependent photometric measurements with physical
properties like periods, amplitudes, and variability characteristics.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import LightcurveTensor

config = LightcurveTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## Pydantic Model Methods

### LightcurveTensor Methods

**`dim(self) -> int`**

Number of dimensions.

**`compute_period_folded(self, period: float, epoch: float = 0.0) -> 'LightcurveTensor'`**

Compute period-folded lightcurve.

Args:
period: Folding period
epoch: Reference epoch

Returns:
Period-folded LightcurveTensor

**`compute_statistics(self) -> Dict[str, torch.Tensor]`**

Compute basic lightcurve statistics.

**`filter_by_band(self, band: str) -> 'LightcurveTensor'`**

Filter lightcurve to specific band.

**`time_bin(self, bin_size: float) -> 'LightcurveTensor'`**

Bin lightcurve data in time.

Args:
bin_size: Size of time bins

Returns:
Binned LightcurveTensor

**`get_period(self, object_idx: int = 0) -> Optional[float]`**

Get period for specific object.

**`get_amplitude(self, object_idx: int = 0) -> Optional[float]`**

Get amplitude for specific object.

**`compute_variability_stats(self) -> Dict[str, torch.Tensor]`**

Compute variability statistics for each lightcurve.

Returns:
Dictionary with variability metrics

**`fold_lightcurve(self, period: Optional[float] = None, object_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]`**

Fold lightcurve with given period.

Args:
period: Period for folding (uses stored period if None)
object_idx: Object index if multiple objects

Returns:
Tuple of (folded_times, magnitudes)

**`detect_periods(self, min_period: float = 0.1, max_period: float = 100.0) -> torch.Tensor`**

Detect periods using Lomb-Scargle periodogram.

Args:
min_period: Minimum period to search
max_period: Maximum period to search

Returns:
Detected periods for each object

**`classify_variability(self) -> List[str]`**

Classify variability type based on lightcurve properties.

Returns:
List of variability classifications

**`phase_lightcurve(self, period: Optional[float] = None, epoch: float = 0.0) -> torch.Tensor`**

Calculate phase for each observation.

Args:
period: Period for phasing
epoch: Reference epoch

Returns:
Phase values (0-1)

**`bin_lightcurve(self, n_bins: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**

Bin lightcurve in phase.

Args:
n_bins: Number of phase bins

Returns:
Tuple of (bin_centers, binned_mags, bin_errors)

**`to_dict(self) -> Dict[str, Any]`**

Convert to dictionary representation.

**`model_post_init(self: 'BaseModel', context: 'Any', /) -> 'None'`**

This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
self: The BaseModel instance.
context: The context.

## Classes

### ALCDEFTemporalGNN

Temporal Graph Neural Network for ALCDEF lightcurve data with native tensor support.

#### Methods

**`forward(self, lightcurve: astro_lab.tensors.lightcurve.LightcurveTensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, return_embeddings: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]`**

Forward pass with native lightcurve support.

### ClassificationHead

Output head for asteroid classification.

#### Methods

**`forward(self, x: torch.Tensor) -> torch.Tensor`**

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### PeriodDetectionHead

Output head for rotation period detection.

#### Methods

**`forward(self, x: torch.Tensor) -> torch.Tensor`**

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### ShapeModelingHead

Output head for shape modeling parameters.

#### Methods

**`forward(self, x: torch.Tensor) -> torch.Tensor`**

Define the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

### TemporalGATCNN

Temporal Graph Attention Network with attention-based processing.

#### Methods

**`encode_snapshot(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**

Encode snapshot with attention mechanism.

Args:
x: Node features
edge_index: Edge indices
batch: Batch assignment

Returns:
Attention-enhanced graph embedding

### TemporalGCN

Base Temporal Graph Convolutional Network for astronomical time-series data.

#### Methods

**`encode_snapshot(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**

Encode a single graph snapshot.

Args:
x: Node features
edge_index: Edge indices
batch: Batch assignment

Returns:
Graph embedding

**`forward(self, snapshot_sequence: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]`**

Forward pass through temporal sequence.

Args:
snapshot_sequence: List of graph snapshots

Returns:
Temporal predictions
