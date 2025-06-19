# Model_Schemas Module

Auto-generated documentation for `schemas.model_schemas`

## AstroPhotModelConfigSchema

Configuration schema for AstroPhot models.

### Parameters

**`model_type`** *(string)* = `astrophot`
  AstroPhot model variant

**`input_dim`** *(integer)*
  Input feature dimension
  *≥1*

**`hidden_dim`** *(integer)* = `64`
  Hidden layer dimension
  *≥1, ≤4096*

**`output_dim`** *(integer)*
  Output dimension
  *≥1*

**`dropout`** *(number)* = `0.1`
  Dropout probability
  *≥0.0, ≤1.0*

**`cutout_size`** *(integer)* = `128`
  Size of image cutouts
  *≥32, ≤512*

**`pixel_scale`** *(number)* = `0.262`
  Pixel scale in arcsec/pixel
  *>0.0*

**`magnitude_range`** *(array)* = `[10.0, 18.0]`
  Magnitude range for training

### Usage

```python
from docs.auto.schemas.data_schemas import AstroPhotModelConfigSchema

config = AstroPhotModelConfigSchema(
    input_dim=1,
    output_dim=1

    # Optional parameters:
    # model_type="example"
    # hidden_dim=1
    # dropout=0.0
    # cutout_size=32
    # pixel_scale=1.0
    # magnitude_range=[]
)
```

## CNNConfigSchema

Configuration schema for Convolutional Neural Networks.

### Parameters

**`model_type`** *(string)* = `cnn`
  CNN architecture type

**`input_dim`** *(integer)*
  Input feature dimension
  *≥1*

**`hidden_dim`** *(integer)* = `64`
  Hidden layer dimension
  *≥1, ≤4096*

**`output_dim`** *(integer)*
  Output dimension
  *≥1*

**`dropout`** *(number)* = `0.1`
  Dropout probability
  *≥0.0, ≤1.0*

**`num_layers`** *(integer)* = `4`
  Number of convolutional layers
  *≥1, ≤20*

**`kernel_sizes`** *(array)*
  Kernel sizes for each layer

**`channels`** *(array)*
  Number of channels for each layer

**`pool_sizes`** *(array)*
  Pooling sizes for each layer

### Usage

```python
from docs.auto.schemas.data_schemas import CNNConfigSchema

config = CNNConfigSchema(
    input_dim=1,
    output_dim=1,
    kernel_sizes=[],
    channels=[],
    pool_sizes=[]

    # Optional parameters:
    # model_type="example"
    # hidden_dim=1
    # dropout=0.0
    # num_layers=1
)
```

## GNNConfigSchema

Configuration schema for Graph Neural Networks.

### Parameters

**`model_type`** *(string)* = `gcn`
  GNN type (gcn, gat, sage, gin, etc.)

**`input_dim`** *(integer)*
  Input feature dimension
  *≥1*

**`hidden_dim`** *(integer)* = `64`
  Hidden layer dimension
  *≥1, ≤4096*

**`output_dim`** *(integer)*
  Output dimension
  *≥1*

**`dropout`** *(number)* = `0.1`
  Dropout probability
  *≥0.0, ≤1.0*

**`num_layers`** *(integer)* = `3`
  Number of GNN layers
  *≥1, ≤20*

**`heads`** *(integer)* = `1`
  Number of attention heads (for GAT)
  *≥1, ≤16*

**`edge_dim`** *(Optional[integer])* = `None`
  Edge feature dimension

**`aggr`** *(string)* = `mean`
  Aggregation method (mean, max, add)

**`residual`** *(boolean)* = `True`
  Use residual connections

**`batch_norm`** *(boolean)* = `True`
  Use batch normalization

### Usage

```python
from docs.auto.schemas.data_schemas import GNNConfigSchema

config = GNNConfigSchema(
    input_dim=1,
    output_dim=1

    # Optional parameters:
    # model_type="example"
    # hidden_dim=1
    # dropout=0.0
    # num_layers=1
    # heads=1
    # edge_dim=None
    # aggr="example"
    # residual=True
    # batch_norm=True
)
```

## ModelConfigSchema

Base configuration schema for all models.

### Parameters

**`model_type`** *(string)*
  Type of model (gnn, transformer, cnn, etc.)

**`input_dim`** *(integer)*
  Input feature dimension
  *≥1*

**`hidden_dim`** *(integer)* = `64`
  Hidden layer dimension
  *≥1, ≤4096*

**`output_dim`** *(integer)*
  Output dimension
  *≥1*

**`dropout`** *(number)* = `0.1`
  Dropout probability
  *≥0.0, ≤1.0*

### Usage

```python
from docs.auto.schemas.data_schemas import ModelConfigSchema

config = ModelConfigSchema(
    model_type="example",
    input_dim=1,
    output_dim=1

    # Optional parameters:
    # hidden_dim=1
    # dropout=0.0
)
```

## TNG50ModelConfigSchema

Configuration schema for TNG50 simulation models.

### Parameters

**`model_type`** *(string)* = `tng50`
  TNG50 model type

**`input_dim`** *(integer)*
  Input feature dimension
  *≥1*

**`hidden_dim`** *(integer)* = `64`
  Hidden layer dimension
  *≥1, ≤4096*

**`output_dim`** *(integer)*
  Output dimension
  *≥1*

**`dropout`** *(number)* = `0.1`
  Dropout probability
  *≥0.0, ≤1.0*

**`particle_types`** *(array)*
  Particle types to include

**`max_particles`** *(integer)* = `10000`
  Maximum particles per sample
  *≥100, ≤1000000*

**`environment_types`** *(integer)* = `4`
  Number of environment types
  *≥2, ≤10*

### Usage

```python
from docs.auto.schemas.data_schemas import TNG50ModelConfigSchema

config = TNG50ModelConfigSchema(
    input_dim=1,
    output_dim=1,
    particle_types=[]

    # Optional parameters:
    # model_type="example"
    # hidden_dim=1
    # dropout=0.0
    # max_particles=100
    # environment_types=2
)
```

## TransformerConfigSchema

Configuration schema for Transformer models.

### Parameters

**`model_type`** *(string)* = `transformer`
  Transformer variant

**`input_dim`** *(integer)*
  Input feature dimension
  *≥1*

**`hidden_dim`** *(integer)* = `64`
  Hidden layer dimension
  *≥1, ≤4096*

**`output_dim`** *(integer)*
  Output dimension
  *≥1*

**`dropout`** *(number)* = `0.1`
  Dropout probability
  *≥0.0, ≤1.0*

**`num_layers`** *(integer)* = `6`
  Number of transformer layers
  *≥1, ≤24*

**`num_heads`** *(integer)* = `8`
  Number of attention heads
  *≥1, ≤32*

**`ff_dim`** *(integer)* = `256`
  Feed-forward layer dimension
  *≥1, ≤8192*

**`max_seq_length`** *(integer)* = `1024`
  Maximum sequence length
  *≥1*

### Usage

```python
from docs.auto.schemas.data_schemas import TransformerConfigSchema

config = TransformerConfigSchema(
    input_dim=1,
    output_dim=1

    # Optional parameters:
    # model_type="example"
    # hidden_dim=1
    # dropout=0.0
    # num_layers=1
    # num_heads=1
    # ff_dim=1
    # max_seq_length=1
)
```
