# Data_Schemas Module

Auto-generated documentation for `schemas.data_schemas`

## DataLoaderConfigSchema

Configuration schema for PyTorch Geometric DataLoaders.

### Parameters

**`batch_size`** *(integer)* = `32`
  Batch size for data loading
  *≥1, ≤10000*

**`shuffle`** *(boolean)* = `True`
  Whether to shuffle the data

**`num_workers`** *(integer)* = `0`
  Number of worker processes for data loading
  *≥0, ≤16*

**`pin_memory`** *(boolean)* = `True`
  Whether to pin memory for GPU transfer

**`use_gpu_optimization`** *(boolean)* = `True`
  Whether to use GPU optimization if available

### Usage

```python
from docs.auto.schemas.data_schemas import DataLoaderConfigSchema

config = DataLoaderConfigSchema(

    # Optional parameters:
    # batch_size=1
    # shuffle=True
    # num_workers=0
    # pin_memory=True
    # use_gpu_optimization=True
)
```

## DatasetConfigSchema

Base configuration schema for all datasets.

### Parameters

**`root`** *(Optional[string])* = `None`
  Root directory for dataset files

**`transform`** *(Optional[string])* = `None`
  Transform to apply to each sample

**`pre_transform`** *(Optional[string])* = `None`
  Transform to apply before saving

**`pre_filter`** *(Optional[string])* = `None`
  Filter to apply before saving

### Usage

```python
from docs.auto.schemas.data_schemas import DatasetConfigSchema

config = DatasetConfigSchema(

    # Optional parameters:
    # root=None
    # transform=None
    # pre_transform=None
    # pre_filter=None
)
```

## ExoplanetDatasetConfigSchema

Configuration schema for ExoplanetGraphDataset.

### Parameters

**`root`** *(Optional[string])* = `None`
  Root directory for dataset files

**`transform`** *(Optional[string])* = `None`
  Transform to apply to each sample

**`pre_transform`** *(Optional[string])* = `None`
  Transform to apply before saving

**`pre_filter`** *(Optional[string])* = `None`
  Filter to apply before saving

**`k_neighbors`** *(integer)* = `5`
  Number of nearest neighbors for graph construction
  *≥1, ≤50*

**`max_distance`** *(number)* = `100.0`
  Maximum distance for connections (parsecs)
  *>0.0*

### Usage

```python
from docs.auto.schemas.data_schemas import ExoplanetDatasetConfigSchema

config = ExoplanetDatasetConfigSchema(

    # Optional parameters:
    # root=None
    # transform=None
    # pre_transform=None
    # pre_filter=None
    # k_neighbors=1
    # max_distance=1.0
)
```

## GaiaDatasetConfigSchema

Configuration schema for GaiaGraphDataset.

### Parameters

**`root`** *(Optional[string])* = `None`
  Root directory for dataset files

**`transform`** *(Optional[string])* = `None`
  Transform to apply to each sample

**`pre_transform`** *(Optional[string])* = `None`
  Transform to apply before saving

**`pre_filter`** *(Optional[string])* = `None`
  Filter to apply before saving

**`magnitude_limit`** *(number)* = `12.0`
  Magnitude limit for star selection
  *≥5.0, ≤20.0*

**`k_neighbors`** *(integer)* = `8`
  Number of nearest neighbors for graph construction
  *≥1, ≤50*

**`max_distance`** *(number)* = `1.0`
  Maximum distance for connections (kpc)
  *>0.0*

### Usage

```python
from docs.auto.schemas.data_schemas import GaiaDatasetConfigSchema

config = GaiaDatasetConfigSchema(

    # Optional parameters:
    # root=None
    # transform=None
    # pre_transform=None
    # pre_filter=None
    # magnitude_limit=5.0
    # k_neighbors=1
    # max_distance=1.0
)
```

## NSADatasetConfigSchema

Configuration schema for NSAGraphDataset.

### Parameters

**`root`** *(Optional[string])* = `None`
  Root directory for dataset files

**`transform`** *(Optional[string])* = `None`
  Transform to apply to each sample

**`pre_transform`** *(Optional[string])* = `None`
  Transform to apply before saving

**`pre_filter`** *(Optional[string])* = `None`
  Filter to apply before saving

**`max_galaxies`** *(integer)* = `10000`
  Maximum number of galaxies to include
  *≥10, ≤1000000*

**`k_neighbors`** *(integer)* = `8`
  Number of nearest neighbors for graph construction
  *≥1, ≤50*

**`distance_threshold`** *(number)* = `50.0`
  Distance threshold for connections (Mpc)
  *>0.0*

### Usage

```python
from docs.auto.schemas.data_schemas import NSADatasetConfigSchema

config = NSADatasetConfigSchema(

    # Optional parameters:
    # root=None
    # transform=None
    # pre_transform=None
    # pre_filter=None
    # max_galaxies=10
    # k_neighbors=1
    # distance_threshold=1.0
)
```

## ProcessingConfigSchema

Configuration schema for data processing.

### Parameters

**`device`** *(string)* = `auto`
  Device for tensor operations (auto, cpu, cuda, mps)

**`batch_size`** *(integer)* = `32`
  Batch size for processing
  *≥1, ≤10000*

**`max_samples`** *(Optional[object])* = `None`
  Maximum samples per survey

**`surveys`** *(Optional[array])* = `None`
  List of surveys to process

### Usage

```python
from docs.auto.schemas.data_schemas import ProcessingConfigSchema

config = ProcessingConfigSchema(

    # Optional parameters:
    # device="example"
    # batch_size=1
    # max_samples=None
    # surveys=None
)
```
