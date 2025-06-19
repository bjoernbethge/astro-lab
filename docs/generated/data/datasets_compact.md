# Datasets Module

Auto-generated documentation for `data.datasets`

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

## PhotometricTensor

Tensor for multi-band photometric measurements using composition.

Stores magnitudes/fluxes across multiple bands with associated metadata
like measurement errors, extinction coefficients, and zero points.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import PhotometricTensor

config = PhotometricTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## Spatial3DTensor

Proper 3D Spatial Tensor for astronomical coordinates.

Uses unified [N, 3] tensor structure for efficient spatial operations.
Compatible with astroML, poliastro, and astropy.

The tensor data is always stored as 3D Cartesian coordinates [x, y, z]
in the specified coordinate system with proper units.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import Spatial3DTensor

config = Spatial3DTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## SpectralTensor

Tensor for astronomical spectral data with wavelength-dependent operations.

Handles spectroscopic data with wavelength calibration, redshift corrections,
and basic spectral analysis operations.

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import SpectralTensor

config = SpectralTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## SurveyTensor

Main coordinator tensor for astronomical survey data using composition.

Coordinates all specialized tensors and provides unified access to:
- Photometric measurements (via PhotometricTensor)
- Time series data (via LightcurveTensor)
- 3D spatial coordinates (via Spatial3DTensor)
- Spectroscopic data (via SpectralTensor)
- Astrometric measurements (via AstrometricTensor)
- Survey transformations and metadata

### Parameters

**`tensor_type`** *(string)* = `base`
  Type of astronomical tensor

### Usage

```python
from docs.auto.schemas.data_schemas import SurveyTensor

config = SurveyTensor(

    # Optional parameters:
    # tensor_type="example"
)
```

## Functions

### get_device()

Get the best available device (GPU if available, else CPU).

### gpu_knn_graph(coords, k_neighbors=8, max_distance=None, device=None)

Create k-NN graph using GPU acceleration when available.

### to_device(tensor, device=None)

Move tensor to device with memory optimization.

## Classes

### AstroLabDataset

General astronomical dataset using PyTorch Geometric's InMemoryDataset.

Provides easy access to astronomical catalogs for machine learning.
Creates tensor-based datasets from magnitude columns.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### AstroPhotDataset

PyTorch Geometric InMemoryDataset for AstroPhot galaxy fitting.

Creates graph datasets from galaxy catalogs with image cutouts
and morphological features.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### ExoplanetGraphDataset

PyTorch Geometric InMemoryDataset for Exoplanet data.

Creates spatial graphs from exoplanet systems with connections
based on stellar distances and system properties.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### GaiaGraphDataset

PyTorch Geometric InMemoryDataset for Gaia DR3 stellar data.

Creates spatial graphs from stellar catalogs with k-nearest neighbor
connections based on sky coordinates.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### LINEARLightcurveDataset

PyTorch Geometric InMemoryDataset for LINEAR lightcurve data.

Creates temporal graph datasets from LINEAR asteroid lightcurves
with connections based on orbital similarity and temporal patterns.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### NSAGraphDataset

PyTorch Geometric InMemoryDataset for NSA (NASA Sloan Atlas) galaxy data.

Creates spatial graphs from galaxy catalogs with k-nearest neighbor
connections based on 3D coordinates.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### RRLyraeDataset

PyTorch Geometric InMemoryDataset for RR Lyrae variable star data.

Creates temporal graph datasets from RR Lyrae observations
with connections based on period similarity and sky position.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### SDSSSpectralDataset

PyTorch Geometric InMemoryDataset for SDSS spectral data.

Creates graph datasets from SDSS spectroscopic observations
with connections based on spectral similarity and sky position.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### SatelliteOrbitDataset

PyTorch Geometric InMemoryDataset for satellite orbit data.

Creates orbital graph datasets from satellite trajectory data
with connections based on orbital similarity and proximity.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

### TNG50GraphDataset

PyTorch Geometric InMemoryDataset for TNG50 simulation data.

Creates spatial graphs from particle data with radius-based connections.

#### Methods

- **save(data_list: Sequence[torch_geometric.data.data.BaseData], path: str) -> None**
  - Saves a list of data objects to the file path :obj:`path`.

## Constants

- **ASTRO_LAB_TENSORS_AVAILABLE** (bool): `True`
