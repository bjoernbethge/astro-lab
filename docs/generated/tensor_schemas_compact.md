# Tensor_Schemas Module

Auto-generated documentation for `schemas.tensor_schemas`

## LightcurveTensorConfigSchema

Configuration schema for lightcurve tensors.

### Parameters

**`dtype`** *(string)* = `float32`
  Data type for tensor operations

**`device`** *(string)* = `auto`
  Device for tensor storage (auto, cpu, cuda, mps)

**`time_format`** *(string)* = `mjd`
  Time format (mjd, jd, isot)

**`time_scale`** *(string)* = `utc`
  Time scale (utc, tai, tt)

**`bands`** *(array)*
  Photometric bands for lightcurve

### Usage

```python
from docs.auto.schemas.data_schemas import LightcurveTensorConfigSchema

config = LightcurveTensorConfigSchema(
    bands=[]

    # Optional parameters:
    # dtype="example"
    # device="example"
    # time_format="example"
    # time_scale="example"
)
```

## OrbitTensorConfigSchema

Configuration schema for orbital tensors.

### Parameters

**`dtype`** *(string)* = `float32`
  Data type for tensor operations

**`device`** *(string)* = `auto`
  Device for tensor storage (auto, cpu, cuda, mps)

**`frame`** *(string)* = `ecliptic`
  Reference frame (ecliptic, equatorial)

**`units`** *(object)*
  Units for orbital elements

### Usage

```python
from docs.auto.schemas.data_schemas import OrbitTensorConfigSchema

config = OrbitTensorConfigSchema(
    units={}

    # Optional parameters:
    # dtype="example"
    # device="example"
    # frame="example"
)
```

## PhotometricTensorConfigSchema

Configuration schema for photometric tensors.

### Parameters

**`dtype`** *(string)* = `float32`
  Data type for tensor operations

**`device`** *(string)* = `auto`
  Device for tensor storage (auto, cpu, cuda, mps)

**`bands`** *(array)*
  Photometric bands

**`magnitude_system`** *(string)* = `AB`
  Magnitude system (AB, Vega, ST)

**`zeropoints`** *(Optional[object])* = `None`
  Zeropoints for each band

### Usage

```python
from docs.auto.schemas.data_schemas import PhotometricTensorConfigSchema

config = PhotometricTensorConfigSchema(
    bands=[]

    # Optional parameters:
    # dtype="example"
    # device="example"
    # magnitude_system="example"
    # zeropoints=None
)
```

## SpatialTensorConfigSchema

Configuration schema for spatial tensors.

### Parameters

**`dtype`** *(string)* = `float32`
  Data type for tensor operations

**`device`** *(string)* = `auto`
  Device for tensor storage (auto, cpu, cuda, mps)

**`coordinate_system`** *(string)* = `icrs`
  Coordinate system (icrs, galactic, ecliptic)

**`units`** *(object)*
  Units for spatial coordinates

**`epoch`** *(Optional[string])* = `J2000.0`
  Coordinate epoch

### Usage

```python
from docs.auto.schemas.data_schemas import SpatialTensorConfigSchema

config = SpatialTensorConfigSchema(
    units={}

    # Optional parameters:
    # dtype="example"
    # device="example"
    # coordinate_system="example"
    # epoch=None
)
```

## SpectralTensorConfigSchema

Configuration schema for spectral tensors.

### Parameters

**`dtype`** *(string)* = `float32`
  Data type for tensor operations

**`device`** *(string)* = `auto`
  Device for tensor storage (auto, cpu, cuda, mps)

**`wavelength_unit`** *(string)* = `angstrom`
  Wavelength units

**`flux_unit`** *(string)* = `erg/s/cm2/A`
  Flux units

**`spectral_resolution`** *(Optional[number])* = `None`
  Spectral resolution R = λ/Δλ

### Usage

```python
from docs.auto.schemas.data_schemas import SpectralTensorConfigSchema

config = SpectralTensorConfigSchema(

    # Optional parameters:
    # dtype="example"
    # device="example"
    # wavelength_unit="example"
    # flux_unit="example"
    # spectral_resolution=None
)
```

## SurveyTensorConfigSchema

Configuration schema for survey tensors.

### Parameters

**`dtype`** *(string)* = `float32`
  Data type for tensor operations

**`device`** *(string)* = `auto`
  Device for tensor storage (auto, cpu, cuda, mps)

**`survey_name`** *(string)*
  Name of the survey

**`data_release`** *(Optional[string])* = `None`
  Data release version

**`selection_function`** *(Optional[string])* = `None`
  Selection function description

### Usage

```python
from docs.auto.schemas.data_schemas import SurveyTensorConfigSchema

config = SurveyTensorConfigSchema(
    survey_name="example"

    # Optional parameters:
    # dtype="example"
    # device="example"
    # data_release=None
    # selection_function=None
)
```

## TensorConfigSchema

Base configuration schema for astronomical tensors.

### Parameters

**`dtype`** *(string)* = `float32`
  Data type for tensor operations

**`device`** *(string)* = `auto`
  Device for tensor storage (auto, cpu, cuda, mps)

### Usage

```python
from docs.auto.schemas.data_schemas import TensorConfigSchema

config = TensorConfigSchema(

    # Optional parameters:
    # dtype="example"
    # device="example"
)
```
