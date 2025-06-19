"""
Tensor Type Protocols for astro_lab.tensors
==========================================

Protocol definitions to avoid circular imports between tensor classes.
Following the refactoring guide to eliminate string literals in type hints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch


@runtime_checkable
class TensorProtocol(Protocol):
    """Base protocol for all astronomical tensors."""

    @property
    def data(self) -> torch.Tensor: ...

    @property
    def shape(self) -> torch.Size: ...

    @property
    def device(self) -> torch.device: ...

    def to(self, *args, **kwargs) -> TensorProtocol: ...

    def clone(self) -> TensorProtocol: ...

    def get_metadata(self, key: str, default: Any = None) -> Any: ...


@runtime_checkable
class PhotometricTensorProtocol(Protocol):
    """Protocol for PhotometricTensor to avoid circular imports."""

    @property
    def bands(self) -> List[str]: ...

    @property
    def n_bands(self) -> int: ...

    @property
    def is_magnitude(self) -> bool: ...

    def get_band_data(self, band: str) -> torch.Tensor: ...

    def get_band_error(self, band: str) -> Optional[torch.Tensor]: ...

    def compute_colors(self, band1: str, band2: str) -> torch.Tensor: ...

    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class Spatial3DTensorProtocol(Protocol):
    """Protocol for Spatial3DTensor to avoid circular imports."""

    @property
    def cartesian(self) -> torch.Tensor: ...

    @property
    def coordinate_system(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def to_spherical(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def angular_separation(self, other: Spatial3DTensorProtocol) -> torch.Tensor: ...

    def cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor: ...


@runtime_checkable
class SpectralTensorProtocol(Protocol):
    """Protocol for SpectralTensor to avoid circular imports."""

    @property
    def wavelengths(self) -> torch.Tensor: ...

    @property
    def redshift(self) -> float: ...

    @property
    def flux_units(self) -> str: ...

    def redshift_correct(self, new_redshift: float) -> SpectralTensorProtocol: ...


@runtime_checkable
class LightcurveTensorProtocol(Protocol):
    """Protocol for LightcurveTensor to avoid circular imports."""

    @property
    def times(self) -> torch.Tensor: ...

    @property
    def magnitudes(self) -> torch.Tensor: ...

    @property
    def bands(self) -> List[str]: ...

    def phase_fold(self, period: float) -> LightcurveTensorProtocol: ...


@runtime_checkable
class OrbitTensorProtocol(Protocol):
    """Protocol for OrbitTensor to avoid circular imports."""

    @property
    def element_type(self) -> str: ...

    @property
    def epoch(self) -> float: ...

    def to_cartesian(self) -> OrbitTensorProtocol: ...

    def to_keplerian(self) -> OrbitTensorProtocol: ...

    def propagate(self, time_span: torch.Tensor) -> OrbitTensorProtocol: ...


@runtime_checkable
class SurveyTensorProtocol(Protocol):
    """Protocol for SurveyTensor (main coordinator) to avoid circular imports."""

    @property
    def survey_name(self) -> str: ...

    @property
    def data_release(self) -> Optional[str]: ...

    @property
    def filter_system(self) -> str: ...

    def get_photometric_tensor(self, **kwargs) -> PhotometricTensorProtocol: ...

    def get_spatial_tensor(self, **kwargs) -> Spatial3DTensorProtocol: ...

    def get_column(self, column_name: str) -> torch.Tensor: ...


# Type aliases for convenience
AstroTensor = Union[
    PhotometricTensorProtocol,
    Spatial3DTensorProtocol,
    SpectralTensorProtocol,
    LightcurveTensorProtocol,
    OrbitTensorProtocol,
    SurveyTensorProtocol,
]

TensorLike = Union[torch.Tensor, TensorProtocol]
