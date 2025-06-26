"""
Base Classes for Graph Building
==============================

Provides base classes and configuration for centralized graph construction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch_geometric.data import Data

from astro_lab.tensors import SurveyTensorDict


@dataclass
class GraphConfig:
    """Configuration for graph building."""

    # Graph construction method
    method: str = "knn"  # "knn", "radius", "astronomical"

    # KNN parameters
    k_neighbors: int = 8

    # Radius parameters
    radius: float = 1.0

    # Astronomical parameters
    use_3d_coordinates: bool = True
    coordinate_system: str = "cartesian"  # "cartesian", "spherical"

    # Feature selection
    use_photometry: bool = True
    use_astrometry: bool = True
    use_spectroscopy: bool = False

    # Graph properties
    directed: bool = False
    self_loops: bool = False

    # Device
    device: Optional[Union[str, torch.device]] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.method not in ["knn", "radius", "astronomical"]:
            raise ValueError(f"Unknown method: {self.method}")

        if self.coordinate_system not in ["cartesian", "spherical"]:
            raise ValueError(f"Unknown coordinate system: {self.coordinate_system}")


class BaseGraphBuilder(ABC):
    """Base class for graph builders."""

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @abstractmethod
    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build graph from SurveyTensorDict."""
        pass

    def validate_input(self, survey_tensor: SurveyTensorDict) -> None:
        """Validate input SurveyTensorDict."""
        if not isinstance(survey_tensor, SurveyTensorDict):
            raise ValueError("Input must be SurveyTensorDict")

        # Check for required spatial data
        if "spatial" not in survey_tensor:
            raise ValueError("SurveyTensorDict must contain 'spatial' data")

        # Validate k_neighbors parameter
        if hasattr(self.config, "k_neighbors"):
            if self.config.k_neighbors <= 0:
                raise ValueError(
                    f"k_neighbors must be positive, got {self.config.k_neighbors}"
                )
            if self.config.k_neighbors > 1000:
                raise ValueError(f"k_neighbors too large: {self.config.k_neighbors}")

    def extract_coordinates(self, survey_tensor: SurveyTensorDict) -> torch.Tensor:
        """Extract coordinates from SurveyTensorDict."""
        spatial_data = survey_tensor["spatial"]

        if self.config.use_3d_coordinates:
            # Use 3D coordinates if available
            if (
                hasattr(spatial_data, "x")
                and hasattr(spatial_data, "y")
                and hasattr(spatial_data, "z")
            ):
                coords = torch.stack(
                    [spatial_data.x, spatial_data.y, spatial_data.z], dim=-1
                )
            else:
                # Fallback to 2D
                coords = torch.stack([spatial_data.x, spatial_data.y], dim=-1)
        else:
            # Use 2D coordinates
            coords = torch.stack([spatial_data.x, spatial_data.y], dim=-1)

        return coords.to(self.device)

    def extract_features(self, survey_tensor: SurveyTensorDict) -> torch.Tensor:
        """Extract features from SurveyTensorDict."""
        features = []

        # Spatial features
        if self.config.use_astrometry and "spatial" in survey_tensor:
            spatial_data = survey_tensor["spatial"]
            if hasattr(spatial_data, "x"):
                features.append(spatial_data.x)
            if hasattr(spatial_data, "y"):
                features.append(spatial_data.y)
            if self.config.use_3d_coordinates and hasattr(spatial_data, "z"):
                features.append(spatial_data.z)

        # Photometric features
        if self.config.use_photometry and "photometric" in survey_tensor:
            phot_data = survey_tensor["photometric"]
            if hasattr(phot_data, "magnitudes"):
                features.append(phot_data.magnitudes)

        # Spectral features
        if self.config.use_spectroscopy and "spectral" in survey_tensor:
            spec_data = survey_tensor["spectral"]
            if hasattr(spec_data, "fluxes"):
                features.append(spec_data.fluxes)

        if not features:
            # Fallback to coordinates only
            return self.extract_coordinates(survey_tensor)

        return torch.cat(features, dim=-1).to(self.device)

    def create_data_object(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        **kwargs,
    ) -> Data:
        """Create PyG Data object."""
        data = Data(x=features, edge_index=edge_index, pos=coords, **kwargs)

        # Add metadata
        data.num_nodes = features.size(0)
        data.num_edges = edge_index.size(1)

        return data
