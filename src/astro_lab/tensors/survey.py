"""
SurveyTensor - Original Design
==============================

This tensor acts as the main coordinator for all specialized astronomical tensors,
providing unified access to photometry, astrometry, etc. based on a raw
data tensor and a column mapping.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import torch

from .base import AstroTensorBase
from .spatial_3d import Spatial3DTensor
from .photometric import PhotometricTensor
from ..utils.config.surveys import get_survey_features, get_survey_config

logger = logging.getLogger(__name__)

class SurveyTensor(AstroTensorBase):
    """
    Coordinator tensor for astronomical survey data.
    This class holds a raw data tensor and uses a column mapping
    to generate specialized tensors on demand.
    """

    def __init__(
        self,
        data: torch.Tensor,
        survey_name: str,
        column_mapping: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """
        Initializes the SurveyTensor.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("SurveyTensor data must be a torch.Tensor.")
        if not survey_name:
            raise ValueError("SurveyTensor requires a survey_name.")

        super().__init__(data, survey_name=survey_name, **kwargs)

        if column_mapping is None:
            # If no mapping is provided, create one from the survey config
            all_features = get_survey_features(survey_name)
            column_mapping = {name: i for i, name in enumerate(all_features)}
        
        self.column_mapping = column_mapping
        self._metadata["column_mapping"] = column_mapping

    def get_column(self, name: str) -> torch.Tensor:
        """Extracts a data column by its name using the column mapping."""
        if name not in self.column_mapping:
            raise KeyError(f"Column '{name}' not found. Available: {list(self.column_mapping.keys())}")
        idx = self.column_mapping[name]
        return self.data[:, idx]

    def get_spatial_tensor(self) -> Spatial3DTensor:
        """Creates a Spatial3DTensor from the survey data."""
        ra = self.get_column('ra')
        dec = self.get_column('dec')
        
        # For visualization, we need a 3rd dimension.
        # We derive a pseudo-distance for the z-coordinate.
        if 'parallax' in self.column_mapping:
            parallax = self.get_column('parallax')
            distance = 1.0 / torch.clamp(parallax, min=1e-5)
        elif 'phot_g_mean_mag' in self.column_mapping:
            z_source = self.get_column('phot_g_mean_mag')
            distance = (z_source - z_source.mean()) / z_source.std()
        else:
            distance = torch.ones_like(ra)

        # Here, we pass the spherical components to a class method that handles conversion
        return Spatial3DTensor.from_spherical(ra=ra, dec=dec, distance=distance, angular_unit="deg")

    def get_photometric_tensor(self) -> PhotometricTensor:
        """Creates a PhotometricTensor for the survey's default bands."""
        survey_config = get_survey_config(self.survey_name)
        bands = survey_config.get("photometric_bands", [])
        
        if not bands:
            raise ValueError(f"No photometric bands defined for survey '{self.survey_name}' in config.")

        mag_data = [self.get_column(col) for col in survey_config['mag_cols']]
        data = torch.stack(mag_data, dim=1)
        
        return PhotometricTensor(
            data=data,
            bands=bands,
            filter_system=survey_config.get("filter_system", "unknown")
        )

    def __repr__(self) -> str:
        return (f"SurveyTensor(survey='{self.survey_name}', "
                f"objects={self.shape[0]}, features={self.shape[1]})")

    @property
    def survey_name(self) -> str:
        return self._metadata.get("survey_name", "unknown")
