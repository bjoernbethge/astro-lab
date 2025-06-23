from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from astropy.table import Table
from pydantic import Field

from .photometric import PhotometricTensor


class SurveyTensor(PhotometricTensor):
    """Tensor for representing survey data, including photometric information."""

    survey_name: str = Field(..., description="Name of the astronomical survey")
    column_mapping: Dict[str, int] = Field(default_factory=dict, description="Mapping of column names to data indices")

    def __init__(self, data: Any = None, *args, **kwargs):
        # Ermöglicht sowohl SurveyTensor(data, ...) als auch SurveyTensor(data=..., ...)
        if data is not None:
            kwargs['data'] = data
        
        # Store survey_name in metadata if provided
        if 'survey_name' in kwargs:
            if 'meta' not in kwargs:
                kwargs['meta'] = {}
            kwargs['meta']['survey_name'] = kwargs['survey_name']
            
        super().__init__(*args, **kwargs)
        self._validate()

    def _validate(self) -> None:
        """Validate the SurveyTensor data and attributes."""
        # Call parent validation first
        super()._validate()
        
        # Validate survey_name
        if not self.survey_name or self.survey_name.strip() == "":
            raise ValueError("survey_name cannot be empty")
        
        # Validate column_mapping types
        if self.column_mapping:
            if not isinstance(self.column_mapping, dict):
                raise ValueError("column_mapping must be a dictionary")
            
            # Check that all values are integers (column indices)
            for col_name, col_idx in self.column_mapping.items():
                if not isinstance(col_idx, int):
                    raise ValueError(f"Column index for '{col_name}' must be an integer, got {type(col_idx)}")
                if col_idx < 0 or col_idx >= self.data.shape[1]:
                    raise ValueError(f"Column index {col_idx} for '{col_name}' is out of bounds")

    @property
    def n_objects(self) -> int:
        """Number of objects in the survey."""
        return self.data.shape[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary for serialization."""
        return {
            "data": self.data.cpu().numpy().tolist(),
            "survey_name": self.survey_name,
            "column_mapping": self.column_mapping,
            "meta": self.meta,
            "tensor_type": "survey"
        }

    @property
    def bands(self) -> List[str]:
        """Returns the list of photometric bands available in the tensor."""
        if self.column_mapping:
            return list(self.column_mapping.keys())
        return []

    def get_band_data(self, band: str) -> np.ndarray:
        """
        Retrieves the data for a specific photometric band.

        Args:
            band (str): The photometric band to retrieve.

        Returns:
            np.ndarray: The data for the specified band.
        """
        if self.column_mapping:
            column_name = self.column_mapping.get(band)
            if column_name and column_name in self.df.columns:
                return self.df[column_name].values
        raise ValueError(f"Band '{band}' not found in SurveyTensor.")

    @property
    def available_bands(self) -> List[str]:
        """Returns a list of available photometric bands."""
        if self.column_mapping:
            return [
                band
                for band, col in self.column_mapping.items()
                if col in self.df.columns
            ]
        return []

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the tensor to a pandas DataFrame, including only the bands
        specified in the column_mapping.
        """
        if not self.column_mapping:
            return self.df

        band_columns = list(self.column_mapping.values())
        return self.df[band_columns]
    
    def get_column(self, column_name: str) -> torch.Tensor:
        """
        Get data for a specific column by name.
        
        Args:
            column_name: Name of the column to retrieve
            
        Returns:
            torch.Tensor: Column data as tensor
        """
        if column_name not in self.column_mapping:
            raise ValueError(f"Column '{column_name}' not found in column_mapping")
        
        column_idx = self.column_mapping[column_name]
        return self.data[:, column_idx]
    
    def get_photometric_tensor(self) -> Optional[PhotometricTensor]:
        """
        Extract photometric data as a PhotometricTensor.
        
        Returns:
            PhotometricTensor with band data, or None if no photometric data
        """
        # Find photometric columns (containing 'mag' or starting with modelMag_)
        photometric_cols = []
        photometric_indices = []
        
        for col_name, col_idx in self.column_mapping.items():
            if 'mag' in col_name.lower() or col_name.startswith('modelMag_'):
                photometric_cols.append(col_name)
                photometric_indices.append(col_idx)
        
        if not photometric_cols:
            return None
        
        # Extract band names from column names
        bands = []
        for col in photometric_cols:
            if col.startswith('modelMag_'):
                bands.append(col.replace('modelMag_', ''))
            elif col.startswith('mag_'):
                bands.append(col.replace('mag_', ''))
            else:
                # Try to extract band from column name
                parts = col.split('_')
                if len(parts) > 1:
                    bands.append(parts[-1])
                else:
                    bands.append(col)
        
        # Extract photometric data
        photometric_data = self.data[:, photometric_indices]
        
        return PhotometricTensor(
            data=photometric_data,
            bands=bands,
            filter_system=self.get_metadata("filter_system", "AB")
        )
    
    def get_spatial_tensor(self) -> Optional['Spatial3DTensor']:
        """
        Extract spatial coordinate data as a Spatial3DTensor.
        
        Returns:
            Spatial3DTensor with coordinate data, or None if no spatial data
        """
        from .spatial_3d import Spatial3DTensor
        
        # Look for standard coordinate columns
        coord_cols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
        found_cols = {}
        
        for col in coord_cols:
            if col in self.column_mapping:
                found_cols[col] = self.column_mapping[col]
        
        if 'ra' not in found_cols or 'dec' not in found_cols:
            return None
        
        # Extract coordinate data
        ra = self.data[:, found_cols['ra']]
        dec = self.data[:, found_cols['dec']]
        
        # Optional: distance from parallax
        distance = None
        if 'parallax' in found_cols:
            parallax = self.data[:, found_cols['parallax']]
            # Convert parallax to distance in kpc (avoiding division by zero)
            mask = parallax > 0.1  # 0.1 mas minimum
            distance = torch.zeros_like(parallax)
            distance[mask] = 1000.0 / parallax[mask]  # mas to kpc
        else:
            # Default distance if no parallax available
            distance = torch.ones_like(ra)
        
        # Use from_spherical class method to create Spatial3DTensor
        return Spatial3DTensor.from_spherical(
            ra=ra,
            dec=dec,
            distance=distance,
            angular_unit='deg',
            frame=self.get_metadata("coordinate_system", "icrs"),
            unit=self.get_metadata("distance_unit", "kiloparsec")
        ) 