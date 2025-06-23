from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.table import Table
from pydantic import Field

from .photometric import PhotometricTensor


class SurveyTensor(PhotometricTensor):
    """Tensor for representing survey data, including photometric information."""

    survey_name: str = Field(..., description="Name of the astronomical survey")
    column_mapping: Dict[str, str] = Field(default_factory=dict, description="Mapping of column names to data indices")

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._validate()

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