from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.table import Table
from pydantic import Field

from .photometric import PhotometricTensor


class SurveyTensor(PhotometricTensor):
    """Tensor for representing survey data, including photometric information."""

    column_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping from standard band names to column names in the data.",
    )

    def __init__(self, data: Any, column_mapping: Optional[Dict[str, str]] = None, **kwargs):
        """
        Initializes the SurveyTensor, ensuring the parent PhotometricTensor
        is created correctly with the necessary 'bands' information.
        """
        if column_mapping is None:
            raise ValueError("SurveyTensor requires a 'column_mapping' keyword argument.")

        # We need to pass 'bands' to the parent, and also 'column_mapping' to the base model
        # so the pydantic field is populated.
        kwargs["column_mapping"] = column_mapping
        bands = list(column_mapping.keys())

        super().__init__(data=data, bands=bands, **kwargs)

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