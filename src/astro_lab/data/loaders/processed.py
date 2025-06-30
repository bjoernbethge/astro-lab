"""
Processed Data Loader
=====================

Load processed astronomical survey data.

DEPRECATED: This module is deprecated. Use astro_lab.data.get_preprocessor() instead.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


class ProcessedDataLoader:
    """
    Loader for processed astronomical survey data.
    
    DEPRECATED: Use astro_lab.data.get_preprocessor() instead.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing processed data files
        """
        warnings.warn(
            "ProcessedDataLoader is deprecated. Use astro_lab.data.get_preprocessor() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.data_dir = data_dir or Path("data/processed")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def list_available_surveys(self) -> List[str]:
        """
        List available surveys with processed data.
        
        Returns:
            List of survey names
        """
        from astro_lab.data import get_supported_surveys
        return get_supported_surveys()
    
    def load_survey_processed_data(
        self, 
        survey: str, 
        columns: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Load processed survey data.
        
        Args:
            survey: Survey name
            columns: Columns to load (None for all)
            max_samples: Maximum number of samples
            
        Returns:
            Polars DataFrame with survey data
        """
        from astro_lab.data import get_preprocessor
        
        # Use the new API
        preprocessor = get_preprocessor(survey)
        df = preprocessor.load_raw_data(max_samples=max_samples)
        
        # Select columns if specified
        if columns is not None:
            available_columns = [col for col in columns if col in df.columns]
            if available_columns:
                df = df.select(available_columns)
            else:
                logger.warning(f"No requested columns found in {survey} data")
                
        return df
