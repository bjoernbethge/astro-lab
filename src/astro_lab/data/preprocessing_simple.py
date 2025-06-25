"""
Simple preprocessing for testing and small datasets.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


def preprocess_catalog(
    input_path: Union[str, Path],
    survey_type: str,
    max