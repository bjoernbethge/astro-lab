# Utils Module

Auto-generated documentation for `data.utils`

## Functions

### check_astroquery_available() -> bool

Check if astroquery is available for data downloads.

### create_training_splits(df: polars.dataframe.frame.DataFrame, test_size: float = 0.2, val_size: float = 0.1, stratify_column: Optional[str] = None, random_state: Optional[int] = 42, shuffle: bool = True) -> Tuple[polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame]

Create train/validation/test splits using native Polars operations.

Parameters
----------
df : pl.DataFrame
    Input DataFrame to split
test_size : float, default 0.2
    Proportion of data for test set
val_size : float, default 0.1
    Proportion of data for validation set
stratify_column : str, optional
    Column to use for stratified splitting (basic implementation)
random_state : int, optional
    Random seed for reproducibility
shuffle : bool, default True
    Whether to shuffle before splitting
    
Returns
-------
Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
    Training, validation, and test DataFrames
    
Raises
------
ValueError
    If split sizes are invalid

### get_data_dir() -> pathlib.Path

Get the configured data directory.

### get_data_statistics(df: polars.dataframe.frame.DataFrame) -> Dict[str, Any]

Get comprehensive statistics for astronomical DataFrame.

Args:
    df: Input Polars DataFrame

Returns:
    Dictionary with statistics

### get_fits_info(fits_path: Union[str, pathlib.Path]) -> Dict[str, Any]

Get comprehensive information about a FITS file.

Args:
    fits_path: Path to FITS file

Returns:
    Dictionary with file information

### load_fits_optimized(fits_path: Union[str, pathlib.Path], hdu_index: int = 0, memmap: bool = True, do_not_scale: bool = False, section: Optional[Tuple[slice, ...]] = None, max_memory_mb: float = 1000.0) -> Union[numpy.ndarray, astropy.table.table.Table, NoneType]

Load FITS data with optimizations for large files.

Args:
    fits_path: Path to FITS file
    hdu_index: HDU index to load
    memmap: Use memory mapping
    do_not_scale: Disable automatic scaling
    section: Data section to load
    max_memory_mb: Memory limit for automatic memmap

Returns:
    Loaded data as ndarray or Table

### load_fits_table_optimized(fits_path: Union[str, pathlib.Path], hdu_index: int = 1, columns: Optional[List[str]] = None, max_rows: Optional[int] = None, as_polars: bool = True) -> Union[polars.dataframe.frame.DataFrame, astropy.table.table.Table, NoneType]

Load FITS table data with optimizations.

Args:
    fits_path: Path to FITS file
    hdu_index: HDU index (typically 1 for tables)
    columns: Specific columns to load
    max_rows: Maximum rows to load
    as_polars: Return as Polars DataFrame

Returns:
    Loaded table as DataFrame or Table

### load_splits_from_parquet(base_path: Union[str, pathlib.Path], dataset_name: str) -> Tuple[polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame, polars.dataframe.frame.DataFrame]

Load train/validation/test splits from Parquet files.

Parameters
----------
base_path : Union[str, Path]
    Base directory containing the split files
dataset_name : str
    Name of the dataset for file naming
    
Returns
-------
Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
    Training, validation, and test DataFrames

### preprocess_catalog(df: polars.dataframe.frame.DataFrame, clean_null_columns: bool = True, min_observations: Optional[int] = None, magnitude_columns: Optional[List[str]] = None, coordinate_columns: Optional[List[str]] = None) -> polars.dataframe.frame.DataFrame

Preprocess astronomical catalog data with common cleaning operations.

Parameters
----------
df : pl.DataFrame
    Input catalog DataFrame
clean_null_columns : bool, default True
    Whether to remove columns with too many nulls
min_observations : int, optional
    Minimum number of valid observations required
magnitude_columns : List[str], optional
    Magnitude columns for cleaning
coordinate_columns : List[str], optional
    Coordinate columns for validation
    
Returns
-------
pl.DataFrame
    Cleaned catalog DataFrame

### save_splits_to_parquet(df_train: polars.dataframe.frame.DataFrame, df_val: polars.dataframe.frame.DataFrame, df_test: polars.dataframe.frame.DataFrame, base_path: Union[str, pathlib.Path], dataset_name: str) -> Dict[str, pathlib.Path]

Save train/validation/test splits to Parquet files.

Parameters
----------
df_train, df_val, df_test : pl.DataFrame
    DataFrames to save
base_path : Union[str, Path]
    Base directory for saving
dataset_name : str
    Name of the dataset for file naming
    
Returns
-------
Dict[str, Path]
    Dictionary mapping split names to file paths

## Constants

- **ASTROPY_AVAILABLE** (bool): `True`
- **ASTROQUERY_AVAILABLE** (bool): `True`
