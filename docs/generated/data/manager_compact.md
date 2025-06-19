# Manager Module

Auto-generated documentation for `data.manager`

## Functions

### download_bright_all_sky(magnitude_limit: float = 12.0) -> pathlib.Path

Download bright all-sky Gaia catalog (~1 GB).

### download_gaia(region: str = 'lmc', magnitude_limit: float = 15.0) -> pathlib.Path

Download Gaia catalog.

### import_fits(fits_file: Union[str, pathlib.Path], catalog_name: str) -> pathlib.Path

Import FITS catalog.

### import_tng50(hdf5_file: Union[str, pathlib.Path], dataset_name: str = 'PartType0') -> pathlib.Path

Import TNG50 simulation data.

### list_catalogs() -> polars.dataframe.frame.DataFrame

List all available catalogs.

### load_bright_stars(limit: Optional[int] = None) -> polars.dataframe.frame.DataFrame

Load bright stars (alias for load_gaia_bright_stars)

### load_catalog(path: Union[str, pathlib.Path]) -> polars.dataframe.frame.DataFrame

Load catalog from path.

### load_gaia_bright_stars(magnitude_limit: float = 12.0) -> polars.dataframe.frame.DataFrame

Load bright Gaia DR3 stars from our real catalogs.

Parameters
----------
magnitude_limit : float, default 12.0
    Magnitude limit (10.0 or 12.0 available)

Returns
-------
pl.DataFrame
    Gaia star catalog

### process_for_ml(raw_file: Union[str, pathlib.Path], **kwargs) -> pathlib.Path

Process raw catalog for ML.

## Classes

### AstroDataManager

Modern astronomical data management with structured storage.
