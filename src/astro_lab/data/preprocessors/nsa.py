"""Enhanced NSA (NASA-Sloan Atlas) preprocessor implementation with unified 3D coordinates.

Handles NSA galaxy catalog preprocessing with multidimensional column support,
proper column mapping, and cosmological distance calculation.
"""

import logging
from typing import Any, Dict, List, Optional

import astropy.units as u
import polars as pl
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Column, Table

from astro_lab.config import get_survey_config

from .astro import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
)

logger = logging.getLogger(__name__)


class NSAPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Enhanced preprocessor for NASA-Sloan Atlas data with unified 3D coordinate handling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize NSA preprocessor."""
        default_config = {
            "redshift_limit": None,  # No filter by default
            "mass_limit": None,  # No filter by default
            "require_sersic": False,  # No filter by default
            "distance_limit_mpc": None,  # No filter by default
            "include_morphology": True,
            "include_colors": True,
            "cosmology": FlatLambdaCDM(Hubble=70, Omega_m=0.3),
            "map_columns": True,  # Enable column mapping
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        # Get survey configuration
        self.survey_config = get_survey_config("nsa")

        # Extended column mapping for NSA (case-insensitive)
        self.column_mapping = {
            # Standard coordinate columns
            "RA": "ra",
            "DEC": "dec",
            "Z": "z",
            "REDSHIFT": "z",
            "NSAID": "nsaid",
            "IAUNAME": "iau_name",
            # Magnitudes - try different prefixes
            "MAG_G": "mag_g",
            "MAG_R": "mag_r",
            "MAG_I": "mag_i",
            "MAG_Z": "mag_z",
            "MAG_U": "mag_u",
            "ELPETRO_ABSMAG": "abs_mag",
            "PETRO_FLUX": "petro_flux",
            "MODELFLUX": "model_flux",
            # Physical properties
            "MASS": "mass",
            "MSTAR": "stellar_mass",
            "SERSIC_N": "sersic_n",
            "SERSIC_R50": "sersic_r50",
            "PETRO_R50": "petro_r50",
            "RE": "effective_radius",
            "PETROTH50": "petro_r50",
            "PETROTH90": "petro_r90",
            # Alternative names
            "OBJID": "objid",
            "ID": "objid",
            "PLATEIFU": "plateifu",
        }

        self.required_columns = ["ra", "dec"]  # Only essential coordinates
        self.cosmology = self.config["cosmology"]

    def get_survey_name(self) -> str:
        """Get the survey name for this preprocessor."""
        return "nsa"

    def get_object_type(self) -> str:
        """Get the primary object type for this survey."""
        return "galaxy"

    def _filter_single_dimensional_columns(self, table: Table) -> List[str]:
        """Filter out multidimensional columns from astropy table."""
        single_dim_cols = []
        multidim_cols = []

        for name in table.colnames:
            col = table[name]
            if isinstance(col, Column):
                if col.ndim <= 1:
                    single_dim_cols.append(name)
                else:
                    multidim_cols.append(name)
            else:
                single_dim_cols.append(name)

        logger.info(f"Filtered out {len(multidim_cols)} multidimensional columns")
        logger.debug(f"Multidimensional columns: {multidim_cols[:10]}...")
        logger.info(f"Keeping {len(single_dim_cols)} single-dimensional columns")

        return single_dim_cols

    def _load_fits_data(self, file_path: str) -> pl.DataFrame:
        """Load NSA FITS data handling multidimensional columns."""
        logger.info(f"Loading NSA FITS data from {file_path}")

        # Load astropy table
        table = Table.read(file_path, format="fits")

        # Filter out multidimensional columns
        single_dim_cols = self._filter_single_dimensional_columns(table)

        # Keep only single dimensional columns
        filtered_table = table[single_dim_cols]

        # Convert astropy table to pandas (handles type conversion automatically)
        logger.info("Converting astropy table to pandas...")

        pandas_df = filtered_table.to_pandas()

        # Convert pandas to polars
        logger.info("Converting pandas to polars...")
        df = pl.from_pandas(pandas_df)

        logger.info(
            f"Successfully loaded {len(df)} NSA galaxies with {len(df.columns)} columns"
        )
        return df

    def _map_and_select_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Map column names and select relevant columns for NSA."""
        if not self.config.get("map_columns", True):
            return df

        logger.info("Mapping and selecting columns for nsa")

        # Apply column mapping (case-insensitive)
        available_columns = df.columns
        mapping_applied = {}

        for old_name, new_name in self.column_mapping.items():
            # Check exact match first
            if old_name in available_columns:
                mapping_applied[old_name] = new_name
            else:
                # Check case-insensitive match
                for col in available_columns:
                    if col.upper() == old_name.upper():
                        mapping_applied[col] = new_name
                        break

        # Apply mapping
        if mapping_applied:
            df = df.rename(mapping_applied)
            logger.info(f"Applied column mapping: {mapping_applied}")

        # Select essential and optional columns for NSA
        essential_columns = ["ra", "dec"]
        optional_columns = [
            "nsaid",
            "iau_name",
            "objid",
            "plateifu",
            "z",
            "redshift",
            "sersic_n",
            "sersic_r50",
            "petro_r50",
            "petro_r90",
            "effective_radius",
            "mass",
            "stellar_mass",
            "abs_mag",
            "petro_flux",
            "model_flux",
            "mag_g",
            "mag_r",
            "mag_i",
            "mag_z",
            "mag_u",
        ]

        # Keep columns that exist
        columns_to_keep = []
        for col in essential_columns:
            if col in df.columns:
                columns_to_keep.append(col)
            else:
                # Try alternative names
                alt_names = {"ra": ["RA", "Ra"], "dec": ["DEC", "Dec", "DE"]}
                if col in alt_names:
                    for alt in alt_names[col]:
                        if alt in df.columns:
                            columns_to_keep.append(alt)
                            break
        for col in optional_columns:
            if col in df.columns:
                columns_to_keep.append(col)

        # Remove duplicates
        columns_to_keep = list(dict.fromkeys(columns_to_keep))

        df = df.select(columns_to_keep)
        logger.info(f"Selected {len(columns_to_keep)} columns for NSA processing")
        return df

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Standard preprocessing pipeline for NSA survey. Expects a DataFrame as input.
        """
        logger.info("Starting preprocessing for nsa survey")
        # 1. Map and select columns
        df = self._map_and_select_columns(df)
        logger.info(f"After column mapping: {len(df.columns)} columns")
        # 2. Apply filters
        df = self.filter(df)
        # 3. Apply transformations
        df = self.transform(df)
        # 4. Extract features
        df = self.extract_features(df)
        # 5. Ensure unified 3D coordinates
        df = self.ensure_3d_coordinates(df)
        # 6. Standardize output format
        df = self.add_standard_fields(df)
        logger.info(
            f"Preprocessing complete: {len(df)} objects with unified 3D coordinates"
        )
        return df

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply NSA-specific filters for galaxies."""
        logger.info("Applying NSA galaxy filters...")
        initial_count = len(df)
        # Apply column mapping first
        df = self._map_and_select_columns(df)
        # Redshift filter
        redshift_cols = ["z", "redshift"]
        redshift_col = None
        for col in redshift_cols:
            if col in df.columns:
                redshift_col = col
                break
        if redshift_col and self.config["redshift_limit"] is not None:
            df = df.filter(
                (pl.col(redshift_col) > 0)
                & (pl.col(redshift_col) < self.config["redshift_limit"])
            )
            logger.info(
                f"Applied redshift filter: 0 < {redshift_col} < {self.config['redshift_limit']}"
            )
        # Stellar mass filter (try different mass column names)
        mass_cols = ["mass", "stellar_mass", "mstar"]
        mass_col = None
        for col in mass_cols:
            if col in df.columns:
                mass_col = col
                break
        if mass_col and self.config["mass_limit"] is not None:
            min_mass, max_mass = self.config["mass_limit"]
            df = df.filter(
                (pl.col(mass_col) >= min_mass) & (pl.col(mass_col) <= max_mass)
            )
            logger.info(
                f"Applied stellar mass filter: {min_mass} < log(M*) < {max_mass}"
            )
        # Require Sersic parameters if requested
        if self.config["require_sersic"]:
            sersic_cols = ["sersic_n"]
            sersic_col = None
            for col in sersic_cols:
                if col in df.columns:
                    sersic_col = col
                    break
            if sersic_col:
                df = df.filter((pl.col(sersic_col) > 0) & (pl.col(sersic_col) < 10))
                logger.info("Applied Sersic index filter: 0 < n < 10")
        # Distance filter based on cosmology
        if redshift_col and self.config["distance_limit_mpc"] is not None:
            redshift_values = df[redshift_col].to_numpy()
            distances_mpc = self.cosmology.luminosity_distance(redshift_values).value
            df = df.filter(
                pl.Series("distance_mpc", distances_mpc)
                < self.config["distance_limit_mpc"]
            )
            logger.info(
                f"Applied distance filter: d < {self.config['distance_limit_mpc']} Mpc"
            )
        final_count = len(df)
        logger.info(
            f"Filtered {initial_count - final_count} galaxies ({((initial_count - final_count) / initial_count * 100):.1f}%)"
        )
        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform NSA data for unified 3D processing."""
        logger.info("Transforming NSA galaxy data...")

        # Convert redshift to distance using cosmology
        redshift_cols = ["z", "redshift"]
        redshift_col = None
        for col in redshift_cols:
            if col in df.columns:
                redshift_col = col
                break

        if redshift_col:
            redshift_values = df[redshift_col].to_numpy()

            # Calculate various cosmological distances
            distances_mpc = self.cosmology.luminosity_distance(redshift_values).value
            distances_pc = distances_mpc * 1e6  # Convert to parsecs

            # Add distance information
            df = df.with_columns(
                [
                    pl.Series("distance_mpc", distances_mpc),
                    pl.Series("distance_pc", distances_pc),
                    pl.Series(
                        "z", redshift_values
                    ),  # Ensure consistent redshift column
                ]
            )
            logger.info("Converted redshift to cosmological distances")

        # Add galactic coordinates
        if "ra" in df.columns and "dec" in df.columns:
            try:
                df = self._add_galactic_coordinates(df)
                logger.info("Added galactic coordinates")
            except Exception as e:
                logger.warning(f"Could not add galactic coordinates: {e}")

        # Galaxy morphology features
        if self.config["include_morphology"]:
            df = self._add_morphology_features(df)
            logger.info("Added morphology features")

        # Color features
        if self.config["include_colors"]:
            df = self._add_color_features(df)
            logger.info("Added color features")

        # Physical property calculations
        df = self._calculate_physical_properties(df)
        logger.info("Calculated physical properties")

        # Add cartesian coordinates if not already present and if possible
        if all(col in df.columns for col in ["ra", "dec"]) and not all(
            col in df.columns for col in ["x", "y", "z"]
        ):
            from astro_lab.data.transforms.astronomical import spherical_to_cartesian

            if "distance_pc" in df.columns:
                dist = df["distance_pc"]
            elif "distance_mpc" in df.columns:
                dist = df["distance_mpc"] * 1e6
            else:
                dist = None
            if dist is not None:
                x, y, z = spherical_to_cartesian(df["ra"], df["dec"], dist)
                df = df.with_columns(
                    [
                        pl.Series("x", x),
                        pl.Series("y", y),
                        pl.Series("z", z),
                    ]
                )
        return df

    def _add_galactic_coordinates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add galactic coordinates using astropy."""
        # Get RA/Dec values
        ra_vals = df["ra"].to_numpy()
        dec_vals = df["dec"].to_numpy()

        # Convert to galactic coordinates
        coords = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg, frame="icrs")
        galactic = coords.galactic

        return df.with_columns(
            [
                pl.Series("gal_l", galactic.l.degree),
                pl.Series("gal_b", galactic.b.degree),
            ]
        )

    def _add_morphology_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add galaxy morphology features."""
        # Find Sersic index column
        sersic_cols = ["sersic_n"]
        sersic_col = None
        for col in sersic_cols:
            if col in df.columns:
                sersic_col = col
                break

        if sersic_col:
            df = df.with_columns(
                [
                    # Disk-like (n~1) vs bulge-like (n~4)
                    ((pl.col(sersic_col) - 1) / 3).alias("bulge_fraction"),
                    # Exponential (n=1) vs de Vaucouleurs (n=4)
                    pl.when(pl.col(sersic_col) < 2.0)
                    .then(pl.lit("disk"))
                    .when(pl.col(sersic_col) > 3.0)
                    .then(pl.lit("bulge"))
                    .otherwise(pl.lit("intermediate"))
                    .alias("morphology_type"),
                ]
            )

        # Find effective radius column
        radius_cols = ["sersic_r50", "petro_r50", "petro_r90", "effective_radius"]
        radius_col = None
        for col in radius_cols:
            if col in df.columns:
                radius_col = col
                break

        if radius_col:
            df = df.with_columns([pl.col(radius_col).alias("effective_radius")])

        return df

    def _add_color_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add galaxy color features."""
        # Standard magnitude columns
        mag_cols = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z"]
        available_mags = [col for col in mag_cols if col in df.columns]

        if len(available_mags) >= 2:
            # Calculate colors
            colors = []
            for i in range(len(available_mags) - 1):
                color_name = f"{available_mags[i][4:]}_{available_mags[i + 1][4:]}_color"  # Remove 'mag_' prefix
                df = df.with_columns(
                    [
                        (
                            pl.col(available_mags[i]) - pl.col(available_mags[i + 1])
                        ).alias(color_name)
                    ]
                )
                colors.append(color_name)

            logger.info(f"Calculated colors: {colors}")

        return df

    def _calculate_physical_properties(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate physical properties of galaxies."""
        # Stellar mass (try different column names)
        mass_cols = ["mass", "stellar_mass", "mstar"]
        mass_col = None
        for col in mass_cols:
            if col in df.columns:
                mass_col = col
                break

        if mass_col:
            df = df.with_columns(
                [
                    pl.col(mass_col).alias("stellar_mass_log"),
                    (10 ** pl.col(mass_col)).cast(pl.Float64).alias("stellar_mass"),
                ]
            )

        # Size-mass relation
        if "effective_radius" in df.columns and "stellar_mass_log" in df.columns:
            df = df.with_columns(
                [
                    (
                        pl.col("effective_radius") / (pl.col("stellar_mass_log") - 10)
                    ).alias("size_mass_ratio")
                ]
            )

        # Surface brightness
        if "mag_r" in df.columns and "effective_radius" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("mag_r") + 5 * pl.col("effective_radius").log10()).alias(
                        "surface_brightness"
                    )
                ]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract galaxy features for ML keeping essential coordinates."""
        logger.info("Extracting ML features from NSA galaxy data...")

        # Apply column mapping first
        df = self._map_and_select_columns(df)

        feature_columns = []

        # Essential identifier
        id_cols = ["nsaid", "iau_name", "objid", "plateifu"]
        for col in id_cols:
            if col in df.columns:
                feature_columns.append(col)
                break

        # ALWAYS keep coordinate features for 3D processing
        essential_coords = ["ra", "dec"]
        for col in essential_coords:
            if col in df.columns:
                feature_columns.append(col)

        # Distance information
        distance_cols = ["z", "distance_pc", "distance_mpc"]
        for col in distance_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Galactic coordinates
        gal_cols = ["gal_l", "gal_b"]
        for col in gal_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Physical properties
        mass_cols = ["stellar_mass_log", "stellar_mass", "mass"]
        for col in mass_cols:
            if col in df.columns:
                feature_columns.append(col)
                break  # Only keep one mass column

        # Morphology
        morph_cols = [
            "sersic_n",
            "bulge_fraction",
            "effective_radius",
            "morphology_type",
            "size_mass_ratio",
        ]
        for col in morph_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Photometry
        photo_cols = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z"]
        for col in photo_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Colors
        color_cols = [col for col in df.columns if "_color" in col]
        feature_columns.extend(color_cols)

        # Surface brightness
        if "surface_brightness" in df.columns:
            feature_columns.append("surface_brightness")

        # Remove duplicates and keep available columns
        feature_columns = list(set(feature_columns))
        available_features = [col for col in feature_columns if col in df.columns]

        if not available_features:
            logger.warning("No feature columns found, keeping all columns")
            available_features = df.columns

        # Ensure we have essential coordinates
        if not all(col in available_features for col in ["ra", "dec"]):
            logger.error("Missing essential coordinate columns in feature extraction")
            # Add them back if they exist in the original dataframe
            for col in ["ra", "dec"]:
                if col in df.columns and col not in available_features:
                    available_features.append(col)

        # Select columns
        df = df.select(available_features)

        # Store feature information
        self.feature_names = available_features

        # Count feature types
        feature_types = {
            "coordinates": len(
                [c for c in available_features if c in ["ra", "dec", "gal_l", "gal_b"]]
            ),
            "physical": len(
                [
                    c
                    for c in available_features
                    if c in ["stellar_mass_log", "stellar_mass", "mass"]
                ]
            ),
            "morphology": len([c for c in available_features if c in morph_cols]),
            "magnitudes": len([c for c in available_features if c in photo_cols]),
            "colors": len([c for c in available_features if "_color" in c]),
        }

        self.stats = {
            "extract_features": {
                "num_features": len(available_features),
                "feature_types": feature_types,
            }
        }

        logger.info(f"Extracted {len(available_features)} features")
        logger.info(f"Feature types: {feature_types}")

        return df

    def get_galaxy_mass(self, df: pl.DataFrame) -> Optional[pl.Series]:
        """Get galaxy stellar mass if available."""
        if "stellar_mass" in df.columns:
            return df["stellar_mass"]
        elif "stellar_mass_log" in df.columns:
            return df["stellar_mass_log"].apply(lambda x: 10**x)
        else:
            return None

    def get_brightness_measure(self, df: pl.DataFrame) -> pl.Series:
        """Get brightness measure for visualization."""
        # Try different magnitude columns
        mag_cols = ["mag_r", "mag_g", "mag_i"]
        for col in mag_cols:
            if col in df.columns:
                return 25.0 - df[col]  # Convert magnitude to brightness

        # Default brightness
        return pl.Series([1.0] * len(df))
