"""
NASA-Sloan Atlas (NSA) Survey Preprocessor
==========================================

NSA data preprocessing with galaxy morphology and environment analysis.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.cosmology import Planck18
from sklearn.neighbors import NearestNeighbors

from .base import BaseSurveyProcessor

logger = logging.getLogger(__name__)


class NSAPreprocessor(BaseSurveyProcessor):
    """
    processor for NASA-Sloan Atlas galaxy survey data.

    Features:
    - Proper redshift to distance conversion using cosmology
    - Galaxy morphology parameters (Sérsic profile, concentration, etc.)
    - Environment analysis (local density, group membership)
    - Multi-band photometry with extinction corrections
    - Star formation rate and stellar mass estimates
    - Quality filtering for reliable galaxy samples
    """

    def __init__(self, survey_name: str = "nsa", cosmology=None):
        super().__init__(survey_name)

        # Cosmology for distance calculations
        self.cosmology = cosmology or Planck18

        # NSA quality cuts following Blanton et al. 2011
        self.z_min = 0.01  # Minimum redshift for reliable distances
        self.z_max = 0.15  # Maximum redshift for NSA completeness
        self.r_petro_min = 5.0  # Minimum Petrosian radius in arcsec

    def get_data_path(self) -> str:
        """Get path to NSA data file."""
        # NSA has a specific FITS file name
        fits_path = self.raw_dir / "nsa_v1_0_1.fits"
        if fits_path.exists():
            return fits_path

        # Fallback to parquet if it exists
        parquet_path = self.raw_dir / "nsa_v1_0_1.parquet"
        if parquet_path.exists():
            return parquet_path

        # Fallback to base method
        return super().get_data_path()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get NSA-specific default configuration."""
        base_config = super()._get_default_config()
        base_config.update(
            {
                "coordinate_system": "icrs",
                "distance_unit": "Mpc",  # Use Mpc for cosmological distances
                "filter_system": "AB",
                "redshift_limit": 0.15,
                "morphology_features": True,
                "environment_analysis": True,
                "extinction_correction": True,
            }
        )
        return base_config

    def get_coordinate_columns(self) -> List[str]:
        """NSA coordinate column names."""
        return ["ra", "dec", "z"]  # z = redshift

    def get_magnitude_columns(self) -> List[str]:
        """NSA photometry column names."""
        # NSA uses SDSS ugriz photometry with different apertures
        return [
            "elpetro_mag_u",  # Elliptical Petrosian magnitudes
            "elpetro_mag_g",
            "elpetro_mag_r",
            "elpetro_mag_i",
            "elpetro_mag_z",
        ]

    def apply_survey_specific_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply NSA-specific quality filters."""
        initial_count = len(df)

        # 1. Redshift range for reliable cosmological distances
        if "z" in df.columns:
            df = df.filter((pl.col("z") >= self.z_min) & (pl.col("z") <= self.z_max))

        # 2. Petrosian radius cut for reliable photometry
        if "petrorad_r" in df.columns:
            df = df.filter(pl.col("petrorad_r") > self.r_petro_min)

        # 3. Remove objects with bad photometry flags
        if "elpetro_mag_r" in df.columns:
            df = df.filter(
                pl.col("elpetro_mag_r").is_finite()
                & (pl.col("elpetro_mag_r") > 10)  # Reasonable magnitude range
                & (pl.col("elpetro_mag_r") < 25)
            )

        # 4. Good surface brightness profile fits
        if "sersic_n" in df.columns:
            df = df.filter(
                pl.col("sersic_n").is_finite()
                & (pl.col("sersic_n") > 0.2)
                & (pl.col("sersic_n") < 8.0)  # Reasonable Sérsic index range
            )

        # 5. Remove very extended galaxies (likely blends)
        if "elpetro_th50_r" in df.columns:
            df = df.filter(pl.col("elpetro_th50_r") < 30.0)  # 30 arcsec limit

        final_count = len(df)
        logger.info(
            f"NSA quality filters: {initial_count} → {final_count} galaxies "
            f"({final_count / initial_count * 100:.1f}% retained)"
        )

        return df

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates using cosmological distances."""

        # Check if we already have Cartesian coordinates (from preprocessing)
        if all(col in df.columns for col in ["x_mpc", "y_mpc", "z_mpc"]):
            coords = df.select(["x_mpc", "y_mpc", "z_mpc"]).to_numpy()
            return torch.tensor(coords, dtype=torch.float32)

        # Extract RA, Dec
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()

        # Try different redshift column names
        redshift_col = None
        for col_name in ["z", "redshift", "Z", "REDSHIFT", "z_helio", "z_cmb"]:
            if col_name in df.columns:
                redshift_col = col_name
                break

        if redshift_col is None:
            # No redshift available, use 2D coordinates with distance=0
            logger.warning("No redshift column found, using 2D coordinates")
            x = np.zeros_like(ra)  # Placeholder for distance
            y = np.zeros_like(ra)  # Placeholder for distance
            z = np.zeros_like(ra)  # Placeholder for distance
            return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

        redshift = df[redshift_col].to_numpy()

        # Convert redshift to comoving distance
        comoving_distances = self.cosmology.comoving_distance(redshift)

        # Create SkyCoord objects
        coords = SkyCoord(
            ra=ra * u.Unit("deg"),
            dec=dec * u.Unit("deg"),
            distance=Distance(comoving_distances),
            frame="icrs",
        )

        # Convert to Cartesian (comoving coordinates)
        cart = coords.cartesian
        x = cart.x.to(u.Mpc).value
        y = cart.y.to(u.Mpc).value
        z = cart.z.to(u.Mpc).value

        return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract comprehensive galaxy feature vector."""
        features = []
        feature_names = []

        # 1. Photometric features
        phot_features, phot_names = self._extract_photometry_features(df)
        if phot_features is not None:
            features.append(phot_features)
            feature_names.extend(phot_names)

        # 2. Morphology features
        morph_features, morph_names = self._extract_morphology_features(df)
        if morph_features is not None:
            features.append(morph_features)
            feature_names.extend(morph_names)

        # 3. Size and scale features
        size_features, size_names = self._extract_size_features(df)
        if size_features is not None:
            features.append(size_features)
            feature_names.extend(size_names)

        # 4. Physical properties
        phys_features, phys_names = self._extract_physical_features(df)
        if phys_features is not None:
            features.append(phys_features)
            feature_names.extend(phys_names)

        # 5. Environment features
        if self.config.get("environment_analysis", True):
            env_features, env_names = self._extract_environment_features(df)
            if env_features is not None:
                features.append(env_features)
                feature_names.extend(env_names)

        if not features:
            return torch.zeros(len(df), 1, dtype=torch.float32)

        # Combine all features
        combined_features = np.concatenate(features, axis=1)
        self.feature_names = feature_names

        return torch.tensor(combined_features, dtype=torch.float32)

    def preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply NSA-specific preprocessing to DataFrame.

        This includes:
        - Cosmological distance calculations
        - Color index calculations
        - Physical property derivations
        - Morphology indicators
        - Extinction corrections
        - Environment analysis
        """
        logger.info("Applying NSA-specific preprocessing...")

        # 1. Add cosmological distances
        df = self._add_cosmological_distances(df)

        # 2. Add color indices
        df = self._add_color_indices(df)

        # 3. Add physical properties
        df = self._add_physical_properties(df)

        # 4. Add morphology indicators
        df = self._add_morphology_indicators(df)

        # 5. Add physical sizes
        df = self._add_physical_sizes(df)

        # 6. Add Cartesian coordinates
        df = self._add_cartesian_coordinates(df)

        # 7. Apply extinction corrections
        df = self._apply_extinction_corrections(df)

        # 8. Calculate environment features
        df = self._calculate_environment_features(df)

        logger.info(f"NSA preprocessing complete: {len(df)} galaxies")
        return df

    def _extract_photometry_features(self, df: pl.DataFrame) -> tuple:
        """Extract photometric features from NSA data."""
        features = []
        names = []

        # Absolute magnitudes in different bands
        for band in ["u", "g", "r", "i", "z"]:
            col_name = f"elpetro_absmag_{band}"
            if col_name in df.columns:
                abs_mag = df[col_name].to_numpy()
                features.append(abs_mag.reshape(-1, 1))
                names.append(f"M_{band}")

        # Surface brightness
        if "elpetro_sb_r" in df.columns:  # r-band surface brightness
            sb_r = df["elpetro_sb_r"].to_numpy()
            features.append(sb_r.reshape(-1, 1))
            names.append("sb_r")

        # Mass-to-light ratio
        if "elpetro_mtol_r" in df.columns:
            mtol = np.log10(df["elpetro_mtol_r"].to_numpy() + 1e-10)
            features.append(mtol.reshape(-1, 1))
            names.append("log_mtol_r")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_morphology_features(self, df: pl.DataFrame) -> tuple:
        """Extract morphological features."""
        features = []
        names = []

        # Sérsic index (primary morphology indicator)
        if "sersic_n" in df.columns:
            sersic_n = df["sersic_n"].to_numpy()
            features.append(sersic_n.reshape(-1, 1))
            names.append("sersic_n")

        # Effective radius
        if "sersic_th50" in df.columns:
            th50 = df["sersic_th50"].to_numpy()
            features.append(np.log10(th50 + 1e-10).reshape(-1, 1))
            names.append("log_th50")

        # Axis ratio (shape)
        if "sersic_ba" in df.columns:
            ba = df["sersic_ba"].to_numpy()
            features.append(ba.reshape(-1, 1))
            names.append("axis_ratio")

        # Position angle
        if "sersic_phi" in df.columns:
            phi = df["sersic_phi"].to_numpy()
            # Convert to sin/cos to handle circular nature
            features.append(np.sin(np.radians(phi)).reshape(-1, 1))
            features.append(np.cos(np.radians(phi)).reshape(-1, 1))
            names.extend(["pa_sin", "pa_cos"])

        # Concentration parameter
        if "concentration_r" in df.columns:
            conc = df["concentration_r"].to_numpy()
            features.append(conc.reshape(-1, 1))
            names.append("concentration")

        # Asymmetry
        if "asymmetry_r" in df.columns:
            asym = df["asymmetry_r"].to_numpy()
            features.append(asym.reshape(-1, 1))
            names.append("asymmetry")

        # Smoothness (clumpiness indicator)
        if "smoothness_r" in df.columns:
            smooth = df["smoothness_r"].to_numpy()
            features.append(smooth.reshape(-1, 1))
            names.append("smoothness")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_size_features(self, df: pl.DataFrame) -> tuple:
        """Extract size-related features."""
        features = []
        names = []

        # Petrosian radius
        if "petrorad_r" in df.columns:
            petro_r = np.log10(df["petrorad_r"].to_numpy() + 1e-10)
            features.append(petro_r.reshape(-1, 1))
            names.append("log_petro_r")

        # Half-light radius
        if "elpetro_th50_r" in df.columns:
            th50_r = np.log10(df["elpetro_th50_r"].to_numpy() + 1e-10)
            features.append(th50_r.reshape(-1, 1))
            names.append("log_th50_r")

        # 90% light radius
        if "elpetro_th90_r" in df.columns:
            th90_r = np.log10(df["elpetro_th90_r"].to_numpy() + 1e-10)
            features.append(th90_r.reshape(-1, 1))
            names.append("log_th90_r")

        # Size ratio
        if all(col in df.columns for col in ["elpetro_th90_r", "elpetro_th50_r"]):
            size_ratio = (df["elpetro_th90_r"] / df["elpetro_th50_r"]).to_numpy()
            features.append(size_ratio.reshape(-1, 1))
            names.append("size_ratio")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_physical_features(self, df: pl.DataFrame) -> tuple:
        """Extract physical properties."""
        features = []
        names = []

        # Stellar mass
        if "elpetro_mass" in df.columns:
            stellar_mass = np.log10(df["elpetro_mass"].to_numpy() + 1e8)
            features.append(stellar_mass.reshape(-1, 1))
            names.append("log_stellar_mass")

        # Star formation rate
        if "sfr_ha" in df.columns:  # H-alpha SFR
            sfr = np.log10(df["sfr_ha"].to_numpy() + 1e-3)
            features.append(sfr.reshape(-1, 1))
            names.append("log_sfr_ha")

        # Specific star formation rate
        if all(col in df.columns for col in ["sfr_ha", "elpetro_mass"]):
            ssfr = np.log10((df["sfr_ha"] / df["elpetro_mass"]).to_numpy() + 1e-12)
            features.append(ssfr.reshape(-1, 1))
            names.append("log_ssfr")

        # Redshift (distance proxy)
        if "z" in df.columns:
            redshift = df["z"].to_numpy()
            features.append(redshift.reshape(-1, 1))
            names.append("redshift")

        # Velocity dispersion if available
        if "sigma" in df.columns:
            sigma = np.log10(df["sigma"].to_numpy() + 1e-10)
            features.append(sigma.reshape(-1, 1))
            names.append("log_sigma")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_environment_features(self, df: pl.DataFrame) -> tuple:
        """Extract environment-based features."""
        features = []
        names = []

        # Local density if calculated
        if "local_density" in df.columns:
            log_dens = np.log10(df["local_density"].to_numpy() + 1e-6)
            features.append(log_dens.reshape(-1, 1))
            names.append("log_local_density")

        # Distance to nearest neighbor
        if "nn_distance" in df.columns:
            nn_dist = np.log10(df["nn_distance"].to_numpy() + 1e-3)
            features.append(nn_dist.reshape(-1, 1))
            names.append("log_nn_distance")

        # Group membership indicator
        if "in_group" in df.columns:
            in_group = df["in_group"].to_numpy().astype(float)
            features.append(in_group.reshape(-1, 1))
            names.append("in_group")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _add_cosmological_distances(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add various cosmological distance measures."""
        redshifts = df["z"].to_numpy()

        # Comoving distance
        d_comoving = self.cosmology.comoving_distance(redshifts).value  # Mpc

        # Luminosity distance
        d_luminosity = self.cosmology.luminosity_distance(redshifts).value  # Mpc

        # Angular diameter distance
        d_angular = self.cosmology.angular_diameter_distance(redshifts).value  # Mpc

        # Distance modulus
        dist_mod = 5 * np.log10(d_luminosity * 1e6 / 10)  # Distance modulus

        df = df.with_columns(
            [
                pl.Series("d_comoving_mpc", d_comoving),
                pl.Series("d_luminosity_mpc", d_luminosity),
                pl.Series("d_angular_mpc", d_angular),
                pl.Series("distance_modulus", dist_mod),
            ]
        )

        return df

    def _add_color_indices(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add galaxy color indices."""
        # Standard SDSS colors
        colors = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]

        for band1, band2 in colors:
            mag1_col = f"elpetro_mag_{band1}"
            mag2_col = f"elpetro_mag_{band2}"

            if all(col in df.columns for col in [mag1_col, mag2_col]):
                color_name = f"{band1}_{band2}_color"
                df = df.with_columns(
                    (pl.col(mag1_col) - pl.col(mag2_col)).alias(color_name)
                )

        # NUV-r color for star formation indicator
        if all(col in df.columns for col in ["elpetro_mag_u", "elpetro_mag_r"]):
            df = df.with_columns(
                (pl.col("elpetro_mag_u") - pl.col("elpetro_mag_r")).alias("u_r_color")
            )

        return df

    def _add_physical_properties(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived physical properties."""

        # Convert apparent to absolute magnitudes
        if "distance_modulus" in df.columns:
            for band in ["u", "g", "r", "i", "z"]:
                app_mag_col = f"elpetro_mag_{band}"
                abs_mag_col = f"elpetro_absmag_{band}"

                if app_mag_col in df.columns and abs_mag_col not in df.columns:
                    df = df.with_columns(
                        (pl.col(app_mag_col) - pl.col("distance_modulus")).alias(
                            abs_mag_col
                        )
                    )

        # Stellar mass from r-band luminosity (rough estimate)
        if "elpetro_absmag_r" in df.columns and "elpetro_mass" not in df.columns:
            # mass-to-light ratio estimate
            Mr = df["elpetro_absmag_r"].to_numpy()
            Lr_solar = 10 ** (-0.4 * (Mr - 4.67))  # L_sun units
            # Assume M/L ~ 3 for typical galaxies
            mass_estimate = Lr_solar * 3.0

            df = df.with_columns(pl.Series("elpetro_mass_estimate", mass_estimate))

        return df

    def _add_morphology_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add morphological classification indicators."""

        # Early-type vs late-type based on Sérsic index
        if "sersic_n" in df.columns:
            df = df.with_columns(
                (pl.col("sersic_n") > 2.5).alias("early_type_candidate")
            )

        # Disk vs spheroid based on axis ratio and concentration
        if all(col in df.columns for col in ["sersic_ba", "concentration_r"]):
            df = df.with_columns(
                ((pl.col("sersic_ba") < 0.6) & (pl.col("concentration_r") < 3.0)).alias(
                    "disk_like"
                )
            )

        # Compact objects
        if "elpetro_th50_r" in df.columns:
            df = df.with_columns((pl.col("elpetro_th50_r") < 2.0).alias("compact"))

        return df

    def _add_physical_sizes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert angular sizes to physical sizes."""
        if all(col in df.columns for col in ["elpetro_th50_r", "d_angular_mpc"]):
            # Convert arcsec to kpc
            angular_size_arcsec = df["elpetro_th50_r"]
            angular_distance_mpc = df["d_angular_mpc"]

            # 1 arcsec = distance_mpc * (pi/180/3600) Mpc = distance_mpc * 4.848e-6 Mpc
            size_kpc = angular_size_arcsec * angular_distance_mpc * 4.848e-3  # kpc

            df = df.with_columns(pl.Series("th50_physical_kpc", size_kpc))

        return df

    def _add_cartesian_coordinates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add Cartesian coordinates in comoving frame."""
        if all(col in df.columns for col in ["ra", "dec", "d_comoving_mpc"]):
            ra = df["ra"].to_numpy()
            dec = df["dec"].to_numpy()
            distance = df["d_comoving_mpc"].to_numpy()

            # Convert to Cartesian
            coords = SkyCoord(
                ra=ra * u.Unit("deg"),
                dec=dec * u.Unit("deg"),
                distance=distance * u.Mpc,
                frame="icrs",
            )

            cart = coords.cartesian
            x_mpc = cart.x.to(u.Mpc).value
            y_mpc = cart.y.to(u.Mpc).value
            z_mpc = cart.z.to(u.Mpc).value

            df = df.with_columns(
                [
                    pl.Series("x_mpc", x_mpc),
                    pl.Series("y_mpc", y_mpc),
                    pl.Series("z_mpc", z_mpc),
                ]
            )

        return df

    def _apply_extinction_corrections(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Galactic extinction corrections."""
        # Apply extinction correction if available
        if "extinction_g" in df.columns and "extinction_r" in df.columns:
            # Apply extinction correction to magnitudes
            df = df.with_columns(
                [
                    pl.col("modelMag_g")
                    - pl.col("extinction_g").alias("modelMag_g_corrected"),
                    pl.col("modelMag_r")
                    - pl.col("extinction_r").alias("modelMag_r_corrected"),
                ]
            )
        else:
            # No extinction data available
            logger.warning("No extinction data available for NSA survey")

        return df

    def _calculate_environment_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate local environment features."""

        if all(col in df.columns for col in ["x_mpc", "y_mpc", "z_mpc"]):
            # Calculate local density using 5th nearest neighbor
            coords = df.select(["x_mpc", "y_mpc", "z_mpc"]).to_numpy()

            # Use 6 neighbors (including self) to get 5th nearest
            nbrs = NearestNeighbors(n_neighbors=6, algorithm="kd_tree").fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            # 5th nearest neighbor distance (index 5, excluding self at index 0)
            nn5_distance = distances[:, 5]  # Mpc

            # Local density: number density within sphere of 5th NN
            volume = (4 / 3) * np.pi * nn5_distance**3  # Mpc^3
            local_density = 5.0 / volume  # galaxies per Mpc^3

            # Distance to nearest neighbor (excluding self)
            nn1_distance = distances[:, 1]  # Mpc

            df = df.with_columns(
                [
                    pl.Series("local_density", local_density),
                    pl.Series("nn_distance", nn1_distance),
                ]
            )

        return df
