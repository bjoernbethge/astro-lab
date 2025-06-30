"""
SDSS data preprocessing with proper astronomical calculations
and spectroscopic features.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18

from astro_lab.tensors import SpectralTensorDict

from .base import BaseSurveyProcessor

logger = logging.getLogger(__name__)


class SDSSPreprocessor(BaseSurveyProcessor):
    """
    processor for SDSS survey data with proper astronomical handling.

    Features:
    - Proper cosmological distance calculations using astropy
    - SDSS ugriz photometry with extinction corrections
    - Spectroscopic redshift handling and quality assessment
    - Galaxy morphology and classification features
    - Stellar parameter extraction for stellar spectra
    - Proper SDSS coordinate systems and transformations
    """

    def __init__(self, survey_name: str = "sdss", data_config: Optional[Dict] = None):
        super().__init__(survey_name, data_config)

        # SDSS-specific configuration
        self.cosmology = Planck18  # Use Planck18 cosmology

        # SDSS magnitude zero points (AB system)
        self.zero_points = {"u": 22.5, "g": 22.5, "r": 22.5, "i": 22.5, "z": 22.5}

        # SDSS filter effective wavelengths (Angstroms)
        self.filter_wavelengths = {
            "u": 3543,
            "g": 4770,
            "r": 6231,
            "i": 7625,
            "z": 9134,
        }

    def get_coordinate_columns(self) -> List[str]:
        """Get SDSS coordinate column names."""
        return ["ra", "dec", "z"]  # RA, Dec, redshift

    def get_magnitude_columns(self) -> List[str]:
        """Get SDSS magnitude column names."""
        # SDSS uses multiple magnitude types
        mag_types = ["modelMag", "psfMag", "petro", "fiberMag", "fiber2Mag"]
        bands = ["u", "g", "r", "i", "z"]

        mag_cols = []
        for mag_type in mag_types:
            for band in bands:
                mag_cols.extend(
                    [f"{mag_type}_{band}", f"{band}_{mag_type}", f"{band}_mag", band]
                )

        return mag_cols

    def apply_survey_specific_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply SDSS-specific quality filters."""
        initial_count = len(df)

        # 1. Valid redshift range for extragalactic objects
        if "z" in df.columns:
            df = df.filter(
                (pl.col("z") >= 0)
                & (pl.col("z") <= 2.0)  # Reasonable redshift range
                & pl.col("z").is_finite()
            )

        # 2. SDSS quality flags
        if "clean" in df.columns:
            df = df.filter(pl.col("clean") == 1)

        # 3. Magnitude limits (bright star cutoff and faint limit)
        if "modelMag_r" in df.columns:
            df = df.filter(
                (pl.col("modelMag_r") > 10)  # Avoid bright stars
                & (pl.col("modelMag_r") < 24)  # Photometric limit
            )
        elif "r" in df.columns:
            df = df.filter((pl.col("r") > 10) & (pl.col("r") < 24))

        # 4. Remove objects with bad photometry
        if "flags_r" in df.columns:
            # SDSS flags: remove objects with bad pixels, saturation, etc.
            bad_flags = [
                1,  # CANONICAL_CENTER
                2,  # BRIGHT
                4,  # EDGE
                8,  # BLENDED
                16,  # CHILD
                64,  # SATURATED
                128,  # NOTCHECKED
                256,  # SUBTRACTED
                512,  # NOSTOKES
                1024,  # BADSKY
                2048,  # PETROFAINT
                4096,  # TOO_LARGE
                8192,  # DEBLENDED_AS_PSF
            ]
            for flag in bad_flags:
                df = df.filter((pl.col("flags_r") & flag) == 0)

        final_count = len(df)
        if final_count < initial_count:
            logger.info(
                f"🔍 SDSS quality filters: {initial_count} → {final_count} objects "
                f"({final_count / initial_count * 100:.1f}% retained)"
            )

        return df

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates from SDSS data using proper cosmology."""

        # Check if we already have Cartesian coordinates
        if all(col in df.columns for col in ["x_mpc", "y_mpc", "z_mpc"]):
            coords = (
                df.select(["x_mpc", "y_mpc", "z_mpc"]).to_numpy() * 1000
            )  # Convert to pc
            return torch.tensor(coords, dtype=torch.float32)

        # Use RA/Dec/redshift for cosmological distances
        if all(col in df.columns for col in ["ra", "dec", "z"]):
            return self._redshift_to_comoving_coordinates(df)

        # Fallback: photometric distance estimation
        elif (
            all(col in df.columns for col in ["ra", "dec"])
            and "modelMag_r" in df.columns
        ):
            return self._photometric_distance_estimation(df)

        raise ValueError("Insufficient coordinate information in SDSS data")

    def _redshift_to_comoving_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert redshift to comoving coordinates using proper cosmology."""

        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        redshift = df["z"].to_numpy()

        # Filter out invalid redshifts
        valid_mask = (redshift >= 0) & (redshift <= 10) & np.isfinite(redshift)

        # Calculate comoving distances using astropy cosmology
        comoving_distances = self.cosmology.comoving_distance(redshift)
        distance_pc = comoving_distances.to_value("pc")

        # Handle invalid redshifts
        distance_pc = np.where(
            valid_mask, distance_pc, 1000.0
        )  # Default 1 kpc for invalid

        # Convert to Cartesian coordinates
        coords = SkyCoord(
            ra=ra * u.Unit("deg"),
            dec=dec * u.Unit("deg"),
            distance=distance_pc * u.Unit("pc"),
            frame="icrs",
        )

        # Get Cartesian coordinates (keeping in ICRS for large-scale structure)
        cart = coords.cartesian
        x = cart.x.to_value("pc")
        y = cart.y.to_value("pc")
        z = cart.z.to_value("pc")

        return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

    def _photometric_distance_estimation(self, df: pl.DataFrame) -> torch.Tensor:
        """Estimate distances using r-band magnitude for galaxies without redshift."""

        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        r_mag = df["modelMag_r"].to_numpy()

        # Estimate absolute magnitude based on SDSS galaxy properties
        # Use color information if available for better estimation
        if "modelMag_g" in df.columns and "modelMag_r" in df.columns:
            g_r_color = (df["modelMag_g"] - df["modelMag_r"]).to_numpy()
            # Absolute magnitude relation: M_r ~ -20.5 - 0.5*(g-r-0.7)
            abs_mag_r = -20.5 - 0.5 * (g_r_color - 0.7)
        else:
            # Default to typical galaxy absolute magnitude
            abs_mag_r = -20.5

        # Distance modulus: m - M = 5*log10(d/10pc)
        distance_modulus = r_mag - abs_mag_r
        distance_pc = 10.0 ** ((distance_modulus + 5) / 5)

        # Convert to Cartesian
        coords = SkyCoord(
            ra=ra * u.Unit("deg"),
            dec=dec * u.Unit("deg"),
            distance=distance_pc * u.Unit("pc"),
            frame="icrs",
        )

        cart = coords.cartesian
        return torch.tensor(
            np.stack([cart.x.value, cart.y.value, cart.z.value], axis=1),
            dtype=torch.float32,
        )

    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract comprehensive feature vector from SDSS data."""
        features = []
        feature_names = []

        # 1. SDSS ugriz photometry
        phot_features, phot_names = self._extract_sdss_photometry(df)
        if phot_features is not None:
            features.append(phot_features)
            feature_names.extend(phot_names)

        # 2. Spectroscopic features
        spec_features, spec_names = self._extract_spectroscopic_features(df)
        if spec_features is not None:
            features.append(spec_features)
            feature_names.extend(spec_names)

        # 3. Morphological features
        morph_features, morph_names = self._extract_morphological_features(df)
        if morph_features is not None:
            features.append(morph_features)
            feature_names.extend(morph_names)

        # 4. Classification features
        class_features, class_names = self._extract_classification_features(df)
        if class_features is not None:
            features.append(class_features)
            feature_names.extend(class_names)

        if not features:
            return torch.zeros(len(df), 1, dtype=torch.float32)

        # Combine all features
        combined_features = np.concatenate(features, axis=1)
        self.feature_names = feature_names

        return torch.tensor(combined_features, dtype=torch.float32)

    def _extract_sdss_photometry(self, df: pl.DataFrame) -> tuple:
        """Extract SDSS ugriz photometry features."""
        features = []
        names = []

        # Prioritize modelMag (better for galaxies) over psfMag
        bands = ["u", "g", "r", "i", "z"]
        mag_data = {}

        for band in bands:
            # Try different magnitude types in order of preference
            for mag_type in ["modelMag", "psfMag", "petro", "fiberMag"]:
                col_name = f"{mag_type}_{band}"
                if col_name in df.columns:
                    mag_data[band] = df[col_name].to_numpy()
                    break

            # Try simple band name
            if band not in mag_data and band in df.columns:
                mag_data[band] = df[band].to_numpy()

        # Individual magnitudes
        for band in bands:
            if band in mag_data:
                features.append(mag_data[band].reshape(-1, 1))
                names.append(f"{band}_mag")

        # Color indices
        color_pairs = [
            ("u", "g"),
            ("g", "r"),
            ("r", "i"),
            ("i", "z"),
            ("u", "r"),
            ("g", "i"),
        ]
        for band1, band2 in color_pairs:
            if band1 in mag_data and band2 in mag_data:
                color = mag_data[band1] - mag_data[band2]
                features.append(color.reshape(-1, 1))
                names.append(f"{band1}_{band2}")

        # Extinction-corrected magnitudes if available
        for band in bands:
            extinction_col = f"extinction_{band}"
            if extinction_col in df.columns and band in mag_data:
                corrected_mag = mag_data[band] - df[extinction_col].to_numpy()
                features.append(corrected_mag.reshape(-1, 1))
                names.append(f"{band}_corrected")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_spectroscopic_features(self, df: pl.DataFrame) -> tuple:
        """Extract spectroscopic features from SDSS data."""
        features = []
        names = []

        # Redshift and velocity
        if "z" in df.columns:
            redshift = df["z"].to_numpy()
            features.append(redshift.reshape(-1, 1))
            names.append("redshift")

            # Convert to velocity
            c_km_s = 299792.458  # km/s
            velocity = redshift * c_km_s
            features.append(velocity.reshape(-1, 1))
            names.append("velocity_km_s")

        # Redshift error/quality
        if "zErr" in df.columns:
            z_err = df["zErr"].to_numpy()
            features.append(z_err.reshape(-1, 1))
            names.append("z_error")

            # Signal-to-noise in redshift
            if "z" in df.columns:
                z_snr = np.abs(redshift) / (z_err + 1e-10)
                features.append(z_snr.reshape(-1, 1))
                names.append("z_snr")

        # Spectral classification confidence
        if "zConf" in df.columns:
            z_conf = df["zConf"].to_numpy()
            features.append(z_conf.reshape(-1, 1))
            names.append("z_confidence")

        # Stellar parameters (if available)
        stellar_params = ["teff", "logg", "feh"]  # Temperature, gravity, metallicity
        for param in stellar_params:
            if param in df.columns:
                values = df[param].to_numpy()
                # Normalize stellar parameters
                if param == "teff":
                    values = values / 5772.0  # Solar units
                elif param == "logg":
                    values = values / 4.44  # Solar units
                # feh is already in dex units

                features.append(values.reshape(-1, 1))
                names.append(param)

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_morphological_features(self, df: pl.DataFrame) -> tuple:
        """Extract galaxy morphology features."""
        features = []
        names = []

        # Petrosian radius (galaxy size)
        if "petroRad_r" in df.columns:
            petro_rad = df["petroRad_r"].to_numpy()
            features.append(np.log10(petro_rad + 1e-3).reshape(-1, 1))  # Log scale
            names.append("log_petro_radius")

        # De Vaucouleurs fit parameters (elliptical component)
        if "deVRad_r" in df.columns:
            dev_rad = df["deVRad_r"].to_numpy()
            features.append(np.log10(dev_rad + 1e-3).reshape(-1, 1))
            names.append("log_dev_radius")

        if "deVAB_r" in df.columns:
            dev_ab = df["deVAB_r"].to_numpy()  # Axis ratio
            features.append(dev_ab.reshape(-1, 1))
            names.append("dev_axis_ratio")

        # Exponential fit parameters (disk component)
        if "expRad_r" in df.columns:
            exp_rad = df["expRad_r"].to_numpy()
            features.append(np.log10(exp_rad + 1e-3).reshape(-1, 1))
            names.append("log_exp_radius")

        if "expAB_r" in df.columns:
            exp_ab = df["expAB_r"].to_numpy()
            features.append(exp_ab.reshape(-1, 1))
            names.append("exp_axis_ratio")

        # Concentration index (central concentration)
        if "petroR50_r" in df.columns and "petroR90_r" in df.columns:
            r50 = df["petroR50_r"].to_numpy()
            r90 = df["petroR90_r"].to_numpy()
            concentration = r90 / (r50 + 1e-6)
            features.append(concentration.reshape(-1, 1))
            names.append("concentration_index")

        # Surface brightness
        if "modelMag_r" in df.columns and "petroRad_r" in df.columns:
            r_mag = df["modelMag_r"].to_numpy()
            petro_rad = df["petroRad_r"].to_numpy()
            # Surface brightness within effective radius
            surf_brightness = r_mag + 2.5 * np.log10(2 * np.pi * petro_rad**2 + 1e-6)
            features.append(surf_brightness.reshape(-1, 1))
            names.append("surface_brightness")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_classification_features(self, df: pl.DataFrame) -> tuple:
        """Extract object classification features."""
        features = []
        names = []

        # Object type (star/galaxy classification)
        if "type" in df.columns:
            obj_type = df["type"].to_numpy()
            # Convert to numerical: 6=star, 3=galaxy
            is_star = (obj_type == 6).astype(float)
            is_galaxy = (obj_type == 3).astype(float)
            features.extend([is_star.reshape(-1, 1), is_galaxy.reshape(-1, 1)])
            names.extend(["is_star", "is_galaxy"])

        # Spectral class (for stars)
        if "subClass" in df.columns:
            subclass = df["subClass"].to_numpy()
            features.append(subclass.reshape(-1, 1))
            names.append("spectral_subclass")

        # Galaxy spectral class (emission/absorption lines)
        if "class" in df.columns:
            spec_class = df["class"].to_numpy()
            # SDSS classes: GALAXY, QSO, STAR
            features.append(spec_class.reshape(-1, 1))
            names.append("spectral_class")

        # Star formation indicators
        if "h_alpha_flux" in df.columns:
            ha_flux = df["h_alpha_flux"].to_numpy()
            # Log of H-alpha flux (star formation rate indicator)
            features.append(np.log10(np.abs(ha_flux) + 1e-20).reshape(-1, 1))
            names.append("log_ha_flux")

        # AGN indicators
        if "nii_6584_flux" in df.columns and "h_alpha_flux" in df.columns:
            nii_flux = df["nii_6584_flux"].to_numpy()
            ha_flux = df["h_alpha_flux"].to_numpy()
            # [NII]/H-alpha ratio (AGN diagnostic)
            nii_ha = np.log10(np.abs(nii_flux) / (np.abs(ha_flux) + 1e-20))
            features.append(nii_ha.reshape(-1, 1))
            names.append("nii_ha_ratio")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply comprehensive SDSS-specific preprocessing."""

        # 1. Standardize magnitude column names
        df = self._standardize_magnitude_columns(df)

        # 2. Calculate extinction-corrected magnitudes
        df = self._apply_extinction_corrections(df)

        # 3. Add comprehensive color indices
        df = self._add_color_indices(df)

        # 4. Add morphological parameters
        df = self._add_morphological_parameters(df)

        # 5. Add distance estimates
        df = self._add_distance_estimates(df)

        # 6. Add object quality flags
        df = self._add_quality_flags(df)

        # 7. Add Cartesian coordinates
        if all(col in df.columns for col in ["ra", "dec", "z"]):
            df = self._add_cartesian_coordinates(df)

        return df

    def _standardize_magnitude_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize SDSS magnitude column names."""
        # Map various SDSS magnitude naming conventions to standard names
        mag_mappings = {
            "u": ["modelMag_u", "psfMag_u", "u_mag", "umag"],
            "g": ["modelMag_g", "psfMag_g", "g_mag", "gmag"],
            "r": ["modelMag_r", "psfMag_r", "r_mag", "rmag"],
            "i": ["modelMag_i", "psfMag_i", "i_mag", "imag"],
            "z": ["modelMag_z", "psfMag_z", "z_mag", "zmag"],
        }

        for standard_name, possible_names in mag_mappings.items():
            if standard_name not in df.columns:
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        df = df.with_columns(pl.col(possible_name).alias(standard_name))
                        break

        return df

    def _apply_extinction_corrections(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Galactic extinction corrections to SDSS photometry."""
        bands = ["u", "g", "r", "i", "z"]

        for band in bands:
            extinction_col = f"extinction_{band}"
            mag_col = band
            corrected_col = f"{band}_corrected"

            if mag_col in df.columns and extinction_col in df.columns:
                df = df.with_columns(
                    (pl.col(mag_col) - pl.col(extinction_col)).alias(corrected_col)
                )

        return df

    def _add_color_indices(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add comprehensive SDSS color indices."""

        # Standard colors
        standard_colors = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]

        for band1, band2 in standard_colors:
            color_name = f"{band1}_{band2}"
            if (
                band1 in df.columns
                and band2 in df.columns
                and color_name not in df.columns
            ):
                df = df.with_columns((pl.col(band1) - pl.col(band2)).alias(color_name))

        # Additional useful colors
        additional_colors = [("u", "r"), ("g", "i"), ("u", "z")]
        for band1, band2 in additional_colors:
            color_name = f"{band1}_{band2}"
            if (
                band1 in df.columns
                and band2 in df.columns
                and color_name not in df.columns
            ):
                df = df.with_columns((pl.col(band1) - pl.col(band2)).alias(color_name))

        return df

    def _add_morphological_parameters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived morphological parameters."""

        # Concentration index
        if "petroR50_r" in df.columns and "petroR90_r" in df.columns:
            df = df.with_columns(
                (pl.col("petroR90_r") / pl.col("petroR50_r")).alias(
                    "concentration_index"
                )
            )

        # Asymmetry (if available)
        if "Q_r" in df.columns and "U_r" in df.columns:
            df = df.with_columns(
                (pl.col("Q_r") ** 2 + pl.col("U_r") ** 2) ** (0.5).alias("asymmetry")
            )

        # Surface brightness
        if "r" in df.columns and "petroRad_r" in df.columns:
            df = df.with_columns(
                (
                    pl.col("r") + 2.5 * (2 * np.pi * pl.col("petroRad_r") ** 2).log10()
                ).alias("surf_brightness_r")
            )

        return df

    def _add_distance_estimates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add various distance estimates."""

        # Luminosity distance from redshift
        if "z" in df.columns:
            z_values = df["z"].to_numpy()
            lum_distances = self.cosmology.luminosity_distance(z_values)

            df = df.with_columns(
                [
                    pl.Series("lum_dist_mpc", lum_distances.to(u.Mpc).value),
                    pl.Series(
                        "dist_modulus",
                        5 * np.log10(lum_distances.to_value("pc") / 10),
                    ),
                ]
            )

        # Photometric distance estimate (if no spec-z)
        if "r" in df.columns and "z" not in df.columns:
            # Crude photometric distance for galaxies
            r_mag = df["r"].to_numpy()
            abs_mag_est = -20.5  # Typical galaxy
            dist_mod = r_mag - abs_mag_est
            phot_dist_mpc = 10 ** ((dist_mod + 5) / 5) / 1e6

            df = df.with_columns(pl.Series("phot_dist_mpc", phot_dist_mpc))

        return df

    def _add_quality_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add data quality flags."""

        # Good photometry flag
        if "flags_r" in df.columns:
            df = df.with_columns((pl.col("flags_r") == 0).alias("good_photometry"))

        # Reliable redshift flag
        if "zConf" in df.columns:
            df = df.with_columns((pl.col("zConf") > 0.95).alias("reliable_redshift"))

        # Galaxy/star classification confidence
        if "type" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("type") == 3).alias("is_galaxy"),
                    (pl.col("type") == 6).alias("is_star"),
                ]
            )

        return df

    def _add_cartesian_coordinates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add Cartesian coordinates in Mpc."""
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        z = df["z"].to_numpy()

        # Calculate comoving distances
        comoving_dist = self.cosmology.comoving_distance(z)
        dist_mpc = comoving_dist.to(u.Mpc).value

        # Convert to Cartesian (Mpc)
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        x_mpc = dist_mpc * np.cos(dec_rad) * np.cos(ra_rad)
        y_mpc = dist_mpc * np.cos(dec_rad) * np.sin(ra_rad)
        z_mpc = dist_mpc * np.sin(dec_rad)

        df = df.with_columns(
            [
                pl.Series("x_mpc", x_mpc),
                pl.Series("y_mpc", y_mpc),
                pl.Series("z_mpc", z_mpc),
                pl.Series("comoving_dist_mpc", dist_mpc),
            ]
        )

        return df

    def extract_spectral(self, df: pl.DataFrame) -> Optional[SpectralTensorDict]:
        """Extract spectral data if SDSS spectra are available."""
        # Extract spectral data if available
        if "spectroFlux" in df.columns:
            # Extract spectral flux data
            spectral_cols = [col for col in df.columns if col.startswith("spectroFlux")]
            if spectral_cols:
                spectral_data = df.select(spectral_cols).to_numpy()
                # Convert to spectral tensor format
                wavelengths = np.arange(3800, 9200, 1)  # SDSS wavelength range
                return SpectralTensorDict(
                    wavelengths=torch.tensor(
                        wavelengths[: len(spectral_cols)], dtype=torch.float32
                    ),
                    fluxes=torch.tensor(spectral_data, dtype=torch.float32),
                    flux_errors=None,
                )
        else:
            # No spectral data available
            logger.warning("No spectral data available for SDSS survey")
            return None
