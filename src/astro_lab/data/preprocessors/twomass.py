"""
2MASS Survey Preprocessor
=========================

2MASS (Two Micron All-Sky Survey) data preprocessing with proper astronomical handling.
"""

import logging
from typing import List

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord

from .base import BaseSurveyProcessor

logger = logging.getLogger(__name__)


class TwoMASSPreprocessor(BaseSurveyProcessor):
    """
    processor for 2MASS survey data with proper astronomical handling.

    Features:
    - Proper near-infrared photometry handling with Vega system
    - Stellar classification using NIR colors
    - Extinction corrections using NIR advantage
    - Proper motion cross-matching preparation
    - Quality filtering based on 2MASS recommendations
    - Temperature and metallicity estimation from colors
    """

    def __init__(self, survey_name: str = "twomass"):
        super().__init__(survey_name)

        # 2MASS-specific configuration
        self.twomass_bands = ["J", "H", "Ks"]
        self.twomass_wavelengths = {  # Central wavelengths in microns
            "J": 1.235,
            "H": 1.662,
            "Ks": 2.159,
        }

        # Quality thresholds
        self.min_snr = 10.0  # Higher SNR for NIR
        self.max_uncertainty = 0.2  # Maximum photometric uncertainty (mag)

        # Stellar classification color boundaries
        self.giant_jh_limit = 0.75  # J-H > 0.75 for giants
        self.dwarf_jks_limit = 0.95  # J-Ks < 0.95 for dwarfs

    def get_coordinate_columns(self) -> List[str]:
        """Get 2MASS coordinate column names."""
        return ["ra", "dec", "j_m"]  # Use J magnitude for distance estimation

    def get_magnitude_columns(self) -> List[str]:
        """Get 2MASS magnitude column names."""
        # 2MASS Point Source Catalog standard names
        return [
            "j_m",  # J-band magnitude
            "h_m",  # H-band magnitude
            "k_m",  # Ks-band magnitude
        ]

    def apply_survey_specific_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply 2MASS-specific quality filters."""
        initial_count = len(df)

        # 1. Photometric quality flags (ph_qual)
        if "ph_qual" in df.columns:
            # Keep only sources with quality A, B, or C in all bands
            good_qual_mask = (
                pl.col("ph_qual").str.slice(0, 1).is_in(["A", "B", "C"])  # J band
                & pl.col("ph_qual").str.slice(1, 1).is_in(["A", "B", "C"])  # H band
                & pl.col("ph_qual").str.slice(2, 1).is_in(["A", "B", "C"])  # Ks band
            )
            df = df.filter(good_qual_mask)

        # 2. Read flags (rd_flg) - require good reads in all bands
        if "rd_flg" in df.columns:
            # Each character represents number of reads: 1,2,4,6,9 are good
            good_reads = ["1", "2", "4", "6", "9"]
            read_mask = (
                pl.col("rd_flg").str.slice(0, 1).is_in(good_reads)
                & pl.col("rd_flg").str.slice(1, 1).is_in(good_reads)
                & pl.col("rd_flg").str.slice(2, 1).is_in(good_reads)
            )
            df = df.filter(read_mask)

        # 3. Blend flags (bl_flg) - avoid heavily blended sources
        if "bl_flg" in df.columns:
            # 0 = no blending, 1 = minor blending (acceptable)
            blend_mask = (
                pl.col("bl_flg").str.slice(0, 1).is_in(["0", "1"])
                & pl.col("bl_flg").str.slice(1, 1).is_in(["0", "1"])
                & pl.col("bl_flg").str.slice(2, 1).is_in(["0", "1"])
            )
            df = df.filter(blend_mask)

        # 4. Contamination flags (cc_flg) - avoid contaminated sources
        if "cc_flg" in df.columns:
            # 0 = no contamination, p = persistence, d = diffraction spike
            clean_mask = (
                pl.col("cc_flg").str.slice(0, 1).is_in(["0"])
                & pl.col("cc_flg").str.slice(1, 1).is_in(["0"])
                & pl.col("cc_flg").str.slice(2, 1).is_in(["0"])
            )
            df = df.filter(clean_mask)

        # 5. Photometric uncertainty thresholds
        error_cols = ["j_cmsig", "h_cmsig", "k_cmsig"]
        for col in error_cols:
            if col in df.columns:
                df = df.filter(
                    pl.col(col).is_null() | (pl.col(col) <= self.max_uncertainty)
                )

        # 6. Remove sources with invalid magnitudes
        mag_cols = self.get_magnitude_columns()
        for col in mag_cols:
            if col in df.columns:
                df = df.filter(
                    pl.col(col).is_not_null()
                    & pl.col(col).is_finite()
                    & (pl.col(col) > 0)
                    & (pl.col(col) < 20)  # Reasonable NIR magnitude limit
                )

        # 7. Galaxy contamination flag - keep point sources for stellar analysis
        if "gal_contam" in df.columns:
            df = df.filter(pl.col("gal_contam") == 0)

        # 8. Extended source flag
        if "ext_key" in df.columns:
            df = df.filter(pl.col("ext_key").is_null())  # Keep point sources

        final_count = len(df)
        if final_count < initial_count:
            logger.info(
                f"🔍 2MASS quality filters: {initial_count} → {final_count} sources "
                f"({final_count / initial_count * 100:.1f}% retained)"
            )

        return df

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates from 2MASS data with distance estimation."""

        # Check if we already have Cartesian coordinates
        if all(col in df.columns for col in ["x_pc", "y_pc", "z_pc"]):
            coords = df.select(["x_pc", "y_pc", "z_pc"]).to_numpy()
            return torch.tensor(coords, dtype=torch.float32)

        # Convert from RA/Dec using NIR photometric distance estimation
        if all(col in df.columns for col in ["ra", "dec"]):
            return self._nir_photometric_distances(df)

        raise ValueError("Insufficient coordinate information in 2MASS data")

    def _nir_photometric_distances(self, df: pl.DataFrame) -> torch.Tensor:
        """Estimate distances using near-infrared photometry."""

        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()

        # Use Ks magnitude for distance estimation (least affected by extinction)
        ks_mag = None
        for col in ["k_m", "ks_mag", "Ks", "k"]:
            if col in df.columns:
                ks_mag = df[col].to_numpy()
                break

        if ks_mag is None:
            # Fallback to J or H
            for col in ["j_m", "h_m", "j_mag", "h_mag"]:
                if col in df.columns:
                    ks_mag = df[col].to_numpy()
                    break

        if ks_mag is None:
            # Fallback to fixed distance
            distances = np.full(len(df), 5000.0)  # 5 kpc default
        else:
            # Get color information for better absolute magnitude estimation
            j_ks_color = None
            if "j_m" in df.columns and "k_m" in df.columns:
                j_mag = df["j_m"].to_numpy()
                j_ks_color = j_mag - ks_mag

            # Estimate absolute Ks magnitude based on color and stellar type
            if j_ks_color is not None:
                # Use J-Ks color to estimate stellar type and absolute magnitude
                # Based on Bessell & Brett (1988) and subsequent calibrations
                abs_ks = np.where(
                    j_ks_color < 0.5,  # Blue/early-type stars
                    -1.0 + 4.0 * j_ks_color,  # Main sequence relation
                    np.where(
                        j_ks_color < 1.0,  # Solar-type stars
                        1.0 + 2.0 * (j_ks_color - 0.5),
                        np.where(
                            j_ks_color < 1.5,  # Red dwarfs
                            2.0 + 1.5 * (j_ks_color - 1.0),
                            np.where(
                                j_ks_color > 2.0,  # Giants/AGB stars
                                -3.0 + 1.0 * j_ks_color,
                                3.0,  # Default red dwarf
                            ),
                        ),
                    ),
                )
            else:
                # Default to main sequence star
                abs_ks = 3.0

            # Distance modulus and distance
            distance_modulus = ks_mag - abs_ks
            distances = 10 ** ((distance_modulus + 5) / 5)  # Distance in pc

            # Apply reasonable limits
            distances = np.clip(distances, 10, 100000.0)  # 10 pc to 100 kpc

        # Create coordinates
        coords = SkyCoord(
            ra=ra * u.Unit("deg"),
            dec=dec * u.Unit("deg"),
            distance=distances * u.Unit("pc"),
            frame="icrs",
        )

        # Convert to Galactocentric for cosmic web analysis
        from astropy.coordinates import Galactocentric

        galcen = coords.transform_to(Galactocentric())

        # Extract Cartesian coordinates
        x = galcen.x.to_value("pc")
        y = galcen.y.to_value("pc")
        z = galcen.z.to_value("pc")

        return torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract comprehensive feature vector from 2MASS data."""
        features = []
        feature_names = []

        # 1. Near-infrared photometry features
        nir_features, nir_names = self._extract_nir_features(df)
        if nir_features is not None:
            features.append(nir_features)
            feature_names.extend(nir_names)

        # 2. Color indices and stellar type indicators
        color_features, color_names = self._extract_color_features(df)
        if color_features is not None:
            features.append(color_features)
            feature_names.extend(color_names)

        # 3. Quality and variability features
        quality_features, quality_names = self._extract_quality_features(df)
        if quality_features is not None:
            features.append(quality_features)
            feature_names.extend(quality_names)

        # 4. Stellar classification features
        stellar_features, stellar_names = self._extract_stellar_features(df)
        if stellar_features is not None:
            features.append(stellar_features)
            feature_names.extend(stellar_names)

        if not features:
            return torch.zeros(len(df), 1, dtype=torch.float32)

        # Combine all features
        combined_features = np.concatenate(features, axis=1)
        self.feature_names = feature_names

        return torch.tensor(combined_features, dtype=torch.float32)

    def _extract_nir_features(self, df: pl.DataFrame) -> tuple:
        """Extract near-infrared photometry features."""
        features = []
        names = []

        # 2MASS magnitudes (Vega system)
        mag_cols = self.get_magnitude_columns()
        for i, col in enumerate(mag_cols):
            if col in df.columns:
                mags = df[col].to_numpy()
                mags = np.nan_to_num(mags, nan=99.0)  # Use 99 for non-detections
                features.append(mags.reshape(-1, 1))
                names.append(f"{self.twomass_bands[i]}_mag")

        # Photometric uncertainties
        error_cols = ["j_cmsig", "h_cmsig", "k_cmsig"]
        for i, col in enumerate(error_cols):
            if col in df.columns:
                errors = df[col].to_numpy()
                errors = np.nan_to_num(errors, nan=0.1)
                features.append(errors.reshape(-1, 1))
                names.append(f"{self.twomass_bands[i]}_error")

        # Signal-to-noise ratios
        if all(col in df.columns for col in ["j_m", "j_cmsig"]):
            j_snr = 1.086 / df["j_cmsig"]  # Convert from mag error to SNR
            features.append(np.log10(j_snr + 1).to_numpy().reshape(-1, 1))
            names.append("log_J_snr")

        # Total near-infrared luminosity indicator
        if "k_m" in df.columns:
            # Use Ks as proxy for total NIR luminosity
            k_mag = df["k_m"].to_numpy()
            # Convert to relative luminosity (solar = 0)
            rel_luminosity = 4.83 - k_mag  # Ks_sun = 4.83 (Vega)
            features.append(rel_luminosity.reshape(-1, 1))
            names.append("nir_luminosity")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_color_features(self, df: pl.DataFrame) -> tuple:
        """Extract NIR color indices and stellar indicators."""
        features = []
        names = []

        mag_cols = ["j_m", "h_m", "k_m"]
        mags = {}

        # Extract available magnitudes
        for i, col in enumerate(mag_cols):
            if col in df.columns:
                mags[self.twomass_bands[i]] = df[col].to_numpy()

        # Standard 2MASS colors
        color_pairs = [("J", "H"), ("H", "Ks"), ("J", "Ks")]

        for band1, band2 in color_pairs:
            if band1 in mags and band2 in mags:
                color = mags[band1] - mags[band2]
                color = np.nan_to_num(color, nan=0.0)
                features.append(color.reshape(-1, 1))
                names.append(f"{band1}_{band2}_color")

        # Infrared excess parameter (J-Ks vs H-Ks)
        if "J" in mags and "H" in mags and "Ks" in mags:
            j_ks = mags["J"] - mags["Ks"]
            h_ks = mags["H"] - mags["Ks"]
            # Deviation from main sequence
            excess = j_ks - 2.0 * h_ks  # Empirical relation for MS stars
            features.append(excess.reshape(-1, 1))
            names.append("infrared_excess")

        # Temperature indicator from J-Ks (Bessell & Brett 1988)
        if "J" in mags and "Ks" in mags:
            j_ks = mags["J"] - mags["Ks"]
            # Empirical temperature relation (valid for dwarfs)
            log_teff = 3.981 - 0.324 * j_ks  # Log10(Teff)
            temp_indicator = 10**log_teff / 5772  # Normalized to solar
            features.append(temp_indicator.reshape(-1, 1))
            names.append("temperature_indicator")

        # Metallicity indicator from colors (rough approximation)
        if "J" in mags and "H" in mags and "Ks" in mags:
            j_h = mags["J"] - mags["H"]
            h_ks = mags["H"] - mags["Ks"]
            # Simplified metallicity indicator
            metallicity_proxy = j_h - 1.5 * h_ks
            features.append(metallicity_proxy.reshape(-1, 1))
            names.append("metallicity_proxy")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_quality_features(self, df: pl.DataFrame) -> tuple:
        """Extract quality and observational features."""
        features = []
        names = []

        # Photometric quality scores
        if "ph_qual" in df.columns:
            # Convert quality letters to numerical scores
            quality_map = {
                "A": 1.0,
                "B": 0.8,
                "C": 0.6,
                "D": 0.4,
                "E": 0.2,
                "F": 0.0,
                "U": 0.0,
            }

            ph_qual_str = df["ph_qual"].to_list()
            j_qual = np.array(
                [quality_map.get(q[0] if len(q) > 0 else "U", 0.0) for q in ph_qual_str]
            )
            h_qual = np.array(
                [quality_map.get(q[1] if len(q) > 1 else "U", 0.0) for q in ph_qual_str]
            )
            k_qual = np.array(
                [quality_map.get(q[2] if len(q) > 2 else "U", 0.0) for q in ph_qual_str]
            )

            features.extend(
                [j_qual.reshape(-1, 1), h_qual.reshape(-1, 1), k_qual.reshape(-1, 1)]
            )
            names.extend(["J_quality", "H_quality", "Ks_quality"])

            # Average quality across bands
            avg_quality = (j_qual + h_qual + k_qual) / 3.0
            features.append(avg_quality.reshape(-1, 1))
            names.append("average_quality")

        # Astrometric fit quality
        if "ccm_flg" in df.columns:
            # Clean photometry flag
            ccm_flag = df["ccm_flg"].to_numpy().astype(float)
            features.append(ccm_flag.reshape(-1, 1))
            names.append("clean_photometry")

        # Variability indicators
        if "dup_src" in df.columns:
            # Duplicate source flag (potential variable)
            dup_flag = df["dup_src"].to_numpy().astype(float)
            features.append(dup_flag.reshape(-1, 1))
            names.append("potential_variable")

        # Proximity to extended sources
        if "prox" in df.columns:
            proximity = df["prox"].to_numpy().astype(float)
            features.append(proximity.reshape(-1, 1))
            names.append("extended_source_proximity")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_stellar_features(self, df: pl.DataFrame) -> tuple:
        """Extract stellar classification features."""
        features = []
        names = []

        # Stellar type classification based on colors
        if all(col in df.columns for col in ["j_m", "h_m", "k_m"]):
            j_mag = df["j_m"].to_numpy()
            h_mag = df["h_m"].to_numpy()
            k_mag = df["k_m"].to_numpy()

            j_h = j_mag - h_mag
            h_k = h_mag - k_mag
            j_k = j_mag - k_mag

            # Giant star indicator (red and luminous)
            giant_prob = np.where((j_h > self.giant_jh_limit) & (j_k > 0.8), 1.0, 0.0)
            features.append(giant_prob.reshape(-1, 1))
            names.append("giant_probability")

            # Main sequence dwarf indicator
            dwarf_prob = np.where((j_k < self.dwarf_jks_limit) & (j_h < 0.7), 1.0, 0.0)
            features.append(dwarf_prob.reshape(-1, 1))
            names.append("dwarf_probability")

            # Carbon star indicator (very red)
            carbon_prob = np.where(j_k > 1.4, 1.0, 0.0)
            features.append(carbon_prob.reshape(-1, 1))
            names.append("carbon_star_probability")

            # Brown dwarf candidate (very red and faint)
            if "k_m" in df.columns:
                bd_prob = np.where((j_k > 1.2) & (k_mag > 12), 1.0, 0.0)
                features.append(bd_prob.reshape(-1, 1))
                names.append("brown_dwarf_candidate")

        # Young Stellar Object indicators (infrared excess)
        if "infrared_excess" in df.columns or all(
            col in df.columns for col in ["j_m", "h_m", "k_m"]
        ):
            if "infrared_excess" not in df.columns:
                # Calculate if not already done
                j_k = df["j_m"] - df["k_m"]
                h_k = df["h_m"] - df["k_m"]
                excess = j_k - 2.0 * h_k
            else:
                excess = df["infrared_excess"].to_numpy()

            # YSO probability based on excess
            yso_prob = np.where(excess > 0.3, 1.0, 0.0)
            features.append(yso_prob.reshape(-1, 1))
            names.append("yso_candidate")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply 2MASS-specific preprocessing."""

        # 1. Standardize column names
        df = self._standardize_column_names(df)

        # 2. Calculate NIR colors
        df = self._add_nir_colors(df)

        # 3. Add stellar classification flags
        df = self._add_stellar_classification(df)

        # 4. Handle non-detections
        df = self._handle_nondetections(df)

        # 5. Add derived quantities
        df = self._add_derived_quantities(df)

        return df

    def _standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize 2MASS column names across different catalogs."""
        # Map various 2MASS column naming conventions
        column_mapping = {
            # Magnitudes
            "jmag": "j_m",
            "hmag": "h_m",
            "kmag": "k_m",
            "j_mag": "j_m",
            "h_mag": "h_m",
            "k_mag": "k_m",
            "J": "j_m",
            "H": "h_m",
            "K": "k_m",
            "Ks": "k_m",
            # Errors
            "j_err": "j_cmsig",
            "h_err": "h_cmsig",
            "k_err": "k_cmsig",
            "e_jmag": "j_cmsig",
            "e_hmag": "h_cmsig",
            "e_kmag": "k_cmsig",
            # Coordinates
            "RA": "ra",
            "DEC": "dec",
            "_RAJ2000": "ra",
            "_DEJ2000": "dec",
        }

        # Apply mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        return df

    def _add_nir_colors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add near-infrared color indices."""

        # Standard 2MASS colors
        color_definitions = [
            ("j_h", "j_m", "h_m"),
            ("h_k", "h_m", "k_m"),
            ("j_k", "j_m", "k_m"),
        ]

        for color_name, mag1, mag2 in color_definitions:
            if (
                mag1 in df.columns
                and mag2 in df.columns
                and color_name not in df.columns
            ):
                df = df.with_columns((pl.col(mag1) - pl.col(mag2)).alias(color_name))

        return df

    def _add_stellar_classification(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add stellar classification flags based on 2MASS colors."""

        # Giant star classification
        if all(col in df.columns for col in ["j_h", "j_k"]):
            df = df.with_columns(
                [
                    (
                        (pl.col("j_h") > self.giant_jh_limit) & (pl.col("j_k") > 0.8)
                    ).alias("giant_candidate"),
                    (
                        (pl.col("j_k") < self.dwarf_jks_limit) & (pl.col("j_h") < 0.7)
                    ).alias("dwarf_candidate"),
                    (pl.col("j_k") > 1.4).alias("carbon_star_candidate"),
                ]
            )

        # Young Stellar Object classification
        if all(col in df.columns for col in ["j_m", "h_m", "k_m"]):
            # Infrared excess indicator
            df = df.with_columns(
                (
                    (pl.col("j_m") - pl.col("k_m"))
                    - 2.0 * (pl.col("h_m") - pl.col("k_m"))
                ).alias("ir_excess")
            )

            df = df.with_columns((pl.col("ir_excess") > 0.3).alias("yso_candidate"))

        # Brown dwarf candidates (very red and faint)
        if all(col in df.columns for col in ["j_k", "k_m"]):
            df = df.with_columns(
                ((pl.col("j_k") > 1.2) & (pl.col("k_m") > 12)).alias(
                    "brown_dwarf_candidate"
                )
            )

        return df

    def _handle_nondetections(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle non-detections and upper limits in 2MASS data."""

        # 2MASS detection limits (approximate 10-sigma limits)
        detection_limits = {
            "j_m": 15.8,  # J-band limit
            "h_m": 15.1,  # H-band limit
            "k_m": 14.3,  # Ks-band limit
        }

        mag_cols = ["j_m", "h_m", "k_m"]

        for col in mag_cols:
            if col in df.columns:
                limit = detection_limits.get(col, 99.0)

                # Add upper limit flag
                df = df.with_columns(pl.col(col).is_null().alias(f"{col}_upper_limit"))

                # Fill nulls with detection limit for analysis
                df = df.with_columns(pl.col(col).fill_null(limit).alias(col))

        return df

    def _add_derived_quantities(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived astrophysical quantities."""

        # Effective temperature from J-Ks color (dwarf stars)
        if "j_k" in df.columns:
            # Bessell & Brett (1988) relation
            df = df.with_columns(
                (3.981 - 0.324 * pl.col("j_k")).alias("log_teff_dwarf")
            )

            df = df.with_columns((10 ** (pl.col("log_teff_dwarf"))).alias("teff_dwarf"))

        # Bolometric correction for K band (approximate)
        if "k_m" in df.columns:
            # BC_K relation
            df = df.with_columns(
                (pl.col("k_m") + 1.9).alias("absolute_bol_mag_est")  # Rough estimate
            )

        # Reddening estimate (simplified, assumes intrinsic colors)
        if all(col in df.columns for col in ["j_h", "h_k"]):
            # Rough reddening estimate using color excess
            # E(J-H) / E(H-K) ≈ 1.7 for standard extinction law
            df = df.with_columns(
                ((pl.col("j_h") - 0.0) * 0.3).alias(
                    "reddening_estimate"
                )  # Very approximate
            )

        return df
