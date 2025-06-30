"""
WISE (Wide-field Infrared Survey Explorer) data preprocessing
with proper astronomical handling.
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


class WISEPreprocessor(BaseSurveyProcessor):
    """
    processor for WISE survey data with proper astronomical handling.

    Features:
    - Proper infrared photometry handling with Vega system
    - Temperature estimation from colors
    - AGN/quasar identification using color criteria
    - Proper motion analysis if available
    - Quality filtering based on WISE recommendations
    - Cross-matching capabilities with other surveys
    """

    def __init__(self, survey_name: str = "wise"):
        super().__init__(survey_name)

        # WISE-specific configuration
        self.wise_bands = ["W1", "W2", "W3", "W4"]
        self.wise_wavelengths = {  # Central wavelengths in microns
            "W1": 3.4,
            "W2": 4.6,
            "W3": 12.0,
            "W4": 22.0,
        }

        # Quality thresholds
        self.min_snr = 2.0  # Reduced from 5.0
        self.max_chi2 = 10.0  # Increased from 3.0

        # AGN color selection criteria (Stern et al. 2012)
        self.agn_w1_w2_min = 0.8  # W1-W2 > 0.8 for AGN candidates

    def get_coordinate_columns(self) -> List[str]:
        """Get WISE coordinate column names."""
        return ["ra", "dec", "w1_mag"]  # Use W1 magnitude for distance estimation

    def get_magnitude_columns(self) -> List[str]:
        """Get WISE magnitude column names."""
        # WISE AllWISE catalog column naming
        return [
            "w1mpro",  # W1 profile-fit magnitude
            "w2mpro",  # W2 profile-fit magnitude
            "w3mpro",  # W3 profile-fit magnitude
            "w4mpro",  # W4 profile-fit magnitude
        ]

    def apply_survey_specific_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply WISE-specific quality filters."""
        initial_count = len(df)

        # 1. Photometric quality flags (ph_qual) - relaxed
        if "ph_qual" in df.columns:
            # Keep sources with quality A, B, C, D, or E in W1 and W2 (more relaxed)
            df = df.filter(
                pl.col("ph_qual").str.slice(0, 1).is_in(["A", "B", "C", "D", "E"])
                & pl.col("ph_qual").str.slice(1, 1).is_in(["A", "B", "C", "D", "E"])
            )

        # 2. Contamination flags (cc_flags) - relaxed
        if "cc_flags" in df.columns:
            # Allow some contamination (less strict)
            df = df.filter(~pl.col("cc_flags").str.contains("1111"))

        # 3. Extended source flag (ext_flg) - allow extended sources
        # Commented out to allow both point and extended sources
        # if "ext_flg" in df.columns:
        #     df = df.filter(pl.col("ext_flg") == 0)

        # 4. Signal-to-noise ratio thresholds - relaxed
        snr_cols = ["w1snr", "w2snr", "w3snr", "w4snr"]
        for col in snr_cols:
            if col in df.columns:
                df = df.filter(pl.col(col).is_null() | (pl.col(col) >= self.min_snr))

        # 5. Chi-squared thresholds for PSF fitting - relaxed
        chi2_cols = ["w1rchi2", "w2rchi2", "w3rchi2", "w4rchi2"]
        for col in chi2_cols:
            if col in df.columns:
                df = df.filter(pl.col(col).is_null() | (pl.col(col) <= self.max_chi2))

        # 6. Remove sources with invalid magnitudes - relaxed
        mag_cols = self.get_magnitude_columns()
        for col in mag_cols:
            if col in df.columns:
                # More relaxed magnitude limits
                df = df.filter(
                    pl.col(col).is_not_null()
                    & pl.col(col).is_finite()
                    & (pl.col(col) > -999)  # Allow more negative values
                    & (pl.col(col) < 30)  # More relaxed upper limit
                )

        final_count = len(df)
        if final_count < initial_count:
            logger.info(
                f"🔍 WISE quality filters: {initial_count} → {final_count} sources "
                f"({final_count / initial_count * 100:.1f}% retained)"
            )

        return df

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates from WISE data with simple distance estimation."""

        # Check if we have RA/Dec coordinates
        if all(col in df.columns for col in ["ra", "dec"]):
            return self._simple_ra_dec_to_xyz(df)

        raise ValueError("Insufficient coordinate information in WISE data")

    def _simple_ra_dec_to_xyz(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert RA/Dec to 3D coordinates with simple distance estimation."""
        import numpy as np

        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()

        # Use simple distance estimation based on W1 magnitude
        w1_mag = None
        for col in ["w1mpro", "w1_mag", "W1", "w1"]:
            if col in df.columns:
                w1_mag = df[col].to_numpy()
                break

        if w1_mag is None:
            # Fallback to fixed distance
            distances = np.full(len(df), 10000.0)  # 10 kpc default
        else:
            # Simple distance estimation
            abs_mag = 2.5  # Default absolute magnitude
            distance_modulus = w1_mag - abs_mag
            distances = 10 ** ((distance_modulus + 5) / 5)  # Distance in pc
            distances = np.clip(distances, 100, 1e6)  # 100 pc to 1 Mpc

        # Create coordinates
        coords = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            distance=distances * u.pc,
            frame="icrs",
        )

        # Convert to Cartesian coordinates
        x = coords.cartesian.x.to_value("pc")
        y = coords.cartesian.y.to_value("pc")
        z = coords.cartesian.z.to_value("pc")

        coords_3d = np.stack([x, y, z], axis=1)
        return torch.tensor(coords_3d, dtype=torch.float32)

    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract comprehensive feature vector from WISE data."""
        features = []
        feature_names = []

        # 1. Infrared photometry features
        ir_features, ir_names = self._extract_infrared_features(df)
        if ir_features is not None:
            features.append(ir_features)
            feature_names.extend(ir_names)

        # 2. Color indices and temperature indicators
        color_features, color_names = self._extract_color_features(df)
        if color_features is not None:
            features.append(color_features)
            feature_names.extend(color_names)

        # 3. Variability and quality features
        quality_features, quality_names = self._extract_quality_features(df)
        if quality_features is not None:
            features.append(quality_features)
            feature_names.extend(quality_names)

        # 4. AGN/stellar classification features
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

    def _extract_infrared_features(self, df: pl.DataFrame) -> tuple:
        """Extract infrared photometry features."""
        features = []
        names = []

        # WISE magnitudes (Vega system)
        mag_cols = self.get_magnitude_columns()
        for i, col in enumerate(mag_cols):
            if col in df.columns:
                mags = df[col].to_numpy()
                mags = np.nan_to_num(mags, nan=99.0)  # Use 99 for non-detections
                features.append(mags.reshape(-1, 1))
                names.append(f"W{i + 1}_mag")

        # Signal-to-noise ratios
        snr_cols = ["w1snr", "w2snr", "w3snr", "w4snr"]
        for i, col in enumerate(snr_cols):
            if col in df.columns:
                snr = df[col].to_numpy()
                snr = np.nan_to_num(snr, nan=0.0)
                features.append(np.log10(snr + 1).reshape(-1, 1))  # Log scale
                names.append(f"log_W{i + 1}_snr")

        # Flux densities (for SED analysis)
        flux_cols = ["w1flux", "w2flux", "w3flux", "w4flux"]
        for i, col in enumerate(flux_cols):
            if col in df.columns:
                flux = df[col].to_numpy()
                flux = np.nan_to_num(flux, nan=0.0)
                features.append(np.log10(flux + 1e-6).reshape(-1, 1))  # Log flux
                names.append(f"log_W{i + 1}_flux")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_color_features(self, df: pl.DataFrame) -> tuple:
        """Extract infrared color indices and temperature indicators."""
        features = []
        names = []

        mag_cols = self.get_magnitude_columns()
        mags = {}

        # Extract available magnitudes
        for i, col in enumerate(mag_cols):
            if col in df.columns:
                mags[f"W{i + 1}"] = df[col].to_numpy()

        # Standard WISE colors
        color_pairs = [
            ("W1", "W2"),
            ("W2", "W3"),
            ("W3", "W4"),
            ("W1", "W3"),
            ("W1", "W4"),
        ]

        for band1, band2 in color_pairs:
            if band1 in mags and band2 in mags:
                color = mags[band1] - mags[band2]
                color = np.nan_to_num(color, nan=0.0)
                features.append(color.reshape(-1, 1))
                names.append(f"{band1}_{band2}_color")

        # Infrared excess indicators
        if "W1" in mags and "W3" in mags:
            # W1-W3 color is sensitive to dust
            w1_w3 = mags["W1"] - mags["W3"]
            features.append(w1_w3.reshape(-1, 1))
            names.append("infrared_excess")

        # Temperature indicator from W1-W2
        if "W1" in mags and "W2" in mags:
            w1_w2 = mags["W1"] - mags["W2"]
            # Convert to approximate temperature (empirical relation)
            temp_indicator = 3000 / (w1_w2 + 0.5)  # Rough temperature scale
            features.append(temp_indicator.reshape(-1, 1))
            names.append("temperature_indicator")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_quality_features(self, df: pl.DataFrame) -> tuple:
        """Extract quality and variability features."""
        features = []
        names = []

        # Chi-squared values for PSF fitting quality
        chi2_cols = ["w1rchi2", "w2rchi2", "w3rchi2", "w4rchi2"]
        for i, col in enumerate(chi2_cols):
            if col in df.columns:
                chi2 = df[col].to_numpy()
                chi2 = np.nan_to_num(chi2, nan=1.0)
                features.append(np.log10(chi2 + 0.1).reshape(-1, 1))
                names.append(f"log_W{i + 1}_chi2")

        # Variability indicators
        var_cols = ["w1sat", "w2sat", "w3sat", "w4sat"]  # Saturation flags
        for i, col in enumerate(var_cols):
            if col in df.columns:
                sat_flag = df[col].to_numpy().astype(float)
                features.append(sat_flag.reshape(-1, 1))
                names.append(f"W{i + 1}_saturated")

        # Moon masking flags (quality indicator)
        if "moon_lev" in df.columns:
            moon_lev = df["moon_lev"].to_numpy()
            features.append(moon_lev.reshape(-1, 1))
            names.append("moon_contamination")

        # Number of exposures (depth indicator)
        if "n" in df.columns:
            n_exp = df["n"].to_numpy()
            features.append(np.log10(n_exp + 1).reshape(-1, 1))
            names.append("log_n_exposures")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_classification_features(self, df: pl.DataFrame) -> tuple:
        """Extract features for AGN/stellar classification."""
        features = []
        names = []

        # AGN color criterion (Stern et al. 2012)
        if "w1mpro" in df.columns and "w2mpro" in df.columns:
            w1 = df["w1mpro"].to_numpy()
            w2 = df["w2mpro"].to_numpy()
            w1_w2 = w1 - w2

            # AGN probability based on color
            agn_prob = np.where(w1_w2 > self.agn_w1_w2_min, 1.0, 0.0)
            features.append(agn_prob.reshape(-1, 1))
            names.append("agn_candidate")

            # Quasar probability (stricter criterion)
            qso_prob = np.where(w1_w2 > 1.0, 1.0, 0.0)
            features.append(qso_prob.reshape(-1, 1))
            names.append("qso_candidate")

        # YSO (Young Stellar Object) indicators
        if all(col in df.columns for col in ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]):
            w1 = df["w1mpro"].to_numpy()
            w2 = df["w2mpro"].to_numpy()
            w3 = df["w3mpro"].to_numpy()
            df["w4mpro"].to_numpy()

            # YSO color criteria (Koenig et al. 2012)
            w1_w2 = w1 - w2
            w2_w3 = w2 - w3

            # Class I/II YSO indicators
            yso_prob = np.where((w1_w2 > 0.25) & (w2_w3 > 1.0), 1.0, 0.0)
            features.append(yso_prob.reshape(-1, 1))
            names.append("yso_candidate")

        # Proper motion indicators (if available)
        pm_cols = ["pmra", "pmdec", "pmw1", "pmw2"]
        if any(col in df.columns for col in pm_cols):
            # Use WISE proper motions if available
            if "pmw1" in df.columns and "pmw2" in df.columns:
                pm_w1 = df["pmw1"].to_numpy()
                pm_w2 = df["pmw2"].to_numpy()
                pm_total = np.sqrt(pm_w1**2 + pm_w2**2)
                pm_total = np.nan_to_num(pm_total, nan=0.0)
                features.append(np.log10(pm_total + 1).reshape(-1, 1))
                names.append("log_proper_motion")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply WISE-specific preprocessing."""

        # 1. Standardize column names
        df = self._standardize_column_names(df)

        # 2. Calculate infrared colors
        df = self._add_infrared_colors(df)

        # 3. Add source classification flags
        df = self._add_classification_flags(df)

        # 4. Handle non-detections and upper limits
        df = self._handle_nondetections(df)

        # 5. Add quality metrics
        df = self._add_quality_metrics(df)

        return df

    def _standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize WISE column names across different data releases."""
        # Map various WISE column naming conventions to standard names
        column_mapping = {
            # Magnitudes
            "w1mag": "w1mpro",
            "w2mag": "w2mpro",
            "w3mag": "w3mpro",
            "w4mag": "w4mpro",
            "W1": "w1mpro",
            "W2": "w2mpro",
            "W3": "w3mpro",
            "W4": "w4mpro",
            # Errors
            "w1mag_err": "w1sigmpro",
            "w2mag_err": "w2sigmpro",
            "w3mag_err": "w3sigmpro",
            "w4mag_err": "w4sigmpro",
            # Coordinates
            "RA": "ra",
            "DEC": "dec",
            "GLON": "glat",
            "GLAT": "glon",
        }

        # Apply mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        return df

    def _add_infrared_colors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add infrared color indices."""

        # Standard WISE colors
        color_definitions = [
            ("w1_w2", "w1mpro", "w2mpro"),
            ("w2_w3", "w2mpro", "w3mpro"),
            ("w3_w4", "w3mpro", "w4mpro"),
            ("w1_w3", "w1mpro", "w3mpro"),
            ("w1_w4", "w1mpro", "w4mpro"),
            ("w2_w4", "w2mpro", "w4mpro"),
        ]

        for color_name, mag1, mag2 in color_definitions:
            if (
                mag1 in df.columns
                and mag2 in df.columns
                and color_name not in df.columns
            ):
                df = df.with_columns((pl.col(mag1) - pl.col(mag2)).alias(color_name))

        return df

    def _add_classification_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add source classification flags based on WISE colors."""

        # AGN classification (Stern et al. 2012)
        if "w1_w2" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("w1_w2") > self.agn_w1_w2_min).alias("agn_candidate"),
                    (pl.col("w1_w2") > 1.0).alias("qso_candidate"),
                ]
            )

        # YSO classification (Koenig et al. 2012)
        if all(col in df.columns for col in ["w1_w2", "w2_w3"]):
            df = df.with_columns(
                ((pl.col("w1_w2") > 0.25) & (pl.col("w2_w3") > 1.0)).alias(
                    "yso_candidate"
                )
            )

        # High proper motion objects
        if "pmw1" in df.columns and "pmw2" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("pmw1") ** 2 + pl.col("pmw2") ** 2)
                    .sqrt()
                    .alias("pm_total"),
                    ((pl.col("pmw1") ** 2 + pl.col("pmw2") ** 2).sqrt() > 40.0).alias(
                        "high_pm_object"
                    ),
                ]
            )

        return df

    def _handle_nondetections(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle non-detections and upper limits in WISE data."""

        # WISE uses null values for non-detections
        # Convert to magnitude limits where appropriate
        mag_cols = ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]

        # Typical WISE detection limits (5-sigma)
        detection_limits = {
            "w1mpro": 17.1,  # W1 5-sigma limit
            "w2mpro": 15.7,  # W2 5-sigma limit
            "w3mpro": 11.5,  # W3 5-sigma limit
            "w4mpro": 8.0,  # W4 5-sigma limit
        }

        for col in mag_cols:
            if col in df.columns:
                limit = detection_limits.get(col, 99.0)

                # Add upper limit flag
                df = df.with_columns(pl.col(col).is_null().alias(f"{col}_upper_limit"))

                # Fill nulls with detection limit for analysis
                df = df.with_columns(pl.col(col).fill_null(limit).alias(col))

        return df

    def _add_quality_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add quality assessment metrics."""

        # Overall quality score based on multiple factors
        quality_components = []

        # Photometric quality from ph_qual
        if "ph_qual" in df.columns:
            # Convert quality letters to scores
            quality_map = {
                "A": 1.0,
                "B": 0.8,
                "C": 0.6,
                "D": 0.4,
                "F": 0.2,
                "N": 0.0,
                "U": 0.0,
            }

            # Average quality across W1 and W2
            w1_qual = (
                pl.col("ph_qual").str.slice(0, 1).replace(quality_map, default=0.0)
            )
            w2_qual = (
                pl.col("ph_qual").str.slice(1, 1).replace(quality_map, default=0.0)
            )

            df = df.with_columns(
                ((w1_qual + w2_qual) / 2.0).alias("photometric_quality")
            )
            quality_components.append("photometric_quality")

        # SNR-based quality
        snr_cols = ["w1snr", "w2snr"]
        for col in snr_cols:
            if col in df.columns:
                # Normalize SNR to 0-1 scale (clamp at SNR=50)
                df = df.with_columns(
                    (pl.col(col).clip(0, 50) / 50.0).alias(f"{col}_norm")
                )
                quality_components.append(f"{col}_norm")

        # Combined quality score
        if quality_components:
            quality_expr = pl.lit(0.0)
            for comp in quality_components:
                quality_expr = quality_expr + pl.col(comp).fill_null(0.0)

            df = df.with_columns(
                (quality_expr / len(quality_components)).alias("overall_quality")
            )

        return df
