"""RR Lyrae variable star preprocessor implementation.

Handles RR Lyrae variable star catalog preprocessing.
"""

import logging
from typing import Any, Dict, Optional

import polars as pl

from astro_lab.data.transforms.astronomical import spherical_to_cartesian

from .astro import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
)

logger = logging.getLogger(__name__)


class RRLyraePreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Preprocessor for RR Lyrae variable star data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RR Lyrae preprocessor."""
        default_config = {
            "magnitude_limit": 18.0,  # Magnitude limit
            "period_limit": [0.2, 1.0],  # Period range in days for RR Lyrae
            "amplitude_limit": 0.1,  # Minimum amplitude for variability
            "distance_limit": 100.0,  # Maximum distance in kpc
            "require_period": True,  # Period required for RR Lyrae
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        self.required_columns = ["id"]  # RR Lyrae star ID

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "rrlyrae"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "variable_star"

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply RR Lyrae-specific quality filters."""
        initial_count = len(df)

        # 1. Magnitude filter - from SURVEY_CONFIGS: umag, gmag, rmag, imag, zmag, Vmag
        mag_cols = ["Vmag", "gmag", "rmag", "magnitude", "mag", "V", "vmag", "mean_mag"]
        mag_col = None
        for col in mag_cols:
            if col in df.columns:
                mag_col = col
                break

        if mag_col:
            df = df.filter(
                pl.col(mag_col).is_not_null()
                & pl.col(mag_col).is_finite()
                & (pl.col(mag_col) > 8)
                & (pl.col(mag_col) < self.config["magnitude_limit"])
            )

        # 2. Period filter (critical for RR Lyrae) - from SURVEY_CONFIGS: Per
        if self.config["require_period"]:
            period_cols = ["Per", "period", "period_days", "P", "pulsation_period"]
            period_col = None
            for col in period_cols:
                if col in df.columns:
                    period_col = col
                    break

            if period_col:
                min_period, max_period = self.config["period_limit"]
                df = df.filter(
                    (pl.col(period_col) >= min_period)
                    & (pl.col(period_col) <= max_period)
                )
            else:
                logger.warning(
                    "No period column found, but period required for RR Lyrae"
                )

        # 3. Amplitude filter (RR Lyrae are high-amplitude variables) - from SURVEY_CONFIGS: uAmp, gAmp, rAmp, iAmp
        amp_cols = [
            "uAmp",
            "gAmp",
            "rAmp",
            "iAmp",
            "amplitude",
            "amp",
            "delta_mag",
            "variability_amplitude",
        ]
        amp_col = None
        for col in amp_cols:
            if col in df.columns:
                amp_col = col
                break

        if amp_col:
            df = df.filter(pl.col(amp_col) >= self.config["amplitude_limit"])

        # 4. Distance filter (if available) - from SURVEY_CONFIGS: Dist
        dist_cols = ["Dist", "distance", "dist", "distance_kpc"]
        dist_col = None
        for col in dist_cols:
            if col in df.columns:
                dist_col = col
                break

        if dist_col:
            df = df.filter(pl.col(dist_col) <= self.config["distance_limit"])

        final_count = len(df)
        logger.info(
            f"Filtered {initial_count - final_count} RR Lyrae stars ({((initial_count - final_count) / initial_count * 100):.1f}%)"
        )

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform RR Lyrae data."""
        # Standardize column names
        df = self._standardize_column_names(df)

        # Add RR Lyrae classification features
        df = self._add_rrlyrae_features(df)

        # Convert to cartesian coordinates - from SURVEY_CONFIGS: RAJ2000, DEJ2000, Dist
        ra_cols = ["RAJ2000", "ra", "RA", "raj2000", "ra_deg"]
        dec_cols = ["DEJ2000", "dec", "DEC", "dej2000", "dec_deg"]

        ra_col = None
        dec_col = None

        for col in ra_cols:
            if col in df.columns:
                ra_col = col
                break

        for col in dec_cols:
            if col in df.columns:
                dec_col = col
                break

        if ra_col and dec_col:
            # Distance estimation - from SURVEY_CONFIGS: Dist
            dist_cols = ["Dist", "distance", "dist", "distance_kpc"]
            dist_col = None
            for col in dist_cols:
                if col in df.columns:
                    dist_col = col
                    break

            if dist_col:
                # Convert to parsecs if in kpc
                df = df.with_columns([(pl.col(dist_col) * 1000).alias("distance_pc")])
            else:
                # Distance estimation using period-luminosity relation
                if "period" in df.columns:
                    # RR Lyrae period-luminosity relation (rough approximation)
                    df = df.with_columns(
                        [
                            (-1.43 - 2.81 * pl.col("period").log()).alias("absolute_V"),
                        ]
                    )

                    # Distance modulus - use Vmag from SURVEY_CONFIGS
                    mag_cols = ["Vmag", "magnitude", "mag", "V", "vmag", "mean_mag"]
                    mag_col = None
                    for col in mag_cols:
                        if col in df.columns:
                            mag_col = col
                            break

                    if mag_col:
                        df = df.with_columns(
                            [
                                pl.pow(
                                    10,
                                    ((pl.col(mag_col) - pl.col("absolute_V") + 5) / 5),
                                )
                                .clip(100, 100000)
                                .alias("distance_pc")
                            ]
                        )
                    else:
                        # Default distance for RR Lyrae (typically halo stars)
                        df = df.with_columns(
                            [
                                pl.lit(10000.0).alias("distance_pc")  # 10 kpc default
                            ]
                        )
                else:
                    # Default distance
                    df = df.with_columns(
                        [
                            pl.lit(10000.0).alias("distance_pc")  # 10 kpc default
                        ]
                    )

            x, y, z = spherical_to_cartesian(df[ra_col], df[dec_col], df["distance_pc"])
            df = df.with_columns(
                [
                    pl.Series("x", x),
                    pl.Series("y", y),
                    pl.Series("z", z),
                ]
            )

        return df

    def _standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize RR Lyrae column names based on SURVEY_CONFIGS."""
        column_mapping = {
            # Coordinates - from SURVEY_CONFIGS
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "RA": "ra",
            "DEC": "dec",
            "raj2000": "ra",
            "dej2000": "dec",
            "ra_deg": "ra",
            "dec_deg": "dec",
            # Magnitudes - from SURVEY_CONFIGS
            "umag": "u_mag",
            "gmag": "g_mag",
            "rmag": "r_mag",
            "imag": "i_mag",
            "zmag": "z_mag",
            "Vmag": "v_mag",
            "V": "v_mag",
            "vmag": "v_mag",
            "mean_mag": "magnitude",
            "mag": "magnitude",
            # Period - from SURVEY_CONFIGS
            "Per": "period",
            "P": "period",
            "period_days": "period",
            "pulsation_period": "period",
            # Amplitudes - from SURVEY_CONFIGS
            "uAmp": "u_amplitude",
            "gAmp": "g_amplitude",
            "rAmp": "r_amplitude",
            "iAmp": "i_amplitude",
            "amp": "amplitude",
            "AmpV": "amplitude",
            "delta_mag": "amplitude",
            "variability_amplitude": "amplitude",
            # Times of maximum - from SURVEY_CONFIGS
            "T0_u": "t0_u",
            "T0_g": "t0_g",
            "T0_r": "t0_r",
            "T0_i": "t0_i",
            "T0_z": "t0_z",
            # Distance - from SURVEY_CONFIGS
            "Dist": "distance",
            "distance_kpc": "distance",
            "dist": "distance",
            # Other RR Lyrae specific - from SURVEY_CONFIGS
            "__SIG2010_": "sig2010",
            "Type": "rrlyrae_type",
            "Ar": "extinction_r",
            # ID
            "name": "id",
            "star_id": "id",
            "rrlyrae_id": "id",
            "object_id": "id",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        return df

    def _add_rrlyrae_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add RR Lyrae-specific features."""
        # RR Lyrae type classification - from SURVEY_CONFIGS: Type
        if "rrlyrae_type" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("rrlyrae_type").str.contains("ab|AB")).alias("rrab_type"),
                    (pl.col("rrlyrae_type").str.contains("c|C")).alias("rrc_type"),
                    (pl.col("rrlyrae_type").str.contains("d|D")).alias("rrd_type"),
                ]
            )
        elif "period" in df.columns:
            # Classify based on period (rough approximation)
            df = df.with_columns(
                [
                    (pl.col("period") > 0.4).alias(
                        "rrab_type"
                    ),  # RRab typically > 0.4 days
                    (pl.col("period") <= 0.4).alias(
                        "rrc_type"
                    ),  # RRc typically < 0.4 days
                    pl.lit(False).alias("rrd_type"),  # RRd is rare
                ]
            )

        # Period-based features
        if "period" in df.columns:
            df = df.with_columns(
                [
                    # Log period (useful for analysis)
                    pl.col("period").log().alias("log_period"),
                    # Frequency
                    (1.0 / pl.col("period")).alias("frequency"),
                    # Period ratio indicators
                    (pl.col("period") / 0.5).alias("period_ratio"),
                ]
            )

        # Amplitude-based features (using band-specific amplitudes from SURVEY_CONFIGS)
        amp_cols = [
            "u_amplitude",
            "g_amplitude",
            "r_amplitude",
            "i_amplitude",
            "amplitude",
        ]
        amp_col = None
        for col in amp_cols:
            if col in df.columns:
                amp_col = col
                break

        if amp_col:
            df = df.with_columns(
                [
                    # Log amplitude
                    pl.col(amp_col).log().alias("log_amplitude"),
                    # High amplitude indicator
                    (pl.col(amp_col) > 1.0).alias("high_amplitude"),
                    # Blazhko effect indicator (amplitude variation)
                    (pl.col(amp_col) < 0.3).alias("possible_blazhko"),
                ]
            )

        # Bailey diagram features (period vs amplitude)
        if "period" in df.columns and amp_col:
            df = df.with_columns(
                [
                    # Bailey diagram position
                    (pl.col("period") * pl.col(amp_col)).alias("bailey_product"),
                    (pl.col("period") / pl.col(amp_col)).alias("bailey_ratio"),
                ]
            )

        # Multi-band analysis if available (from SURVEY_CONFIGS magnitudes)
        if all(col in df.columns for col in ["u_mag", "g_mag", "r_mag"]):
            df = df.with_columns(
                [
                    # Colors
                    (pl.col("u_mag") - pl.col("g_mag")).alias("u_g"),
                    (pl.col("g_mag") - pl.col("r_mag")).alias("g_r"),
                    # Metal-rich vs metal-poor indicator from color
                    ((pl.col("g_mag") - pl.col("r_mag")) > 0.3).alias(
                        "metal_rich_candidate"
                    ),
                    ((pl.col("g_mag") - pl.col("r_mag")) < 0.2).alias(
                        "metal_poor_candidate"
                    ),
                ]
            )

        # Phase information if available (from SURVEY_CONFIGS: T0_*)
        phase_cols = ["t0_u", "t0_g", "t0_r", "t0_i", "t0_z"]
        available_phases = [col for col in phase_cols if col in df.columns]
        if available_phases and "period" in df.columns:
            # Phase difference between bands
            if len(available_phases) >= 2:
                df = df.with_columns(
                    [
                        (
                            (pl.col(available_phases[1]) - pl.col(available_phases[0]))
                            / pl.col("period")
                        ).alias("phase_diff")
                    ]
                )

        # Galactic population indicators
        if "distance" in df.columns:
            df = df.with_columns(
                [
                    # Halo population (distant stars)
                    (pl.col("distance") > 8.0).alias("halo_population"),
                    # Disk population (nearby stars)
                    (pl.col("distance") < 3.0).alias("disk_population"),
                ]
            )

        # Extinction correction if available (from SURVEY_CONFIGS: Ar)
        if "extinction_r" in df.columns and "r_mag" in df.columns:
            df = df.with_columns(
                [(pl.col("r_mag") - pl.col("extinction_r")).alias("r_mag_corrected")]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract RR Lyrae features."""
        feature_columns = []

        # ID column
        id_cols = ["id", "name", "star_id", "rrlyrae_id", "object_id"]
        id_col = None
        for col in id_cols:
            if col in df.columns:
                id_col = col
                break

        # Position
        if all(col in df.columns for col in ["x", "y", "z"]):
            feature_columns.extend(["x", "y", "z"])
        elif all(col in df.columns for col in ["ra", "dec"]):
            feature_columns.extend(["ra", "dec"])

        # Basic properties
        basic_cols = ["v_mag", "magnitude", "period", "distance"]
        for col in basic_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Multi-band magnitudes (from SURVEY_CONFIGS)
        mag_cols = ["u_mag", "g_mag", "r_mag", "i_mag", "z_mag"]
        for col in mag_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Amplitudes (from SURVEY_CONFIGS)
        amp_cols = [
            "u_amplitude",
            "g_amplitude",
            "r_amplitude",
            "i_amplitude",
            "amplitude",
        ]
        for col in amp_cols:
            if col in df.columns:
                feature_columns.append(col)

        # RR Lyrae type classification
        type_cols = ["rrab_type", "rrc_type", "rrd_type"]
        for col in type_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Derived features
        derived_cols = [
            "log_period",
            "frequency",
            "log_amplitude",
            "bailey_product",
            "bailey_ratio",
            "phase_diff",
        ]
        for col in derived_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Population indicators
        pop_cols = [
            "metal_rich_candidate",
            "metal_poor_candidate",
            "halo_population",
            "disk_population",
        ]
        for col in pop_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Variability indicators
        var_cols = ["high_amplitude", "possible_blazhko"]
        for col in var_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Color information
        color_cols = ["u_g", "g_r"]
        for col in color_cols:
            if col in df.columns:
                feature_columns.append(col)

        # RR Lyrae specific from SURVEY_CONFIGS
        specific_cols = ["sig2010", "extinction_r", "r_mag_corrected"]
        for col in specific_cols:
            if col in df.columns:
                feature_columns.append(col)

        # If no standard features, use all numeric columns
        if not feature_columns:
            logger.warning("No standard RR Lyrae features found, using numeric columns")
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]
            feature_columns = numeric_cols[:10]  # Limit to first 10

        # Keep available features
        available_features = [col for col in feature_columns if col in df.columns]

        # Normalize features
        if available_features:
            # Separate numeric and boolean features
            numeric_features = []
            boolean_features = []

            for col in available_features:
                if df[col].dtype == pl.Boolean:
                    boolean_features.append(col)
                else:
                    numeric_features.append(col)

            # Normalize numeric features
            if numeric_features:
                df = self.normalize_columns(df, numeric_features, method="standard")

                # Remove _norm suffix
                for col in numeric_features:
                    if f"{col}_norm" in df.columns:
                        df = df.with_columns([pl.col(f"{col}_norm").alias(col)])
                        df = df.drop(f"{col}_norm")

            # Convert boolean features to float
            for col in boolean_features:
                df = df.with_columns([pl.col(col).cast(pl.Float32).alias(col)])

        # Select final columns
        keep_columns = []
        if id_col:
            keep_columns.append(id_col)
        keep_columns.extend(available_features)

        df = df.select([col for col in keep_columns if col in df.columns])

        logger.info(
            f"Extracted {len(available_features)} RR Lyrae features: {available_features}"
        )

        return df
