"""
NASA Exoplanet Archive Preprocessor
===================================

preprocessor for NASA Exoplanet Archive data with stellar host analysis.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord

from .base import BaseSurveyProcessor

logger = logging.getLogger(__name__)


class ExoplanetPreprocessor(BaseSurveyProcessor):
    """
    processor for NASA Exoplanet Archive data with stellar host clustering.

    Features:
    - Multi-planetary system analysis
    - Stellar host star clustering for cosmic web studies
    - Habitable zone calculations
    - Planet-star interaction features
    - Cross-matching with stellar catalogs (Gaia, 2MASS, etc.)
    - Discovery method bias corrections
    """

    def __init__(
        self, survey_name: str = "exoplanet", data_config: Optional[Dict] = None
    ):
        super().__init__(survey_name, data_config)

        # Exoplanet-specific configuration
        self.min_planet_radius = 0.1  # Earth radii
        self.max_planet_radius = 30.0  # Earth radii (avoid brown dwarfs)
        self.min_orbital_period = 0.1  # days
        self.max_orbital_period = 10000.0  # days

        # Discovery method weighting for bias correction
        self.method_weights = {
            "Transit": 1.0,
            "Radial Velocity": 1.2,  # Slightly weight up due to M sin(i) degeneracy
            "Direct Imaging": 0.8,  # Weight down due to bias toward massive planets
            "Microlensing": 1.1,
            "Astrometry": 1.0,
            "Timing": 1.0,
            "Other": 0.9,
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for exoplanet data."""
        config = super()._get_default_config()
        config.update(
            {
                "coordinate_system": "icrs",
                "distance_unit": "pc",
                "filter_system": "Vega",  # Most stellar photometry in Vega
                "feature_cols": [
                    # Planet properties
                    "pl_orbper",
                    "pl_orbsmax",
                    "pl_orbeccen",
                    "pl_orbincl",
                    "pl_rade",
                    "pl_masse",
                    "pl_dens",
                    "pl_eqt",
                    # Stellar properties
                    "st_teff",
                    "st_rad",
                    "st_mass",
                    "st_met",
                    "st_age",
                    "st_dens",
                    "st_lum",
                    # Discovery metadata
                    "disc_year",
                ],
                "mag_cols": [
                    "sy_vmag",
                    "sy_jmag",
                    "sy_hmag",
                    "sy_kmag",  # System magnitudes
                    "st_optmag",  # Stellar optical magnitude
                ],
                "error_cols": [
                    "sy_vmagerr",
                    "sy_jmagerr",
                    "sy_hmagerr",
                    "sy_kmagerr",
                    "st_optmagerr",
                ],
            }
        )
        return config

    def get_coordinate_columns(self) -> List[str]:
        """Get exoplanet coordinate column names."""
        return ["ra", "dec", "sy_dist"]  # System coordinates and distance

    def apply_survey_specific_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply exoplanet-specific quality filters."""
        initial_count = len(df)

        # 1. Remove obviously invalid planets
        if "pl_rade" in df.columns:
            df = df.filter(
                (pl.col("pl_rade") >= self.min_planet_radius)
                & (pl.col("pl_rade") <= self.max_planet_radius)
            )

        if "pl_orbper" in df.columns:
            df = df.filter(
                (pl.col("pl_orbper") >= self.min_orbital_period)
                & (pl.col("pl_orbper") <= self.max_orbital_period)
            )

        # 2. Require minimum stellar parameters for host analysis
        required_stellar = ["st_teff", "st_rad", "st_mass"]
        for param in required_stellar:
            if param in df.columns:
                df = df.filter(pl.col(param).is_not_null() & (pl.col(param) > 0))

        # 3. Remove very uncertain distance measurements
        if "sy_disterr1" in df.columns and "sy_dist" in df.columns:
            # Relative error < 50%
            df = df.filter((pl.col("sy_disterr1") / pl.col("sy_dist")) < 0.5)

        # 4. Filter by discovery quality flags if available
        if "pl_tranflag" in df.columns:
            # Prefer transiting planets for higher confidence
            pass  # Could add preference weights here

        final_count = len(df)
        logger.info(f"Exoplanet filters: {initial_count} → {final_count} systems")
        return df

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates for stellar hosts."""
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()

        # Use system distance if available
        if "sy_dist" in df.columns:
            distance = df["sy_dist"].to_numpy()
            # Handle missing distances with stellar radius scaling
            valid_dist = ~np.isnan(distance) & (distance > 0)

            if not np.all(valid_dist):
                # Estimate distance from stellar properties
                if "st_rad" in df.columns and "sy_vmag" in df.columns:
                    st_rad = df["st_rad"].to_numpy()
                    v_mag = df["sy_vmag"].to_numpy()

                    # luminosity-based distance estimate
                    # M_V = V - 5*log10(d) + 5, assuming M_V from stellar radius
                    abs_mag_est = 4.8 - 2.5 * np.log10(st_rad**2)  # Rough scaling
                    distance_est = 10 ** ((v_mag - abs_mag_est + 5) / 5)

                    distance = np.where(valid_dist, distance, distance_est)
                else:
                    # Default distance for missing values
                    distance = np.where(valid_dist, distance, 100.0)  # 100 pc default
        else:
            # No distance information - use photometric estimate
            distance = np.full_like(ra, 100.0)  # Default 100 pc

        # Convert to Cartesian using astropy
        coords = SkyCoord(
            ra=ra * u.Unit("deg"),
            dec=dec * u.Unit("deg"),
            distance=distance * u.Unit("pc"),
            frame="icrs",
        )

        # Convert to Galactocentric for cosmic web analysis
        galcen = coords.galactocentric
        galcen_cart = galcen.cartesian

        # Extract coordinate values as numpy arrays
        x_coords = galcen_cart.x.to_value("pc")
        y_coords = galcen_cart.y.to_value("pc")
        z_coords = galcen_cart.z.to_value("pc")

        return torch.tensor(
            np.stack([x_coords, y_coords, z_coords], axis=1),
            dtype=torch.float32,
        )

    def get_magnitude_columns(self) -> List[str]:
        """Get magnitude columns for stellar hosts."""
        return [
            "sy_vmag",  # V magnitude
            "sy_jmag",  # J magnitude (2MASS)
            "sy_hmag",  # H magnitude (2MASS)
            "sy_kmag",  # K magnitude (2MASS)
        ]

    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract comprehensive exoplanet system features."""
        features = []
        feature_names = []

        # 1. Planet orbital features
        orbital_features, orbital_names = self._extract_orbital_features(df)
        if orbital_features is not None:
            features.append(orbital_features)
            feature_names.extend(orbital_names)

        # 2. Planet physical features
        physical_features, physical_names = self._extract_planet_features(df)
        if physical_features is not None:
            features.append(physical_features)
            feature_names.extend(physical_names)

        # 3. Stellar host features
        stellar_features, stellar_names = self._extract_stellar_features(df)
        if stellar_features is not None:
            features.append(stellar_features)
            feature_names.extend(stellar_names)

        # 4. System-level features
        system_features, system_names = self._extract_system_features(df)
        if system_features is not None:
            features.append(system_features)
            feature_names.extend(system_names)

        # 5. Discovery metadata
        discovery_features, discovery_names = self._extract_discovery_features(df)
        if discovery_features is not None:
            features.append(discovery_features)
            feature_names.extend(discovery_names)

        # 6. Habitability features
        hab_features, hab_names = self._extract_habitability_features(df)
        if hab_features is not None:
            features.append(hab_features)
            feature_names.extend(hab_names)

        if not features:
            # Minimal fallback
            logger.warning("No features extracted, using coordinate-based features")
            coords = self.extract_coordinates(df)
            distances = torch.norm(coords, dim=1, keepdim=True)
            return distances

        self.feature_names = feature_names
        combined_features = np.concatenate(features, axis=1)
        return torch.tensor(combined_features, dtype=torch.float32)

    def _extract_orbital_features(self, df: pl.DataFrame) -> tuple:
        """Extract orbital mechanics features."""
        features = []
        names = []

        # Orbital period (log scale due to wide range)
        if "pl_orbper" in df.columns:
            period = df["pl_orbper"].to_numpy()
            period = np.nan_to_num(period, nan=365.25)  # Default to 1 year
            features.append(np.log10(period).reshape(-1, 1))
            names.append("log_orbital_period")

        # Semi-major axis
        if "pl_orbsmax" in df.columns:
            sma = df["pl_orbsmax"].to_numpy()
            sma = np.nan_to_num(sma, nan=1.0)  # Default to 1 AU
            features.append(np.log10(sma).reshape(-1, 1))
            names.append("log_semi_major_axis")

        # Eccentricity
        if "pl_orbeccen" in df.columns:
            ecc = df["pl_orbeccen"].to_numpy()
            ecc = np.nan_to_num(ecc, nan=0.0)
            features.append(ecc.reshape(-1, 1))
            names.append("eccentricity")

        # Inclination
        if "pl_orbincl" in df.columns:
            inc = df["pl_orbincl"].to_numpy()
            inc = np.nan_to_num(inc, nan=90.0)  # Default to edge-on
            # Normalize to [0, 1]
            inc_norm = inc / 90.0
            features.append(inc_norm.reshape(-1, 1))
            names.append("inclination_norm")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_planet_features(self, df: pl.DataFrame) -> tuple:
        """Extract planet physical properties."""
        features = []
        names = []

        # Planet radius (log scale, Earth radii)
        if "pl_rade" in df.columns:
            radius = df["pl_rade"].to_numpy()
            radius = np.nan_to_num(radius, nan=1.0)  # Default to Earth size
            radius = np.clip(radius, 0.1, 30.0)  # Reasonable range
            features.append(np.log10(radius).reshape(-1, 1))
            names.append("log_planet_radius")

        # Planet mass (log scale, Earth masses)
        if "pl_masse" in df.columns:
            mass = df["pl_masse"].to_numpy()
            mass = np.nan_to_num(mass, nan=1.0)  # Default to Earth mass
            mass = np.clip(mass, 0.01, 1000.0)  # Reasonable range
            features.append(np.log10(mass).reshape(-1, 1))
            names.append("log_planet_mass")

        # Planet density
        if "pl_dens" in df.columns:
            density = df["pl_dens"].to_numpy()
            density = np.nan_to_num(density, nan=5.5)  # Earth density
            density = np.clip(density, 0.1, 20.0)
            features.append(np.log10(density).reshape(-1, 1))
            names.append("log_planet_density")
        elif "pl_rade" in df.columns and "pl_masse" in df.columns:
            # Calculate density if not given
            radius = df["pl_rade"].to_numpy()
            mass = df["pl_masse"].to_numpy()
            volume = (4 / 3) * np.pi * radius**3  # Relative to Earth
            density = mass / volume  # Relative to Earth
            density = np.nan_to_num(density, nan=1.0)
            density = np.clip(density, 0.1, 20.0)
            features.append(np.log10(density).reshape(-1, 1))
            names.append("log_planet_density_calc")

        # Equilibrium temperature
        if "pl_eqt" in df.columns:
            temp = df["pl_eqt"].to_numpy()
            temp = np.nan_to_num(temp, nan=288.0)  # Earth temperature
            temp = np.clip(temp, 50.0, 3000.0)
            # Normalize to Earth temperature
            temp_norm = temp / 288.0
            features.append(np.log10(temp_norm).reshape(-1, 1))
            names.append("log_eq_temp_norm")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_stellar_features(self, df: pl.DataFrame) -> tuple:
        """Extract stellar host properties."""
        features = []
        names = []

        # Effective temperature
        if "st_teff" in df.columns:
            teff = df["st_teff"].to_numpy()
            teff = np.nan_to_num(teff, nan=5772.0)  # Solar temperature
            # Normalize to solar temperature
            teff_norm = teff / 5772.0
            features.append(teff_norm.reshape(-1, 1))
            names.append("stellar_teff_solar")

        # Stellar radius
        if "st_rad" in df.columns:
            radius = df["st_rad"].to_numpy()
            radius = np.nan_to_num(radius, nan=1.0)  # Solar radii
            radius = np.clip(radius, 0.1, 20.0)
            features.append(np.log10(radius).reshape(-1, 1))
            names.append("log_stellar_radius")

        # Stellar mass
        if "st_mass" in df.columns:
            mass = df["st_mass"].to_numpy()
            mass = np.nan_to_num(mass, nan=1.0)  # Solar masses
            mass = np.clip(mass, 0.1, 10.0)
            features.append(np.log10(mass).reshape(-1, 1))
            names.append("log_stellar_mass")

        # Metallicity
        if "st_met" in df.columns:
            met = df["st_met"].to_numpy()
            met = np.nan_to_num(met, nan=0.0)  # Solar metallicity
            # Clip to reasonable range
            met = np.clip(met, -2.0, 1.0)
            features.append(met.reshape(-1, 1))
            names.append("stellar_metallicity")

        # Stellar age
        if "st_age" in df.columns:
            age = df["st_age"].to_numpy()
            age = np.nan_to_num(age, nan=4.6)  # Solar age in Gyr
            age = np.clip(age, 0.1, 15.0)
            features.append(np.log10(age).reshape(-1, 1))
            names.append("log_stellar_age")

        # Stellar luminosity
        if "st_lum" in df.columns:
            lum = df["st_lum"].to_numpy()
            lum = np.nan_to_num(lum, nan=1.0)  # Solar luminosities
            lum = np.clip(lum, 0.001, 1000.0)
            features.append(np.log10(lum).reshape(-1, 1))
            names.append("log_stellar_luminosity")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_system_features(self, df: pl.DataFrame) -> tuple:
        """Extract planetary system-level features."""
        features = []
        names = []

        # System distance
        if "sy_dist" in df.columns:
            dist = df["sy_dist"].to_numpy()
            dist = np.nan_to_num(dist, nan=100.0)  # Default 100 pc
            dist = np.clip(dist, 1.0, 10000.0)
            features.append(np.log10(dist).reshape(-1, 1))
            names.append("log_system_distance")

        # Number of planets in system (if available in hostnames)
        # This would require grouping by hostname - simplified for now

        # Galactic coordinates (computed from RA/Dec)
        if "ra" in df.columns and "dec" in df.columns:
            ra = df["ra"].to_numpy()
            dec = df["dec"].to_numpy()

            coords = SkyCoord(
                ra=ra * u.Unit("deg"), dec=dec * u.Unit("deg"), frame="icrs"
            )
            galactic = coords.galactic

            # Galactic latitude (indicator of thick disk/halo membership)
            b_values = galactic.lat.to_value("deg")
            b_abs = np.abs(b_values)
            features.append(b_abs.reshape(-1, 1))
            names.append("abs_galactic_latitude")

            # Distance from Galactic plane
            if "sy_dist" in df.columns:
                dist = df["sy_dist"].to_numpy()
                b_rad = np.radians(b_values)
                z_height = dist * np.sin(b_rad)
                z_height = np.abs(z_height)
                features.append(np.log10(z_height + 1).reshape(-1, 1))
                names.append("log_galactic_z_height")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_discovery_features(self, df: pl.DataFrame) -> tuple:
        """Extract discovery method and temporal features."""
        features = []
        names = []

        # Discovery year (normalized)
        if "disc_year" in df.columns:
            year = df["disc_year"].to_numpy()
            year = np.nan_to_num(year, nan=2010.0)
            # Normalize to range [0, 1] from 1995 to present
            year_norm = np.clip((year - 1995) / 30.0, 0.0, 1.0)
            features.append(year_norm.reshape(-1, 1))
            names.append("discovery_year_norm")

        # Discovery method one-hot encoding
        if "discoverymethod" in df.columns:
            methods = [
                "Transit",
                "Radial Velocity",
                "Direct Imaging",
                "Microlensing",
                "Astrometry",
            ]
            method_data = df["discoverymethod"].to_numpy()

            for method in methods:
                is_method = (method_data == method).astype(float)
                features.append(is_method.reshape(-1, 1))
                names.append(f"discovery_{method.lower().replace(' ', '_')}")

            # Add "other" category
            is_other = ~np.isin(method_data, methods)
            features.append(is_other.astype(float).reshape(-1, 1))
            names.append("discovery_other")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def _extract_habitability_features(self, df: pl.DataFrame) -> tuple:
        """Extract features related to planetary habitability."""
        features = []
        names = []

        # Check if we have necessary data for habitability calculations
        has_radius = "pl_rade" in df.columns
        has_temp = "pl_eqt" in df.columns
        has_orbital = "pl_orbsmax" in df.columns and "st_lum" in df.columns

        if has_temp:
            # Temperature-based habitability
            temp = df["pl_eqt"].to_numpy()
            temp = np.nan_to_num(temp, nan=288.0)

            # Earth similarity index (temperature component)
            temp_esi = 1.0 - np.abs(temp - 288.0) / (temp + 288.0)
            temp_esi = np.clip(temp_esi, 0.0, 1.0)
            features.append(temp_esi.reshape(-1, 1))
            names.append("temperature_esi")

            # In habitable zone flag (roughly 273-373 K)
            in_hz = ((temp >= 273.0) & (temp <= 373.0)).astype(float)
            features.append(in_hz.reshape(-1, 1))
            names.append("in_habitable_zone")

        if has_radius:
            # Size-based habitability
            radius = df["pl_rade"].to_numpy()
            radius = np.nan_to_num(radius, nan=1.0)

            # Earth-size similarity (radius component)
            radius_esi = 1.0 - np.abs(radius - 1.0) / (radius + 1.0)
            radius_esi = np.clip(radius_esi, 0.0, 1.0)
            features.append(radius_esi.reshape(-1, 1))
            names.append("radius_esi")

            # Potentially rocky flag (< 2 Earth radii)
            is_rocky = (radius < 2.0).astype(float)
            features.append(is_rocky.reshape(-1, 1))
            names.append("potentially_rocky")

        if has_orbital:
            # Orbital habitability features
            sma = df["pl_orbsmax"].to_numpy()
            sma = np.nan_to_num(sma, nan=1.0)

            st_lum = df["st_lum"].to_numpy()
            st_lum = np.nan_to_num(st_lum, nan=1.0)

            # Conservative habitable zone boundaries
            hz_inner = 0.95 * np.sqrt(st_lum)  # AU
            hz_outer = 1.37 * np.sqrt(st_lum)  # AU

            # Distance from HZ center
            hz_center = 0.5 * (hz_inner + hz_outer)
            hz_distance = np.abs(sma - hz_center) / (hz_outer - hz_inner)
            features.append(hz_distance.reshape(-1, 1))
            names.append("hz_distance_norm")

        if features:
            return np.concatenate(features, axis=1), names
        return None, []

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply exoplanet-specific preprocessing."""

        # 1. Handle multi-planetary systems
        df = self._process_multi_planet_systems(df)

        # 2. Add derived orbital properties
        df = self._add_derived_orbital_properties(df)

        # 3. Add stellar classification
        df = self._add_stellar_classification(df)

        # 4. Add habitability indicators
        df = self._add_habitability_indicators(df)

        # 5. Apply discovery bias corrections
        df = self._apply_discovery_bias_corrections(df)

        return df

    def _process_multi_planet_systems(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add features for multi-planetary systems."""
        # Count planets per host (simplified - would need proper grouping)
        # For now, just add a flag for known multi-planet systems

        # Systems with multiple entries (same hostname)
        if "hostname" in df.columns:
            host_counts = df.group_by("hostname").count()
            host_counts = host_counts.rename({"count": "n_planets_system"})
            df = df.join(host_counts, on="hostname", how="left")

        return df

    def _add_derived_orbital_properties(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived orbital mechanics properties."""

        # Orbital velocity (simplified circular orbit)
        if all(col in df.columns for col in ["pl_orbsmax", "st_mass"]):
            # v = sqrt(GM/r)
            sma_m = df["pl_orbsmax"] * 1.496e11  # AU to meters
            m_star_kg = df["st_mass"] * 1.989e30  # Solar masses to kg
            G = 6.674e-11  # Gravitational constant

            v_orbit = (G * m_star_kg / sma_m) ** 0.5  # m/s
            v_orbit_km_s = v_orbit / 1000.0  # km/s

            df = df.with_columns(pl.Series("orbital_velocity_km_s", v_orbit_km_s))

        # Hill sphere radius
        if all(col in df.columns for col in ["pl_orbsmax", "pl_masse", "st_mass"]):
            sma = df["pl_orbsmax"]
            m_planet = (
                df["pl_masse"] * 5.972e24 / 1.989e30
            )  # Earth masses to Solar masses
            m_star = df["st_mass"]

            r_hill = sma * (m_planet / (3 * m_star)) ** (1 / 3)

            df = df.with_columns(pl.Series("hill_radius_au", r_hill))

        return df

    def _add_stellar_classification(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add stellar classification based on temperature and mass."""

        if "st_teff" in df.columns and "st_mass" in df.columns:
            teff = df["st_teff"]
            mass = df["st_mass"]

            # stellar classification
            def classify_star(temp, m):
                if temp >= 30000:
                    return "O"
                elif temp >= 10000:
                    return "B"
                elif temp >= 7500:
                    return "A"
                elif temp >= 6000:
                    return "F"
                elif temp >= 5200:
                    return "G"
                elif temp >= 3700:
                    return "K"
                else:
                    return "M"

            # Apply classification
            spectral_types = []
            for i in range(len(df)):
                temp_val = teff[i] if not np.isnan(teff[i]) else 5772
                mass_val = mass[i] if not np.isnan(mass[i]) else 1.0
                spectral_types.append(classify_star(temp_val, mass_val))

            df = df.with_columns(pl.Series("stellar_type", spectral_types))

        return df

    def _add_habitability_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add comprehensive habitability indicators."""

        # Overall Earth Similarity Index
        if all(col in df.columns for col in ["pl_rade", "pl_eqt"]):
            radius = df["pl_rade"]
            temp = df["pl_eqt"]

            # Radius component
            radius_esi = 1.0 - abs(radius - 1.0) / (radius + 1.0)

            # Temperature component
            temp_esi = 1.0 - abs(temp - 288.0) / (temp + 288.0)

            # Combined ESI
            esi = (radius_esi * temp_esi) ** 0.5

            df = df.with_columns(pl.Series("earth_similarity_index", esi))

        return df

    def _apply_discovery_bias_corrections(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply weights to correct for discovery method biases."""

        if "discoverymethod" in df.columns:
            # Add bias correction weights
            weights = []
            for method in df["discoverymethod"]:
                weight = self.method_weights.get(method, 1.0)
                weights.append(weight)

            df = df.with_columns(pl.Series("discovery_weight", weights))

        return df

    def create_tensordict(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Create comprehensive TensorDict for exoplanet systems."""
        tensors = super().create_tensordict(df)

        # Add exoplanet-specific tensors

        # Planet orbital parameters
        if all(col in df.columns for col in ["pl_orbper", "pl_orbsmax", "pl_orbeccen"]):
            orbital_data = torch.stack(
                [
                    torch.tensor(df["pl_orbper"].to_numpy(), dtype=torch.float32),
                    torch.tensor(df["pl_orbsmax"].to_numpy(), dtype=torch.float32),
                    torch.tensor(df["pl_orbeccen"].to_numpy(), dtype=torch.float32),
                ],
                dim=1,
            )

            # Handle NaN values
            orbital_data = torch.nan_to_num(orbital_data, nan=0.0)

            tensors["orbital_parameters"] = orbital_data

        # Habitability scores
        if "earth_similarity_index" in df.columns:
            esi = torch.tensor(
                df["earth_similarity_index"].to_numpy(), dtype=torch.float32
            )
            tensors["habitability_scores"] = esi

        # Discovery metadata
        if "discoverymethod" in df.columns:
            methods = df["discoverymethod"].to_list()
            years = df["disc_year"].to_numpy() if "disc_year" in df.columns else None

            tensors["discovery_metadata"] = {
                "methods": methods,
                "years": years,
                "weights": df.select("discovery_weight").to_series().to_list()
                if "discovery_weight" in df.columns
                else [1.0] * len(df),
            }

        # Update metadata
        tensors["meta"].update(
            {
                "survey_type": "exoplanet_hosts",
                "has_orbital_data": "pl_orbper" in df.columns,
                "has_habitability_data": "earth_similarity_index" in df.columns,
                "discovery_methods": list(set(df["discoverymethod"].to_list()))
                if "discoverymethod" in df.columns
                else [],
            }
        )

        return tensors
